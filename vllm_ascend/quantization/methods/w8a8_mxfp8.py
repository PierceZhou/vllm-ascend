#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import importlib
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
import torch_npu
from vllm.config import CompilationMode, get_current_vllm_config
from vllm.logger import logger
from vllm.model_executor.utils import replace_parameter
from vllm.utils.math_utils import cdiv

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.device.mxfp_compat import (
    FLOAT8_E8M0FNU_DTYPE,
    ensure_mxfp8_linear_available,
    ensure_mxfp8_moe_available,
)
from vllm_ascend.flash_common3_context import get_flash_common3_context
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input

from .base import AscendLinearScheme, AscendMoEScheme, QuantType, get_moe_num_logical_experts
from .registry import register_scheme


_ONLINE_WEIGHT_DTYPES = (torch.float16, torch.bfloat16)
_ONLINE_COMPUTED_SCALE_NAMES = frozenset(
    {
        'weight_scale',
        'w13_weight_scale',
        'w2_weight_scale',
    }
)


def _register_online_generated_scales_for_layerwise_reload() -> dict[str, tuple[str, ...]]:
    '''Exclude online MXFP8 scales from checkpoint weight-load accounting.

    Online MXFP8 checkpoints contain only FP16/BF16 weights. Their E8M0 scales
    are produced by ``npu_dynamic_mx_quant`` after a layer's weight has loaded,
    so waiting for scale tensors from the checkpoint prevents vLLM's layerwise
    loader from finalizing any linear layer. The buffered BF16 weights then
    accumulate across the whole model.

    vLLM releases use different skip-set names and import styles. Update every
    live binding used by metadata restoration and load accounting, including
    module-local copies and immutable sets. The registration is process-local,
    idempotent, and fails early when the installed API cannot be patched.
    '''
    module_names = (
        'vllm.model_executor.model_loader.reload.meta',
        'vllm.model_executor.model_loader.reload.layerwise',
        'vllm.model_executor.model_loader.reload.utils',
    )
    skip_set_names = ('SKIP_LOAD_TENSORS', 'SKIP_TENSORS')
    registered: dict[str, tuple[str, ...]] = {}

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue

        updated_names = []
        for skip_set_name in skip_set_names:
            current = getattr(module, skip_set_name, None)
            if current is None:
                continue

            try:
                updated = set(current) | set(_ONLINE_COMPUTED_SCALE_NAMES)
            except TypeError:
                continue
            # Always rebind. vLLM 0.23 aliases SKIP_TENSORS across meta.py and
            # utils.py; mutating the shared object would make a unit of the
            # registration impossible to roll back and obscures which binding
            # the accounting function actually reads.
            setattr(module, skip_set_name, updated)

            if _ONLINE_COMPUTED_SCALE_NAMES.issubset(updated):
                updated_names.append(skip_set_name)

        if updated_names:
            registered[module_name] = tuple(updated_names)

    try:
        reload_utils = importlib.import_module(
            'vllm.model_executor.model_loader.reload.utils'
        )
        get_layer_size = reload_utils.get_layer_size
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            'Online MXFP8 could not locate vLLM layerwise load accounting.'
        ) from exc

    # Validate the behavior used by layerwise.py, rather than assuming a
    # particular module exposes a particular set name. This catches copied or
    # rebound skip sets before a real model update allocates checkpoint buffers.
    probe = torch.nn.Module()
    for name in ('weight', *_ONLINE_COMPUTED_SCALE_NAMES):
        probe.register_parameter(
            name,
            torch.nn.Parameter(torch.empty(1, device='meta'), requires_grad=False),
        )
    if get_layer_size(probe) != probe.weight.numel():
        raise RuntimeError(
            'Online MXFP8 computed scale tensors are still included in vLLM '
            'layerwise load accounting. BF16 checkpoint weights would '
            'accumulate on device until finish_weight_update().'
        )

    logger.info_once(
        'Online MXFP8 layerwise reload skips computed scales via %s.',
        ', '.join(
            f'{module_name.rsplit(".", 1)[-1]}.{name}'
            for module_name, names in registered.items()
            for name in names
        ),
    )
    return registered

def _to_mxfp8_scale_storage(scale: torch.Tensor, context: str) -> torch.Tensor:
    '''Return the E8M0 scale in the uint8 storage expected by Ascend kernels.'''
    scale = scale.contiguous()
    if scale.dtype == torch.uint8:
        return scale
    if scale.element_size() != 1:
        raise RuntimeError(
            f'{context} produced an unsupported scale dtype {scale.dtype}; '
            'MXFP8 E8M0 scales must use one byte per element.'
        )
    return scale.view(torch.uint8)


def _quantize_online_weight(
    weight: torch.Tensor,
    group_size: int,
    context: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    '''Quantize one FP16/BF16 weight tensor to E4M3 plus block E8M0 scales.'''
    if weight.dtype not in _ONLINE_WEIGHT_DTYPES:
        raise TypeError(
            f'{context} expects an FP16/BF16 checkpoint weight before online '
            f'MXFP8 quantization, but received {weight.dtype}.'
        )
    if weight.shape[-1] % group_size != 0:
        raise ValueError(
            f'{context} requires the input dimension ({weight.shape[-1]}) to '
            f'be divisible by the MXFP8 group size ({group_size}).'
        )

    original_shape = tuple(weight.shape)
    quantization_input = weight.contiguous().view(-1, original_shape[-1])
    quantized_weight, weight_scale = torch_npu.npu_dynamic_mx_quant(
        quantization_input,
        dst_type=torch.float8_e4m3fn,
    )
    expected_flat_scale_shape = (
        quantization_input.shape[0],
        original_shape[-1] // group_size,
    )
    if tuple(quantized_weight.shape) != tuple(quantization_input.shape):
        raise RuntimeError(
            f'{context} returned weight shape {tuple(quantized_weight.shape)}, '
            f'expected {tuple(quantization_input.shape)}.'
        )
    # torch_npu versions expose the E8M0 scale in either logical layout
    # [rows, groups] or packed layout [rows, ceil(groups / 2), 2].  Normalize
    # both forms here; process_weights_after_loading() performs the final
    # transpose into the layout consumed by npu_quant_matmul.
    returned_scale_shape = tuple(weight_scale.shape)
    weight_scale = _to_mxfp8_scale_storage(weight_scale, context)
    scale_rows, scale_groups = expected_flat_scale_shape
    try:
        flat_scale = weight_scale.reshape(scale_rows, -1)
    except RuntimeError as exc:
        raise RuntimeError(
            f'{context} returned scale shape {returned_scale_shape}, which '
            f'cannot be normalized to {expected_flat_scale_shape}.'
        ) from exc

    padded_scale_groups = cdiv(scale_groups, 2) * 2
    if flat_scale.shape[1] not in (scale_groups, padded_scale_groups):
        raise RuntimeError(
            f'{context} returned scale shape {returned_scale_shape} with '
            f'{flat_scale.shape[1]} values per row; expected {scale_groups} '
            f'logical values or {padded_scale_groups} packed values.'
        )
    weight_scale = flat_scale[:, :scale_groups]
    expected_scale_shape = (*original_shape[:-1], original_shape[-1] // group_size)
    quantized_weight = quantized_weight.view(original_shape).contiguous()
    weight_scale = weight_scale.reshape(expected_scale_shape).contiguous()
    return quantized_weight, weight_scale


def _allocate_reload_parameter(
    layer: torch.nn.Module,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    '''Replace a runtime MXFP8 parameter with an empty checkpoint-format one.'''
    current = getattr(layer, name)
    checkpoint_tensor = torch.empty(shape, dtype=dtype, device=current.device)
    replace_parameter(layer, name, checkpoint_tensor)


def _record_or_validate_runtime_layout(
    layer: torch.nn.Module,
    tensor_names: tuple[str, ...],
    context: str,
) -> None:
    """Require a stable post-load contract for graph-captured tensors."""
    signature = {
        name: (tuple(getattr(layer, name).shape), tuple(getattr(layer, name).stride()), getattr(layer, name).dtype)
        for name in tensor_names
    }
    expected = getattr(layer, '_mxfp8_runtime_layout', None)
    if expected is None:
        layer._mxfp8_runtime_layout = signature
        return
    if signature != expected:
        raise RuntimeError(
            f'{context} changed MXFP8 runtime layout: '
            f'expected={expected}, actual={signature}. ACL Graph replay is unsafe.'
        )


@register_scheme("W8A8_MXFP8", "linear")
class AscendW8A8MXFP8DynamicLinearMethod(AscendLinearScheme):
    """Linear method for Ascend W8A8_MXFP8 (Microscaling FP8) quantization.

    This scheme uses microscaling FP8 quantization with per-group scales.
    The activation is dynamically quantized to FP8 (E4M3FN format) with
    microscaling, and weights are stored in FP8 format with per-group scales.
    """

    model_dtype = None

    def __init__(self, online_quantization: bool = False):
        ensure_mxfp8_linear_available("W8A8_MXFP8 linear quantization")
        self.online_quantization = online_quantization
        if self.online_quantization:
            _register_online_generated_scales_for_layerwise_reload()
        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        if self.online_quantization:
            return {'weight': torch.empty(output_size, input_size, dtype=params_dtype)}
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn)}
        return params_dict

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, cdiv(input_size, self.group_size), dtype=torch.uint8)
        return params_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        if isinstance(x, tuple):
            quantized_x, pertoken_scale = x
            original_shape = quantized_x.shape
            output_dtype = torch.bfloat16
        else:
            # reshape x for Qwen VL models
            original_shape = x.shape
            if x.dim() > 2:
                x = x.view(-1, x.shape[-1])
            quantized_x, pertoken_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
            output_dtype = x.dtype

        if bias is not None and bias.dtype != torch.float32:
            bias = bias.to(torch.float32)

        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            pertoken_scale=pertoken_scale,
            pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            bias=bias,
            output_dtype=output_dtype,
            group_sizes=[1, 1, self.group_size],
        )
        # reshape output for Qwen VL models
        if len(original_shape) > 2:
            output = output.view(*original_shape[:-1], -1)

        return output

    def process_weights_after_loading(self, layer):
        """Process weights after loading for MXFP8 inference.

        This method transforms weights for NPU MXFP8 computation:
        - weight: (output_size, input_size) -> (input_size, output_size)
        - weight_scale: (n_dim, k_dim) -> (k_dim//2, n_dim, 2)

        For RL training scenarios where weights need to be reloaded multiple times,
        this method stores original shapes and can be called multiple times safely.
        Use restore_weights_for_rl_loading() before weight reload, then call this
        method again after loading.
        """

        # Layerwise reload restores the checkpoint dtype but keeps module flags.
        if self.online_quantization and layer.weight.dtype in _ONLINE_WEIGHT_DTYPES:
            layer._mxfp8_transformed = False
        if getattr(layer, "_mxfp8_transformed", False):
            return

        if self.online_quantization:
            if not hasattr(layer, '_mxfp8_online_checkpoint'):
                layer._mxfp8_online_checkpoint = {
                    'weight': (tuple(layer.weight.shape), layer.weight.dtype),
                    'weight_scale': tuple(layer.weight_scale.shape),
                }
            quantized_weight, weight_scale = _quantize_online_weight(
                layer.weight.data,
                self.group_size,
                'Ascend online MXFP8 linear quantization',
            )
            replace_parameter(layer, 'weight', quantized_weight)
            replace_parameter(layer, 'weight_scale', weight_scale)

        # Store original shapes for RL weight reloading
        # Only store on first call (when shapes are in original format)
        if not hasattr(layer, "_mxfp8_original_shapes"):
            layer._mxfp8_original_shapes = {
                "weight": tuple(layer.weight.data.shape),
                "weight_scale": tuple(layer.weight_scale.data.shape),
            }

        n_dim, k_dim = layer.weight_scale.data.shape
        # Shape should be padded if it cannot be divided by 2
        if layer.weight_scale.data.shape[-1] % 2 != 0:
            layer.weight_scale.data = F.pad(layer.weight_scale.data, (0, 1), mode="constant", value=0)
            layer.weight_scale.data = layer.weight_scale.data.reshape(n_dim, k_dim // 2 + 1, 2)
        else:
            layer.weight_scale.data = layer.weight_scale.data.reshape(n_dim, k_dim // 2, 2)
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1).contiguous()

        if self.online_quantization:
            _record_or_validate_runtime_layout(
                layer,
                ('weight', 'weight_scale'),
                'Ascend online MXFP8 linear quantization',
            )

        # Mark as transformed
        layer._mxfp8_transformed = True

    def restore_weights_for_rl_loading(self, layer):
        """Restore weights to original shapes for RL weight reloading.

        This method must be called BEFORE model.load_weights() in RL training
        loops to restore the tensors to their original shapes that the weight
        loader expects.

        After weight loading, call process_weights_after_loading() again to
        re-apply the MXFP8 transformations.

        Shape transformations reversed:
        - weight: (input_size, output_size) -> (output_size, input_size)
        - weight_scale: (k_dim//2, n_dim, 2) -> (n_dim, k_dim)
        """

        if not getattr(layer, "_mxfp8_transformed", False):
            # Not transformed, nothing to restore
            return

        if self.online_quantization:
            checkpoint_state = layer._mxfp8_online_checkpoint
            weight_shape, weight_dtype = checkpoint_state['weight']
            _allocate_reload_parameter(layer, 'weight', weight_shape, weight_dtype)
            _allocate_reload_parameter(
                layer,
                'weight_scale',
                checkpoint_state['weight_scale'],
                torch.uint8,
            )
            layer._mxfp8_transformed = False
            return

        if not hasattr(layer, "_mxfp8_original_shapes"):
            err_msg = (
                "[vllm-ascend/W8A8_MXFP8] Cannot restore weights: original "
                "shapes not recorded. "
                "This should not happen if process_weights_after_loading was called first."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        orig_shapes = layer._mxfp8_original_shapes
        orig_scale_shape = orig_shapes["weight_scale"]

        # Restore weight: (input_size, output_size) -> (output_size, input_size)
        target_weight = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = layer.weight.data.transpose(0, 1)
        layer.weight.data.copy_(target_weight)

        # Restore weight_scale: (k_dim//2, n_dim, 2) -> (n_dim, k_dim)
        # Current shape: (k_dim//2, n_dim, 2)
        # Target shape: (n_dim, k_dim)
        target_scale = layer.weight_scale.data.transpose(0, 1).reshape(orig_scale_shape).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1).reshape(orig_scale_shape)
        layer.weight_scale.data.copy_(target_scale)

        # Mark as not transformed (ready for weight loading)
        layer._mxfp8_transformed = False


@register_scheme("W8A8_MXFP8", "moe")
class AscendW8A8MXFP8DynamicFusedMoEMethod(AscendMoEScheme):
    """FusedMoe method for Ascend W8A8_DYNAMIC."""

    model_dtype = None
    quant_type: QuantType = QuantType.MXFP8

    def __init__(self, online_quantization: bool = False):
        ensure_mxfp8_moe_available("W8A8_MXFP8 MoE quantization")

        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)
        self.online_quantization = online_quantization
        if self.online_quantization:
            _register_online_generated_scales_for_layerwise_reload()
        ascend_config = get_ascend_config()
        self.use_aclgraph = (
            vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and not vllm_config.model_config.enforce_eager
        )
        self.dynamic_eplb = ascend_config.eplb_config.dynamic_eplb
        self.multistream_overlap_gate = ascend_config.multistream_overlap_gate

    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        if self.online_quantization:
            return {
                'w13_weight': torch.empty(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_sizes,
                    dtype=params_dtype,
                ),
                'w2_weight': torch.empty(
                    num_experts,
                    hidden_sizes,
                    intermediate_size_per_partition,
                    dtype=params_dtype,
                ),
            }
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes, dtype=torch.float8_e4m3fn
        )
        param_dict["w2_weight"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition, dtype=torch.float8_e4m3fn
        )
        return param_dict

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes // self.group_size, dtype=torch.uint8
        )

        param_dict["w2_weight_scale"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition // self.group_size, dtype=torch.uint8
        )
        return param_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = True,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: Any | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
        tid2eid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_shared_experts = getattr(layer, "n_shared_experts", 0)
        if num_shared_experts is None:
            num_shared_experts = 0
        num_logical_experts = get_moe_num_logical_experts(
            layer,
            num_experts,
            global_redundant_expert_num=global_redundant_expert_num,
            num_shared_experts=num_shared_experts,
        )
        assert router_logits.shape[1] == num_logical_experts, "Number of global experts mismatch (excluding redundancy)"
        if self.multistream_overlap_gate:
            fc3_context = get_flash_common3_context()
            assert fc3_context is not None
            topk_weights = fc3_context.topk_weights
            topk_ids = fc3_context.topk_ids
        else:
            topk_weights, topk_ids = select_experts(
                hidden_states=x,
                router_logits=router_logits,
                top_k=top_k,
                use_grouped_topk=use_grouped_topk,
                renormalize=renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                routed_scaling_factor=routed_scaling_factor,
                e_score_correction_bias=e_score_correction_bias,
                num_experts=num_logical_experts,
                tid2eid=tid2eid,
            )

        if topk_weights is None or topk_ids is None:
            raise RuntimeError("topk_weights and topk_ids must be set before fused MoE execution.")

        # this is a naive implementation for experts load balance so as
        # to avoid accumulating too much tokens on a single rank.
        # currently it is only activated when doing profile runs.
        if enable_force_load_balance:
            random_matrix = torch.rand(topk_ids.size(0), num_logical_experts, device=topk_ids.device)
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        if x.dtype not in [torch.float8_e4m3fn]:
            topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                quant_type=self.quant_type,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
                mxfp_act_quant_type=torch.float8_e4m3fn,
                mxfp_weight_quant_type=torch.float8_e4m3fn,
                mxfp_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                mxfp_per_token_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                mxfp_use_bf16=(x.dtype in [torch.bfloat16, torch.float8_e4m3fn]),
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                swiglu_limit=layer.swiglu_limit,
            )
        )

    def process_weights_after_loading(self, layer):
        """Process weights after loading for MXFP8 inference.

        This method transforms weights for NPU MXFP8 computation:
        - w13_weight: (g_num, n_size, k_size) -> (g_num, k_size, n_size)
        - w2_weight: (g_num, n_size, k_size) -> (g_num, k_size, n_size)
        - w13_weight_scale: (g_num, n_size, k_size) -> (g_num, k_size//2, n_size, 2)
        - w2_weight_scale: (g_num, n_size, k_size) -> (g_num, k_size//2, n_size, 2)

        For RL training scenarios where weights need to be reloaded multiple times,
        this method stores original shapes and can be called multiple times safely.
        Use restore_weights_for_rl_loading() before weight reload, then call this
        method again after loading.
        """

        # Layerwise reload restores the checkpoint dtype but keeps module flags.
        if self.online_quantization and layer.w13_weight.dtype in _ONLINE_WEIGHT_DTYPES:
            layer._mxfp8_transformed = False
        if getattr(layer, "_mxfp8_transformed", False):
            return

        # Store original shapes for RL weight reloading
        # Only store on first call (when shapes are in original format)
        if not hasattr(layer, "_mxfp8_original_shapes"):
            layer._mxfp8_original_shapes = {
                "w13_weight": tuple(layer.w13_weight.data.shape),
                "w13_weight_scale": tuple(layer.w13_weight_scale.data.shape),
                "w2_weight": tuple(layer.w2_weight.data.shape),
                "w2_weight_scale": tuple(layer.w2_weight_scale.data.shape),
            }

        if self.online_quantization:
            if not hasattr(layer, '_mxfp8_online_checkpoint'):
                layer._mxfp8_online_checkpoint = {
                    'w13_weight': (tuple(layer.w13_weight.shape), layer.w13_weight.dtype),
                    'w13_weight_scale': tuple(layer.w13_weight_scale.shape),
                    'w2_weight': (tuple(layer.w2_weight.shape), layer.w2_weight.dtype),
                    'w2_weight_scale': tuple(layer.w2_weight_scale.shape),
                }
            w13_weight, w13_weight_scale = _quantize_online_weight(
                layer.w13_weight.data,
                self.group_size,
                'Ascend online MXFP8 MoE w13 quantization',
            )
            w2_weight, w2_weight_scale = _quantize_online_weight(
                layer.w2_weight.data,
                self.group_size,
                'Ascend online MXFP8 MoE w2 quantization',
            )
            replace_parameter(layer, 'w13_weight', w13_weight)
            replace_parameter(layer, 'w13_weight_scale', w13_weight_scale)
            replace_parameter(layer, 'w2_weight', w2_weight)
            replace_parameter(layer, 'w2_weight_scale', w2_weight_scale)

        g_num, n_size, k_size = layer.w13_weight_scale.shape
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.reshape(g_num, n_size, k_size // 2, 2)
        g_num, n_size, k_size = layer.w2_weight_scale.shape
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.reshape(g_num, n_size, k_size // 2, 2)
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.transpose(1, 2)
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.transpose(1, 2)

        if self.online_quantization:
            _record_or_validate_runtime_layout(
                layer,
                ('w13_weight', 'w13_weight_scale', 'w2_weight', 'w2_weight_scale'),
                'Ascend online MXFP8 MoE quantization',
            )

        # Mark as transformed
        layer._mxfp8_transformed = True

    def restore_weights_for_rl_loading(self, layer):
        """Restore weights to original shapes for RL weight reloading.

        This method must be called BEFORE model.load_weights() in RL training
        loops to restore the tensors to their original shapes that the weight
        loader expects.

        After weight loading, call process_weights_after_loading() again to
        re-apply the MXFP8 transformations.

        Shape transformations reversed:
        - w13_weight: (g_num, k_size, n_size) -> (g_num, n_size, k_size)
        - w2_weight: (g_num, k_size, n_size) -> (g_num, n_size, k_size)
        - w13_weight_scale: (g_num, k_size//2, n_size, 2) -> (g_num, n_size, k_size)
        - w2_weight_scale: (g_num, k_size//2, n_size, 2) -> (g_num, n_size, k_size)
        """

        if not getattr(layer, "_mxfp8_transformed", False):
            # Not transformed, nothing to restore
            return

        if not hasattr(layer, "_mxfp8_original_shapes"):
            err_msg = (
                "[vllm-ascend/W8A8_MXFP8] Cannot restore weights: original "
                "shapes not recorded. "
                "This should not happen if process_weights_after_loading was called first."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        orig_shapes = layer._mxfp8_original_shapes

        if self.online_quantization:
            checkpoint_state = layer._mxfp8_online_checkpoint
            for weight_name, scale_name in (
                ('w13_weight', 'w13_weight_scale'),
                ('w2_weight', 'w2_weight_scale'),
            ):
                weight_shape, weight_dtype = checkpoint_state[weight_name]
                _allocate_reload_parameter(layer, weight_name, weight_shape, weight_dtype)
                _allocate_reload_parameter(
                    layer,
                    scale_name,
                    checkpoint_state[scale_name],
                    torch.uint8,
                )
            layer._mxfp8_transformed = False
            return

        def _restore(weight_key: str, scale_key: str):
            """Helper to restore a single MoE weight and its scale using safe memory copies."""
            # --- 1. Restore Weight ---
            weight_tensor = getattr(layer, weight_key)
            target_weight = weight_tensor.data.transpose(1, 2).contiguous()
            weight_tensor.data = weight_tensor.data.transpose(1, 2)
            weight_tensor.data.copy_(target_weight)

            # --- 2. Restore Weight Scale ---
            scale_tensor = getattr(layer, scale_key)
            orig_scale_shape = orig_shapes[scale_key]

            target_scale = scale_tensor.data.transpose(1, 2).reshape(orig_scale_shape).contiguous()
            scale_tensor.data = scale_tensor.data.transpose(1, 2).view(orig_scale_shape)
            scale_tensor.data.copy_(target_scale)

        _restore("w13_weight", "w13_weight_scale")
        _restore("w2_weight", "w2_weight_scale")

        # Mark as not transformed (ready for weight loading)
        layer._mxfp8_transformed = False
