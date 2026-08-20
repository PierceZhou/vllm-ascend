'''Ascend online MXFP8 quantization configuration.

This configuration accepts an FP16/BF16 checkpoint (including weights pushed
by an RL trainer) and quantizes weights on the rollout NPU after loading.
'''

from typing import Any

import torch
from vllm.logger import logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase

from vllm_ascend.utils import (
    AscendDeviceType,
    MXFP8_METHOD,
    get_ascend_device_type,
    vllm_version_is,
)

if vllm_version_is('0.23.0'):
    from vllm.model_executor.layers.fused_moe import FusedMoE
else:
    from vllm.model_executor.layers.fused_moe import MoERunner


MXFP8_GROUP_SIZE = 32


def _is_fused_moe_layer(layer: torch.nn.Module) -> bool:
    if vllm_version_is('0.23.0'):
        return isinstance(layer, FusedMoE)
    return isinstance(layer, MoERunner)


def _is_ignored_layer(prefix: str, ignored_layers: list[str]) -> bool:
    return any(prefix == ignored or prefix.startswith(f'{ignored}.') for ignored in ignored_layers)


@register_quantization_config(MXFP8_METHOD)
class AscendMxfp8Config(QuantizationConfig):
    '''Online block-32 MXFP8 configuration backed by Ascend NPU operators.'''

    def __init__(
        self,
        ignored_layers: list[str] | None = None,
        group_size: int = MXFP8_GROUP_SIZE,
    ) -> None:
        super().__init__()
        if get_ascend_device_type() != AscendDeviceType.A5:
            raise RuntimeError('Ascend online MXFP8 is supported only on Ascend A5.')
        if group_size != MXFP8_GROUP_SIZE:
            raise ValueError(
                f'Ascend online MXFP8 only supports group_size={MXFP8_GROUP_SIZE}, '
                f'but received {group_size}.'
            )
        self.ignore = ignored_layers or []
        self.group_size = group_size
        self.quant_description = {'group_size': group_size}

    def __repr__(self) -> str:
        return 'AscendMxfp8Config:\n' + super().__repr__()

    @classmethod
    def get_name(cls) -> str:
        return MXFP8_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError('Ascend hardware does not use CUDA compute capability.')

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'AscendMxfp8Config':
        activation_scheme = config.get('activation_scheme', 'dynamic')
        if activation_scheme != 'dynamic':
            raise ValueError('Ascend MXFP8 only supports dynamic activation quantization.')
        ignored_layers = (
            config.get('ignored_layers')
            or config.get('modules_to_not_convert')
            or config.get('ignore')
            or []
        )
        return cls(
            ignored_layers=list(ignored_layers),
            group_size=int(config.get('group_size', MXFP8_GROUP_SIZE)),
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        tid2eid=None,
    ) -> QuantizeMethodBase | None:
        from .method_adapters import AscendFusedMoEMethod, AscendLinearMethod
        from .methods.w8a8_mxfp8 import (
            AscendW8A8MXFP8DynamicFusedMoEMethod,
            AscendW8A8MXFP8DynamicLinearMethod,
        )

        if isinstance(layer, LinearBase):
            if _is_ignored_layer(prefix, self.ignore):
                from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

                return AscendUnquantizedLinearMethod()
            layer.ascend_quant_method = MXFP8_METHOD
            logger.info_once('Using vLLM Ascend online MXFP8 quantization.')
            return AscendLinearMethod(
                AscendW8A8MXFP8DynamicLinearMethod(online_quantization=True)
            )

        if _is_fused_moe_layer(layer):
            if _is_ignored_layer(prefix, self.ignore):
                from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod

                return AscendUnquantizedFusedMoEMethod(layer.moe_config)
            layer.ascend_quant_method = MXFP8_METHOD
            logger.info_once('Using vLLM Ascend online MXFP8 MoE quantization.')
            return AscendFusedMoEMethod(
                AscendW8A8MXFP8DynamicFusedMoEMethod(online_quantization=True),
                layer.moe_config,
                tid2eid=tid2eid,
            )
        return None
