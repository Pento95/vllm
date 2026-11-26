# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.dflash2.config import DFlash2SpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.dspark.config import DSparkSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.speculator_config import (
    SpeculatorConfig,
    _get_draft_declared_config_class,
)

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


class Inkling2DSparkSpeculatorConfig(DSparkSpeculatorConfig):
    speculator_class = "vllm.models.inkling2.speculator.Inkling2DSparkSpeculator"

    @classmethod
    def adapt_draft_model_config(cls, spec: "SpeculativeConfig") -> None:
        super().adapt_draft_model_config(spec)
        if current_platform.is_device_capability_family(120):
            # The target's clipped, unit-scale FP8 cache does not describe the
            # draft's unnormalized values. Keep the draft cache in BF16.
            if spec.kv_cache_dtype is None:
                spec.kv_cache_dtype = "auto"
            elif spec.kv_cache_dtype not in ("auto", "bfloat16"):
                raise ValueError("Inkling2 DSpark on SM12 requires BF16 draft KV")


class Inkling2DFlash2SpeculatorConfig(DFlash2SpeculatorConfig):
    """DFlash2 drafts of Inkling2 (e.g. incoai/Inkling-2-Mini-DFlash2); other
    DFlash drafts keep the speculator they declare."""

    speculator_class = "vllm.models.inkling2.speculator.Inkling2DFlash2Speculator"

    @staticmethod
    def _is_dflash2(spec: "SpeculativeConfig") -> bool:
        draft = spec.draft_model_config
        if draft is None:
            return False
        return "DFlash2DraftModel" in (
            getattr(draft.hf_config, "architectures", None) or []
        )

    @classmethod
    def specialize(cls, spec: "SpeculativeConfig") -> type[SpeculatorConfig]:
        if cls._is_dflash2(spec):
            return cls
        # What the generic DFlash config resolves to without this one.
        declared = _get_draft_declared_config_class(spec)
        if declared is not None and issubclass(declared, DFlashSpeculatorConfig):
            return declared
        return DFlashSpeculatorConfig

    @classmethod
    def adapt_draft_model_config(cls, spec: "SpeculativeConfig") -> None:
        if not cls._is_dflash2(spec):
            DFlashSpeculatorConfig.adapt_draft_model_config(spec)
            return
        super().adapt_draft_model_config(spec)
        draft = spec.draft_model_config.hf_config
        dflash_config = draft.dflash_config
        text = spec.target_model_config.hf_text_config
        loop = text.loop_config
        if loop is not None and text.num_loop_passes != loop["num_passes"]:
            raise ValueError(
                "Inkling2 DFlash2 drafts read hidden states at fixed layer visits; "
                f"serve them at the checkpoint's {loop['num_passes']} loop passes"
            )
        if (
            draft.hidden_size != text.hidden_size
            or getattr(draft, "num_target_layers", text.num_layer_visits)
            != text.num_layer_visits
        ):
            raise ValueError(
                "This DFlash2 draft was trained for another target: its "
                "hidden_size or num_target_layers differs from Inkling2's "
                f"({text.hidden_size}, {text.num_layer_visits} layer visits)"
            )
        features = list(dflash_config.get("target_layer_ids") or [])
        if (
            len(features) < 2
            or features[0] < 0
            or features != sorted(set(features))
            or features[-1] != text.num_layer_visits
        ):
            raise ValueError(
                "Inkling2 DFlash2 target_layer_ids must be sorted target layer "
                "visits followed by num_layer_visits (the final-norm output)"
            )
        # The target captures the input of each listed visit and appends its
        # final-norm output, which the last id names: no DFlash +1 shift.
        draft.eagle_aux_hidden_state_layer_ids = features
        num_spec = int(dflash_config["block_size"]) - 1
        if spec.num_speculative_tokens is None:
            spec.num_speculative_tokens = num_spec
        elif spec.num_speculative_tokens != num_spec:
            raise ValueError(
                f"Inkling2 DFlash2 requires num_speculative_tokens = {num_spec} "
                "(the draft's block_size - 1)"
            )
