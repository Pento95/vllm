# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 with DFlash2 drafts: the speculator chosen for the target, the
capture ids it hands the target, the draft's input embeddings and the
selector's soft cap."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
from vllm.model_executor.models.qwen3_dflash2 import (
    CandidateSelector,
    _score_edges_reference,
)
from vllm.models.inkling2.speculator_config import Inkling2DFlash2SpeculatorConfig
from vllm.v1.worker.gpu.spec_decode import speculator_config as sc
from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.dflash2.config import DFlash2SpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
)

_SPEC_DECODE = "vllm.v1.worker.gpu.spec_decode"
_DFLASH2 = f"{_SPEC_DECODE}.dflash2.config.DFlash2SpeculatorConfig"
_DFLASH = f"{_SPEC_DECODE}.dflash.config.DFlashSpeculatorConfig"


def _spec(text_model_type="inkling2_loop_text", ids=(5, 23, 41, 48), dflash2=True):
    # Inkling 2 Mini: 24 layers, a 12-layer core run 3 times, 48 visits.
    text = SimpleNamespace(
        model_type=text_model_type,
        loop_config={"num_passes": 3},
        num_loop_passes=3,
        num_layer_visits=48,
        hidden_size=2304,
    )
    draft_hf = SimpleNamespace(
        architectures=["DFlash2DraftModel" if dflash2 else "DFlashDraftModel"],
        hidden_size=2304,
        num_target_layers=48,
        dflash_config={"target_layer_ids": list(ids), "block_size": 8},
    )
    return SimpleNamespace(
        method="dflash",
        custom_method=None,
        num_speculative_tokens=None,
        target_model_config=SimpleNamespace(hf_text_config=text),
        draft_model_config=SimpleNamespace(
            hf_config=draft_hf,
            speculator_config_class=_DFLASH2 if dflash2 else _DFLASH,
        ),
    )


def _resolve(spec, method):
    """The config used to set up the draft, then the one that builds the
    speculator (as SpeculativeConfig runs them)."""
    spec.method = method
    configured = sc.get_method_config_class(spec)
    setup = sc._resolve_checkpoint_config_class(spec, configured)
    spec.method = "dflash"
    return setup, sc.get_speculator_config_class(spec)


@pytest.mark.parametrize("method", ["dflash", None])
@pytest.mark.parametrize("text_model_type", ["inkling2_model", "inkling2_loop_text"])
def test_inkling2_target_specializes_dflash2_drafts(text_model_type, method):
    setup, speculator = _resolve(_spec(text_model_type), method)
    assert setup is speculator is Inkling2DFlash2SpeculatorConfig


@pytest.mark.parametrize("method", ["dflash", None])
def test_other_drafts_and_targets_keep_their_speculator(method):
    # A DFlash (v1) draft of an Inkling2 target, as before this config.
    _, speculator = _resolve(_spec(dflash2=False), method)
    assert speculator is DFlashSpeculatorConfig
    spec = _spec(dflash2=False)
    with mock.patch.object(
        DFlashSpeculatorConfig, "adapt_draft_model_config"
    ) as generic:
        Inkling2DFlash2SpeculatorConfig.adapt_draft_model_config(spec)
    generic.assert_called_once_with(spec)
    # A DFlash2 draft of another target.
    assert _resolve(_spec("qwen3"), method)[1] is DFlash2SpeculatorConfig


def test_no_draft_yet_keeps_the_generic_speculator():
    # method="dflash" without a model: no draft config to inspect.
    spec = _spec()
    spec.draft_model_config = None
    assert Inkling2DFlash2SpeculatorConfig.specialize(spec) is DFlashSpeculatorConfig


def _adapt(spec):
    with mock.patch.object(DFlash2SpeculatorConfig, "adapt_draft_model_config"):
        Inkling2DFlash2SpeculatorConfig.adapt_draft_model_config(spec)
    return spec


def test_target_captures_the_draft_ids_without_the_dflash_shift():
    # The generic DFlash resolver adds 1 to target_layer_ids; Inkling2
    # captures the input of a visit, and its last id is the final-norm output.
    spec = _adapt(_spec())
    assert get_eagle3_aux_layers_from_config(spec) == (5, 23, 41, 48)
    assert spec.num_speculative_tokens == 7


def _reject(spec, match):
    with pytest.raises(ValueError, match=match):
        _adapt(spec)


@pytest.mark.parametrize(
    "ids", [(5, 23, 41, 47), (23, 5, 41, 48), (-1, 23, 41, 48), (48,)]
)
def test_rejects_ids_that_are_not_visits_then_the_final_norm(ids):
    _reject(_spec(ids=ids), "target_layer_ids")


def test_rejects_other_loop_passes():
    spec = _spec()
    spec.target_model_config.hf_text_config.num_loop_passes = 2
    _reject(spec, "loop passes")


@pytest.mark.parametrize(
    "field,value", [("hidden_size", 4608), ("num_target_layers", 24)]
)
def test_rejects_drafts_trained_for_another_target(field, value):
    spec = _spec()
    setattr(spec.draft_model_config.hf_config, field, value)
    _reject(spec, "another target")


def test_rejects_fewer_drafts_than_the_block_holds():
    spec = _spec()
    spec.num_speculative_tokens = 4
    _reject(spec, "num_speculative_tokens = 7")


def test_draft_embeds_through_the_target_norm():
    weight = torch.randn(10, 4)
    draft = SimpleNamespace(
        embed_tokens=torch.nn.Embedding.from_pretrained(weight),
        embed_norm=None,
        has_separate_mask_embedding=False,
        mask_token_id=9,
    )
    ids = torch.tensor([1, 9])
    torch.testing.assert_close(
        DFlashQwen3Model.embed_input_ids(draft, ids), weight[ids]
    )
    draft.embed_norm = torch.nn.RMSNorm(4, eps=1e-6)
    expected = draft.embed_norm(weight[ids])
    torch.testing.assert_close(DFlashQwen3Model.embed_input_ids(draft, ids), expected)


def test_soft_cap_after_correction_caps_the_summed_scores(
    dist_init, default_vllm_config
):
    torch.manual_seed(3)
    batch, steps, top_k, rank, hidden_size, vocab = 2, 3, 4, 8, 16, 32
    selector = CandidateSelector(
        hidden_size=hidden_size,
        vocab_size=vocab,
        rank=rank,
        top_k=top_k,
        params_dtype=torch.float32,
        prefix="candidate_selector",
        post_softcap=5.0,
    )
    for p in (selector.predecessor_codebook, selector.successor_codebook):
        p.data.normal_(0, 4)
    selector.hidden_projection.weight.data.normal_(0, 4)
    candidate_ids = torch.randint(vocab, (batch, steps, top_k))
    unary = torch.randn(batch, steps, top_k) * 10
    hidden = torch.randn(batch, steps, hidden_size)
    anchors = torch.randint(vocab, (batch,))

    actual = selector(candidate_ids, unary, hidden, anchors)
    raw = _score_edges_reference(
        selector.predecessor_codebook,
        selector.successor_codebook,
        candidate_ids,
        unary,
        selector.hidden_projection(hidden),
        anchors,
        top_k,
    )
    torch.testing.assert_close(actual, 5.0 * torch.tanh(raw / 5.0))
