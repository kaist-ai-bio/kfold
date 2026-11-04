"""Optional preselection scores a pool without running any full folding path."""

from contextlib import contextmanager
from dataclasses import make_dataclass, replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kfold.model.modules.apo_relation import ApoUtilityHead
from kfold.model.modules.apo_selection import score_apo_combinations


def _record(name, **values):
    return make_dataclass(name, [(key, object) for key in values])(**values)


def _input(num_apo=3):
    ids = torch.tensor([[1, 2, 3]])
    uid = torch.tensor([[10, 10, 20]])
    observed = torch.ones(1, 3, num_apo, dtype=torch.bool)
    slots = torch.arange(num_apo)[None, None, :, None].expand(1, 3, num_apo, 3).float()
    token = _record(
        "SelectionTokens",
        apo_uid=uid,
        asym_id=ids,
        pad_mask=torch.ones(1, 3, dtype=torch.bool),
        apo_center_coords=slots,
        apo_repr_coords=slots,
        apo_frame_coords=slots[..., None].expand(1, 3, num_apo, 3, 3),
        apo_center_mask=observed,
        apo_repr_mask=observed,
        apo_frame_mask=observed,
    )
    atom = _record(
        "SelectionAtoms",
        token_index=torch.tensor([[0, 1, 2]]),
        apo_coords=slots,
        apo_mask=observed,
        prior_coords=torch.randn(1, 3, 1, 3),
    )
    sequence = _record(
        "SelectionSequence",
        asym_id=ids,
        bb_struct_token_id=slots[..., 0].long(),
        fa_struct_token_id=slots[..., 0].long(),
    )
    return _record(
        "SelectionInput",
        batch_size=1,
        device=torch.device("cpu"),
        token=token,
        atom=atom,
        sequence=sequence,
    )


class _StructureCapture(nn.Module):
    def __init__(self, calls):
        super().__init__()
        self.encoder = nn.Identity()
        self.calls = calls

    def forward(self, f, *, feature_taps):
        self.calls["structure"] += 1
        ids = f.sequence.bb_struct_token_id.transpose(1, 2)
        feature_taps["apo_single"] = ids[..., None].expand(-1, -1, -1, 2560).float()
        feature_taps["apo_single_mask"] = ids >= 0


class _PairCapture(nn.Module):
    channel_apo = 64

    def __init__(self, calls):
        super().__init__()
        self.calls = calls

    def forward(self, f, *, feature_taps):
        self.calls["apo"] += 1
        xyz = f.token.apo_repr_coords.transpose(1, 2)
        b, k, length, _ = xyz.shape
        observed = f.token.apo_repr_mask.transpose(1, 2)
        mask = observed[..., :, None] & observed[..., None, :]
        mask &= (f.token.apo_uid[:, :, None] == f.token.apo_uid[:, None, :])[:, None]
        pair = xyz[..., 0, None].expand(b, k, length, length)
        feature_taps["apo_pair"] = pair[..., None].expand(-1, -1, -1, -1, 64)
        feature_taps["apo_pair_mask"] = mask


def _model(with_structure=True):
    calls = dict(input=0, structure=0, apo=0, forbidden=0, offload=0)

    def input_embedder(f):
        calls["input"] += 1
        return torch.arange(12).reshape(1, 3, 4).float(), torch.zeros(1, 3, 3, 2)

    def forbidden(*args, **kwargs):
        calls["forbidden"] += 1
        raise AssertionError("Preselection must not run full folding or sequence LMs")

    @contextmanager
    def encoder_on_device(encoder, device):
        calls["offload"] += 1
        yield

    model = SimpleNamespace(
        apo_repr_mix=SimpleNamespace(apo_selector=ApoUtilityHead(4, 8)),
        input_embedder=input_embedder,
        prot_struct_encoder=_StructureCapture(calls) if with_structure else None,
        apo_module=_PairCapture(calls),
        _encoder_on_device=encoder_on_device,
        run_trunk=forbidden,
        prot_seq_encoder=forbidden,
        rna_seq_encoder=forbidden,
        confidence_head=forbidden,
        diffusion_head=SimpleNamespace(sample_structure=forbidden),
    )
    return model, calls


def test_preselector_never_runs_trunk_generation_or_sequence_lm():
    model, calls = _model()
    f = _input()
    before = torch.get_rng_state().clone()
    result = score_apo_combinations(model, f, max_combinations=3)
    assert torch.equal(before, torch.get_rng_state())
    assert calls == dict(input=1, structure=3, apo=3, forbidden=0, offload=3)
    assert result["scores"].shape == (1, 3) and torch.isfinite(result["scores"]).all()
    assert result["valid"].all() and len(result["branches"]) == 3
    assert len(result["assignments"][0]) == 3
    context = result["context"]
    assert context["apo_single"].shape == (1, 3, 3, 2560)
    assert context["apo_pair"].shape == (1, 3, 3, 3, 64)
    assert "apo_distance" not in context and "branch_z" not in context
    assert not result["scores"].requires_grad
    assert not context["apo_single"].requires_grad
    expected = model.apo_repr_mix.apo_selector(f, context, result["s_inputs"])
    assert torch.equal(result["scores"], expected)
    for branch in result["branches"]:
        assert branch.atom.prior_coords is f.atom.prior_coords


def test_geometry_only_selector_fills_unavailable_single_features():
    model, calls = _model(with_structure=False)
    result = score_apo_combinations(model, _input(), max_combinations=2)
    assert calls == dict(input=1, structure=0, apo=2, forbidden=0, offload=0)
    assert result["scores"].shape == (1, 2) and torch.isfinite(result["scores"]).all()
    assert result["valid"].all()
    context = result["context"]
    assert not context["apo_single_mask"].any()
    assert torch.count_nonzero(context["apo_single"]) == 0
    assert context["apo_pair_mask"].any()


def test_empty_pool_is_invalid_without_any_full_model_call():
    model, calls = _model()
    result = score_apo_combinations(model, _input(num_apo=0), max_combinations=3)
    assert calls["forbidden"] == 0
    assert not result["valid"].any()
    assert torch.isneginf(result["scores"]).all()
    assert result["context"]["apo_single"].shape == (1, 1, 3, 2560)
    assert result["context"]["apo_pair"].shape == (1, 1, 3, 3, 64)
    assert result["assignments"] == [[{}]]


def test_preselector_rejects_unsupported_batch_and_invalid_configuration():
    model, _ = _model()
    f = _input()
    with pytest.raises(ValueError, match="batch_size=1"):
        score_apo_combinations(model, replace(f, batch_size=2))
    with pytest.raises(ValueError, match="positive"):
        score_apo_combinations(model, f, max_combinations=0)
    model.apo_repr_mix.apo_selector = None
    with pytest.raises(ValueError, match="apo selector"):
        score_apo_combinations(model, f)


def test_topk_routes_complete_assignments_and_skips_invalid():
    from kfold.model.modules.apo_selection import select_scored_combinations

    scored = dict(
        scores=torch.tensor([[0.1, 0.9, 3.0, 0.9]]),
        valid=torch.tensor([[True, True, False, True]]),
        branches=["a", "b", "c", "d"],
        assignments=[[{10: i, 20: i} for i in range(4)]],
    )
    selected = select_scored_combinations(scored, 2)
    assert selected["branches"] == ["b", "d"]
    assert selected["indices"].tolist() == [[1, 3]]
    assert selected["assignments"] == [[{10: 1, 20: 1}, {10: 3, 20: 3}]]
    scored["valid"].zero_()
    assert select_scored_combinations(scored, 2) is None
    with pytest.raises(ValueError):
        select_scored_combinations(scored, 0)
