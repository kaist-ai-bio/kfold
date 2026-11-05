"""Exercise inference routing without loading external models or using a GPU."""

from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from test_apo_selection import _input, _model

from kfold.cli.main import create_parser
from kfold.cli.predict import _load_queries
from kfold.model import KFold
from kfold.model.modules.apo_repr_mix import (
    ApoRepresentationMixers,
    ApoReprMixConfig,
    capture_pre_trunk_apo_features,
    mix_representations,
)


def inference_model(num_apo=3, relation_depth=1, with_structure=True):
    model, calls = _model(with_structure=with_structure)
    cfg = ApoReprMixConfig(
        enabled=True,
        hidden=8,
        relation_hidden=8,
        relation_depth=relation_depth,
        weighted_structure=True,
    )
    model.apo_repr_mix_config = cfg
    model.apo_repr_mix = ApoRepresentationMixers(2, 4, cfg).eval()
    with torch.no_grad():
        model.apo_repr_mix.structure_z.out.bias.fill_(0.4)
        if relation_depth:
            model.apo_repr_mix.confidence_relation.confidence_z.out.bias.fill_(0.2)
            model.apo_repr_mix.confidence_relation.confidence_s.out.bias.fill_(0.3)
    f = _input(num_apo)
    f.is_batched = True
    model.base_z = torch.ones(1, 3, 3, 2)
    model.base_s = torch.ones(1, 3, 4)
    calls["trunks"] = []

    def run_trunk(branch, recycles, *, feature_taps=None):
        calls["trunks"].append(("base" if branch is f else "apo", recycles))
        torch.rand(1)  # Includes stochastic inference, even for one apo.
        if feature_taps is not None:
            feature_taps.update(
                capture_pre_trunk_apo_features(model, branch, model.base_s)
            )
        return model.base_s, model.base_s, model.base_z

    def sample_structure(f_input, s_inputs, z, num_steps, num_samples, **kwargs):
        assert f_input is f and f_input.atom.prior_coords is prior
        model.generation_z = z
        return {"coordinates": torch.randn(1, num_samples, 3, 3) + z.mean()}

    def confidence(f_input, s_inputs, lm, z, coordinates, *, apo_relation, apo_context):
        model.confidence_context = apo_context
        assert torch.equal(z, model.base_z)
        pairs, singles = [], []
        for x in coordinates.unbind(1):
            if apo_relation is None:
                dz, ds = torch.zeros_like(z), torch.zeros_like(lm)
            else:
                dz, ds = apo_relation(f_input, apo_context, z, s_inputs, x)
            pairs.append(dz)
            singles.append(ds)
        return {
            "pair_delta": torch.stack(pairs, 1),
            "single_delta": torch.stack(singles, 1),
        }

    prior = f.atom.prior_coords
    model.run_trunk = run_trunk
    model.diffusion_head = SimpleNamespace(sample_structure=sample_structure)
    model.confidence_head = confidence
    model.predict = MethodType(KFold.predict, model)
    model.inference = MethodType(KFold.inference, model)
    model.score_apo_combinations = MethodType(KFold.score_apo_combinations, model)
    model.cpu_offload = True
    model.rna_seq_encoder = None
    return model, calls, f


@pytest.mark.parametrize("num_apo", [1, 3])
@pytest.mark.parametrize("depth", [1, 3])
def test_default_extra_trunks_have_independent_two_recycles(num_apo, depth):
    model, calls, f = inference_model(num_apo, relation_depth=depth)
    out = model.inference(f, num_recycles=10, num_steps=1, num_samples=2)
    assert calls["trunks"] == [("base", 10)] + [("apo", 2)] * num_apo
    torch.testing.assert_close(model.generation_z, model.base_z + 0.4)
    assert torch.count_nonzero(out["confidence"]["pair_delta"]) > 0
    assert out["apo_utility"]["scores"].shape == (1, num_apo)


@pytest.mark.parametrize("num_apo", [0, 1, 3])
@pytest.mark.parametrize("with_structure", [False, True])
def test_disabled_extra_trunks_keep_apo_confidence_and_utility(num_apo, with_structure):
    model, calls, f = inference_model(num_apo, with_structure=with_structure)
    torch.manual_seed(71)
    before = torch.get_rng_state().clone()
    out = model.inference(
        f, num_recycles=10, num_steps=1, num_samples=2, use_apo_trunk=False
    )
    assert calls["trunks"] == [("base", 10)]
    assert torch.equal(model.generation_z, model.base_z)
    assert model.confidence_context is not None
    assert "branch_z" not in model.confidence_context
    assert torch.isfinite(out["apo_utility"]["scores"]).all()
    if num_apo:
        assert torch.count_nonzero(out["confidence"]["pair_delta"]) > 0
        assert torch.count_nonzero(out["confidence"]["single_delta"]) > 0
    else:
        assert not model.confidence_context["valid"].any()
        assert torch.count_nonzero(out["confidence"]["pair_delta"]) == 0
    # Extra apo encodings must not change the original generation RNG stream.
    expected_state = torch.get_rng_state().clone()
    model.apo_repr_mix = None
    torch.set_rng_state(before)
    baseline = model.inference(f, num_recycles=10, num_steps=1, num_samples=2)
    assert torch.equal(
        baseline["diffusion"]["coordinates"], out["diffusion"]["coordinates"]
    )
    assert torch.equal(expected_state, torch.get_rng_state())


@pytest.mark.parametrize("recycles", [0, 5, 10])
def test_override_and_training_recycle_inheritance(recycles):
    model, calls, f = inference_model()
    model.inference(f, num_recycles=10, num_steps=1, apo_trunk_recycles=recycles)
    assert calls["trunks"] == [("base", 10)] + [("apo", recycles)] * 3
    calls["trunks"].clear()
    mix_representations(
        model, f, model.base_s, model.base_s, model.base_z, 7, training=True
    )
    assert calls["trunks"] == [("apo", 7)] * 3


def test_legacy_mixers_can_be_bypassed_without_branch_work():
    model, calls, f = inference_model(relation_depth=0)
    out = model.inference(f, num_recycles=10, num_steps=1, use_apo_trunk=False)
    assert calls["trunks"] == [("base", 10)]
    assert torch.equal(model.generation_z, model.base_z)
    assert "apo_utility" not in out
    assert model.last_apo_mix_assignments is None


def test_preselection_remains_pre_trunk_when_extra_trunks_are_disabled():
    model, calls, f = inference_model()
    out = model.inference(
        f,
        num_recycles=10,
        num_steps=1,
        use_apo_trunk=False,
        apo_selection_top_k=1,
        apo_selection_pool_size=3,
    )
    assert calls["trunks"] == [("base", 10)]
    assert out["apo_selection"]["selected_indices"].shape == (1, 1)
    assert out["apo_utility"]["scores"].shape == (1, 1)


def test_negative_recycles_fail_before_any_trunk_call():
    model, calls, f = inference_model()
    with pytest.raises(ValueError, match="non-negative"):
        model.inference(f, apo_trunk_recycles=-1)
    assert calls["trunks"] == []


def test_cli_defaults_overrides_and_validation():
    parser = create_parser()
    args = parser.parse_args(["-i", "query.json", "-o", "out"])
    assert args.use_apo_trunk is True and args.apo_trunk_recycles == 2
    args = parser.parse_args(
        [
            "-i",
            "query.json",
            "-o",
            "out",
            "--no-apo-trunk",
            "--apo-trunk-recycles",
            "0",
        ]
    )
    assert args.use_apo_trunk is False and args.apo_trunk_recycles == 0
    args.apo_trunk_recycles = -1
    with pytest.raises(ValueError, match="--apo-trunk-recycles"):
        _load_queries(args)


@pytest.mark.parametrize("enabled", [False, True])
def test_runner_passes_and_records_controls(monkeypatch, enabled):
    pytest.importorskip("numba")
    from kfold.inference import runner as runner_module
    from kfold.inference.runner import KFoldRunner

    model, calls, f = inference_model()
    f.to = lambda device: f
    item = SimpleNamespace(
        query=SimpleNamespace(name="test"),
        ref_struct=SimpleNamespace(num_atoms=3, num_tokens=3),
        f_input=f,
        apos=[],
        priors=[],
    )
    # The runner's production path is CUDA-only; these stubs isolate API routing.
    f.is_batched = False
    inference = model.inference
    model.inference = lambda value, **kw: inference(value, **kw)
    f.add_batch_dim = lambda: f
    model.predict = lambda value, *a, **kw: KFold.predict(model, value, *a, **kw)
    monkeypatch.setattr(
        runner_module.torch.random, "fork_rng", lambda **kwargs: nullcontext()
    )
    monkeypatch.setattr(
        runner_module.confidence_metrics,
        "summarize_confidence_metrics",
        lambda *args: (
            [{"complex": {"ranking_score": 0.5}}] * 2,
            [{"plddt": np.ones(3)}] * 2,
        ),
    )
    runner = object.__new__(KFoldRunner)
    runner.model, runner.device = model, torch.device("cpu")
    result = runner.predict_from_input(
        item,
        seed=3,
        num_samples=2,
        use_apo_trunk=enabled,
        apo_trunk_recycles=5,
    )
    assert calls["trunks"] == [("base", 10)] + ([("apo", 5)] * 3 if enabled else [])
    assert result.settings["use_apo_trunk"] is enabled
    assert result.settings["apo_trunk_recycles"] == 5
