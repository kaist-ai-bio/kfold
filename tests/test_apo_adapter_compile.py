"""Compile integration preserves adapter identities, checkpoints and gradients."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kfold.model.model_train import KFoldForTrain
from kfold.model.modules.apo_repr_mix import ApoRepresentationMixers, ApoReprMixConfig


class _Head(nn.Linear):
    def do_compile(self, **kwargs):
        self.compile(**kwargs)


def _small_model(depth):
    model = KFoldForTrain.__new__(KFoldForTrain)
    nn.Module.__init__(model)
    for name in (
        "prot_seq_encoder",
        "apo_module",
        "lm_stack",
        "main_stack",
        "refine_stack",
    ):
        setattr(model, name, nn.Linear(3, 3))
    model.rna_seq_encoder = model.prot_struct_encoder = None
    model.score_model, model.confidence_head = _Head(3, 3), _Head(3, 3)
    config = ApoReprMixConfig(
        enabled=True,
        weighted_structure=True,
        hidden=8,
        chunk_size=4,
        relation_depth=depth,
        relation_hidden=8,
    )
    model.apo_repr_mix = ApoRepresentationMixers(4, 3, config)
    # Nonzero trained-style residuals make output and upstream-gradient checks
    # meaningful rather than comparing only identical zero-initialized outputs.
    for module in (
        model.apo_repr_mix.structure_z,
        model.apo_repr_mix.confidence_relation.confidence_z,
        model.apo_repr_mix.confidence_relation.confidence_s,
    ):
        nn.init.normal_(module.out.weight, std=0.1)
        nn.init.normal_(module.out.bias, std=0.1)
    return model


def _inputs():
    b, k, length = 1, 2, 3
    f = SimpleNamespace(
        token=SimpleNamespace(
            pad_mask=torch.ones(b, length, dtype=torch.bool),
            asym_id=torch.tensor([[1, 1, 2]]),
        )
    )
    context = {
        "apo_single": torch.randn(b, k, length, 2560),
        "apo_single_mask": torch.ones(b, k, length, dtype=torch.bool),
        "apo_pair": torch.randn(b, k, length, length, 64),
        "apo_pair_mask": torch.ones(b, k, length, length, dtype=torch.bool),
        "apo_distance": torch.rand(b, k, length, length) * 20,
        "apo_distance_mask": torch.ones(b, k, length, length, dtype=torch.bool),
        "valid": torch.ones(b, k, dtype=torch.bool),
    }
    return (
        f,
        context,
        torch.randn(b, length, length, 4),
        torch.randn(b, length, 3),
        torch.randn(b, k, length, length, 4),
        torch.randn(b, k, length, 3),
        torch.randn(b, length, 3),
    )


def _forward_and_backward(model, args):
    f, context, z, s, branch_z, branch_s, x = args
    mixers = model.apo_repr_mix
    structure, conf_z, conf_s = mixers(z, s, branch_z, branch_s, context["valid"])
    dz, ds = mixers.confidence_relation(f, context, conf_z, s, x)
    scores = mixers.apo_selector(f, context, s)
    outputs = (structure, dz, ds, scores)
    loss = sum(value.square().mean() for value in outputs)
    loss.backward()
    return outputs


@pytest.mark.parametrize("depth", (1, 3))
def test_do_compile_preserves_adapter_groups_outputs_and_gradients(monkeypatch, depth):
    torch.manual_seed(43)
    reference = _small_model(depth)
    compiled = deepcopy(reference)
    expected_keys = set(compiled.apo_repr_mix.state_dict())
    expected_parameters = dict(compiled.apo_repr_mix.named_parameters())
    original_compile = torch.compile
    compiled_calls = []

    def eager_compile(callable_or_module, **kwargs):
        compiled_calls.append(getattr(callable_or_module, "__self__", None))
        return original_compile(
            callable_or_module,
            backend="eager",
            dynamic=kwargs.get("dynamic", False),
        )

    # Exercise the real do_compile integration while avoiding CPU Inductor
    # compilation and external compilers in this small contract test.
    monkeypatch.setattr(torch, "compile", eager_compile)
    compiled.do_compile()
    assert compiled.is_compiled
    groups = (
        compiled.apo_repr_mix.structure_z,
        compiled.apo_repr_mix.confidence_relation.confidence_z,
        compiled.apo_repr_mix.confidence_relation.confidence_s,
        compiled.apo_repr_mix.apo_selector,
    )
    assert all(any(call is group for call in compiled_calls) for group in groups)
    assert compiled.apo_repr_mix.structure_z.chunk_is_compiled
    assert compiled.apo_repr_mix.structure_z._compiled_call_impl is None
    assert set(compiled.apo_repr_mix.state_dict()) == expected_keys
    actual_parameters = dict(compiled.apo_repr_mix.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    assert all(
        actual_parameters[name] is value for name, value in expected_parameters.items()
    )
    for model in (reference, compiled):
        model.requires_grad_(False)
        model.apo_repr_mix.requires_grad_(True)
        model.train()
    args = _inputs()
    expected = _forward_and_backward(reference, args)
    actual = _forward_and_backward(compiled, args)
    for old, new in zip(expected, actual, strict=True):
        torch.testing.assert_close(new, old)
    for name, parameter in reference.apo_repr_mix.named_parameters():
        gradient = actual_parameters[name].grad
        assert gradient is not None and parameter.grad is not None, name
        assert torch.isfinite(gradient).all(), name
        torch.testing.assert_close(gradient, parameter.grad)
    assert all(
        p.grad is None
        for n, p in compiled.named_parameters()
        if not n.startswith("apo_repr_mix.")
    )
