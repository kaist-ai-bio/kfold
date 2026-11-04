from types import SimpleNamespace

import pytest
import torch

from kfold.model.modules.apo_relation import ApoCandidateRelation, ApoUtilityHead
from kfold.model.modules.apo_repr_mix import ApoReprMixConfig, WeightedResidualSetMixer


def example(k=3):
    torch.manual_seed(123)
    b, length = 2, 5
    f = SimpleNamespace(
        token=SimpleNamespace(
            pad_mask=torch.ones(b, length, dtype=torch.bool),
            asym_id=torch.tensor([[1, 1, 2, 2, 3]]).expand(b, -1),
        )
    )
    context = dict(
        apo_single=torch.randn(b, k, length, 6),
        apo_single_mask=torch.ones(b, k, length, dtype=torch.bool),
        apo_pair=torch.randn(b, k, length, length, 3),
        apo_pair_mask=torch.ones(b, k, length, length, dtype=torch.bool),
        apo_distance=torch.rand(b, k, length, length) * 20,
        apo_distance_mask=torch.ones(b, k, length, length, dtype=torch.bool),
        valid=torch.ones(b, k, dtype=torch.bool),
    )
    return (
        f,
        context,
        torch.randn(b, length, length, 4),
        torch.randn(b, length, 3),
        torch.randn(b, length, 3),
    )


def model(depth):
    cfg = ApoReprMixConfig(relation_depth=depth, relation_hidden=8, chunk_size=128)
    return ApoCandidateRelation(4, 3, cfg, single_channels=6, pair_channels=3)


def activate(net):
    for branch in (net.confidence_z, net.confidence_s):
        torch.nn.init.normal_(branch.out.weight, std=0.1)
        torch.nn.init.normal_(branch.out.bias, std=0.1)


@pytest.mark.parametrize("depth", [1, 3])
def test_identity_local_order_mask_and_variable_counts(depth):
    net = model(depth)
    args = example()
    dz, ds = net(*args)
    assert torch.equal(dz, torch.zeros_like(dz))
    assert torch.equal(ds, torch.zeros_like(ds))
    activate(net)
    for k in (1, 2, 3, 7):
        f, c, z, s, x = example(k)
        expected = net(f, c, z, s, x)
        order = torch.randperm(k)
        permuted = {name: value[:, order] for name, value in c.items()}
        for a, b in zip(expected, net(f, permuted, z, s, x), strict=True):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        # Invalid references must have no effect, even with arbitrary features.
        padded = {name: torch.cat((value, value[:, :1]), 1) for name, value in c.items()}
        padded["valid"][:, -1] = False
        padded["apo_single"][:, -1] = 1000
        padded["apo_pair"][:, -1] = -1000
        for a, b in zip(expected, net(f, padded, z, s, x), strict=True):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        padded["valid"].zero_()
        for output in net(f, padded, z, s, x):
            assert torch.isfinite(output).all() and torch.count_nonzero(output) == 0


@pytest.mark.parametrize("depth", [1, 3])
def test_candidate_partner_dependency_and_isolated_gradients(depth):
    net = model(depth)
    activate(net)
    f, c, z, s, x = example()
    base = net(f, c, z, s, x)
    altered_x = x.clone()
    altered_x[:, 2:] += 8
    candidate = net(f, c, z, s, altered_x)
    assert all(not torch.allclose(a, b) for a, b in zip(base, candidate, strict=True))
    altered_s = s.clone()
    altered_s[:, 2:] += torch.tensor([3.0, -5.0, 1.0])
    partner = net(f, c, z, altered_s, x)
    assert all(not torch.allclose(a, b) for a, b in zip(base, partner, strict=True))
    base[0].square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in net.confidence_z.parameters()
    )
    assert all(p.grad is None for p in net.confidence_s.parameters())
    assert not (
        {id(p) for p in net.confidence_z.parameters()}
        & {id(p) for p in net.confidence_s.parameters()}
    )


def test_utility_has_no_candidate_or_upstream_gradient():
    f, c, z, s, x = example()
    c["apo_single"].requires_grad_()
    s.requires_grad_()
    net = ApoUtilityHead(3, 8, single_channels=6, pair_channels=3)
    score = net(f, c, s)
    assert score.shape == (2, 3)
    score.square().mean().backward()
    assert c["apo_single"].grad is None and s.grad is None
    assert net.score[-1].weight.grad.abs().sum() > 0


def test_weighted_generation_preserves_identity_and_has_local_selection_gradients():
    torch.manual_seed(42)
    net = WeightedResidualSetMixer(4, 8, 3)
    base = torch.randn(2, 7, 4)
    peer = torch.randn(2, 3, 7, 4)
    valid = torch.ones(2, 3, dtype=torch.bool)
    assert torch.equal(base, net(base, peer, valid))
    torch.nn.init.normal_(net.out.weight, std=0.2)
    output = net(base, peer, valid)
    output.square().mean().backward()
    assert net.score.weight.grad.abs().sum() > 0
    torch.testing.assert_close(output, net(base, peer[:, [2, 0, 1]], valid[:, [2, 0, 1]]))
    assert torch.equal(base, net(base, peer, torch.zeros_like(valid)))


def test_depth_only_changes_extra_blocks_at_initialization():
    from kfold.model.modules.apo_repr_mix import ApoRepresentationMixers

    torch.manual_seed(71)
    shallow = ApoRepresentationMixers(
        4,
        3,
        ApoReprMixConfig(
            hidden=8, relation_hidden=8, relation_depth=1, weighted_structure=True
        ),
    )
    torch.manual_seed(71)
    deep = ApoRepresentationMixers(
        4,
        3,
        ApoReprMixConfig(
            hidden=8, relation_hidden=8, relation_depth=3, weighted_structure=True
        ),
    )
    actual = deep.state_dict()
    for name, expected in shallow.state_dict().items():
        assert torch.equal(actual[name], expected), name
    assert any("updates.1" in name for name in actual)
