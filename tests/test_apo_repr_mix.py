import importlib.util
import random
import sys
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    "apo_repr_mix", Path(__file__).parents[1] / "src/kfold/model/modules/apo_repr_mix.py"
)
m = importlib.util.module_from_spec(spec)

sys.modules[spec.name] = m
spec.loader.exec_module(m)


def test_identity_and_isolated_gradients():
    net = m.ApoRepresentationMixers(4, 3, m.ApoReprMixConfig(hidden=8, chunk_size=3))
    z, s = torch.randn(2, 2, 2, 4), torch.randn(2, 2, 3)
    peers, singles = torch.randn(2, 3, 2, 2, 4), torch.randn(2, 3, 2, 3)
    valid = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    zd, zc, sc = net(z, s, peers, singles, valid)
    assert torch.equal(z, zd) and torch.equal(z, zc) and torch.equal(s, sc)
    zd.square().sum().backward()
    assert net.structure_z.out.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in net.confidence_z.parameters())
    assert all(p.grad is None for p in net.confidence_s.parameters())
    net.zero_grad(set_to_none=True)
    (zc.square().sum() + sc.square().sum()).backward()
    assert all(p.grad is None for p in net.structure_z.parameters())
    assert net.confidence_z.out.weight.grad.abs().sum() > 0
    assert net.confidence_s.out.weight.grad.abs().sum() > 0


def test_variable_count_order_padding_and_save_load():
    net = m.ResidualSetMixer(4, 8, 3)
    torch.nn.init.normal_(net.out.weight)
    base = torch.randn(2, 5, 4)
    for k in [1, 2, 3, 7]:
        peers = torch.randn(2, k, 5, 4)
        mask = torch.ones(2, k, dtype=torch.bool)
        actual = net(base, peers, mask)
        order = torch.randperm(k)
        torch.testing.assert_close(actual, net(base, peers[:, order], mask[:, order]))
        padded = torch.cat([peers, torch.randn(2, 2, 5, 4)], 1)
        pmask = torch.cat([mask, torch.zeros(2, 2, dtype=torch.bool)], 1)
        torch.testing.assert_close(actual, net(base, padded, pmask))
        other = m.ResidualSetMixer(4, 8, 3)
        other.load_state_dict(net.state_dict())
        torch.testing.assert_close(actual, other(base, peers, mask))
    assert torch.equal(base, net(base, peers, torch.zeros_like(mask)))


def test_combinations_unique_balanced_and_bounded():
    rows = m.sample_combinations([[0, 1, 2]] * 3, 3, random.Random(42))
    assert len(set(rows)) == 3
    assert all(len(set(r[g] for r in rows)) == 3 for g in range(3))
    assert rows == m.sample_combinations([[0, 1, 2]] * 3, 3, random.Random(42))
    assert len(m.sample_combinations([[0, 1, 2]] * 3, 100, random.Random(1))) == 27
    assert len(m.sample_combinations([list(range(20))] * 30, 3, random.Random(1))) == 3


def test_branch_slicing_preserves_correlated_groups_and_prior():
    from dataclasses import make_dataclass

    def record(name, **values):
        return make_dataclass(name, [(key, object) for key in values])(**values)

    # Chains 1 and 2 share a multimer apo group; chain 3 is independent.
    ids = torch.tensor([[1, 2, 3]])
    uid = torch.tensor([[10, 10, 20]])
    mask = torch.ones(1, 3, 3, dtype=torch.bool)
    slots = torch.arange(3)[None, None, :, None].expand(1, 3, 3, 3).float()
    token = record(
        "Token",
        apo_uid=uid,
        asym_id=ids,
        pad_mask=torch.ones(1, 3, dtype=torch.bool),
        apo_center_coords=slots,
        apo_repr_coords=slots,
        apo_frame_coords=slots[..., None].expand(1, 3, 3, 3, 3),
        apo_center_mask=mask,
        apo_repr_mask=mask,
        apo_frame_mask=mask,
    )
    atom = record(
        "Atom",
        token_index=torch.tensor([[0, 1, 2]]),
        apo_coords=slots,
        apo_mask=mask,
        prior_coords=torch.randn(1, 3, 5, 3),
    )
    seq = record(
        "Seq",
        asym_id=ids,
        bb_struct_token_id=slots[..., 0].long(),
        fa_struct_token_id=slots[..., 0].long(),
    )
    f = record(
        "Input",
        batch_size=1,
        device=torch.device("cpu"),
        token=token,
        atom=atom,
        sequence=seq,
    )
    torch.manual_seed(1)
    branches, valid, assignments = m.make_apo_branches(f, 3, 3)
    assert valid.all() and len(branches) == 3
    for branch in branches:
        tok = branch.token.apo_center_coords[0, :, 0, 0]
        assert tok[0] == tok[1]
        assert torch.equal(tok.long(), branch.sequence.bb_struct_token_id[0, :, 0])
        assert torch.equal(tok, branch.atom.apo_coords[0, :, 0, 0])
        assert branch.atom.prior_coords is atom.prior_coords
    assert len({row[10] for row in assignments[0]}) == 3
