"""Pre-trunk taps preserve apo identity, observed masks, and baseline outputs."""

from types import SimpleNamespace

import torch
from torch import nn

from kfold.model.modules.apo_module import ApoModule
from kfold.model.modules.prot_struct_encoder import StructureEncoder


class _TokenEncoder(nn.Module):
    def forward(self, seq, bb, fa, *, seq_id, pos_id):
        return torch.stack((bb, fa, seq_id), dim=-1).float()


def _single_encoder():
    net = StructureEncoder.__new__(StructureEncoder)
    nn.Module.__init__(net)
    net.cfg = SimpleNamespace(d_model=3)
    net.encoder = _TokenEncoder()
    net.offset = 4
    return net


def _single_input(num_apo=2):
    bb = torch.tensor([[[1, 5], [-1, 6], [3, -1]]])[..., :num_apo]
    return SimpleNamespace(
        device=torch.device("cpu"),
        sequence=SimpleNamespace(
            seq_token_id=torch.tensor([[1, 2, 3]]),
            bb_struct_token_id=bb,
            fa_struct_token_id=bb.clone(),
            pad_mask=torch.ones(1, 3, dtype=torch.bool),
            is_protein=torch.ones(1, 3, dtype=torch.bool),
            asym_id=torch.ones(1, 3, dtype=torch.long),
            pos_id=torch.arange(3)[None],
        ),
        token=SimpleNamespace(
            seq_token_index=torch.tensor([[2, 0, -1, 1, 2]]),
            pad_mask=torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool),
            is_protein=torch.tensor([[1, 1, 1, 0, 1]], dtype=torch.bool),
        ),
    )


def test_single_taps_keep_slots_and_mask_invalid_token_mapping():
    net, f = _single_encoder(), _single_input()
    baseline = net(f)
    taps = {}
    actual = net(f, feature_taps=taps)
    assert torch.equal(actual, baseline)
    assert taps["apo_single"].shape == (1, 2, 5, 3)
    expected_mask = torch.tensor([[[1, 1, 0, 0, 0], [0, 1, 0, 0, 0]]], dtype=torch.bool)
    assert torch.equal(taps["apo_single_mask"], expected_mask)
    assert not taps["apo_single"].requires_grad
    assert torch.count_nonzero(taps["apo_single"][~expected_mask]) == 0
    # Original slot IDs survive token remapping and do not get averaged away.
    assert taps["apo_single"][0, 0, 0, 0] == 3 + net.offset
    assert taps["apo_single"][0, 0, 1, 0] == 1 + net.offset
    assert taps["apo_single"][0, 1, 1, 0] == 5 + net.offset

    reordered = _single_input()
    for name in ("bb_struct_token_id", "fa_struct_token_id"):
        setattr(reordered.sequence, name, getattr(reordered.sequence, name).flip(-1))
    other = {}
    assert torch.equal(net(reordered, feature_taps=other), baseline)
    assert torch.equal(other["apo_single"], taps["apo_single"].flip(1))
    assert torch.equal(other["apo_single_mask"], expected_mask.flip(1))


class _RelativePosition(nn.Module):
    def forward(self, f, dtype):
        b, length = f.token.pad_mask.shape
        return torch.zeros(b, length, length, 1, dtype=dtype)


class _PairStack(nn.Module):
    def forward(self, x, mask):
        return x


def _pair_encoder():
    net = ApoModule.__new__(ApoModule)
    nn.Module.__init__(net)
    net.channel_z, net.channel_apo = 3, 4
    net.register_buffer("distogram_boundaries", torch.tensor([1.0, 3.0, 5.0]))
    net.linear_apo = nn.Linear(3 + 1 + 3 + 1 + 2 * 32, 4)
    net.rel_pos_encoding = _RelativePosition()
    net.linear_rel_pos = nn.Linear(1, 4)
    net.stack = _PairStack()
    net.layernorm_out = nn.LayerNorm(4)
    net.linear_out = nn.Linear(4, 3)
    return net


def _pair_input(num_apo=2):
    coords = torch.arange(5 * 2 * 3).reshape(1, 5, 2, 3).float()[..., :num_apo, :]
    observed = torch.tensor([[[1, 1], [1, 0], [1, 1], [1, 1], [1, 1]]], dtype=torch.bool)[
        ..., :num_apo
    ]
    return SimpleNamespace(
        token=SimpleNamespace(
            apo_repr_coords=coords,
            apo_repr_mask=observed,
            apo_frame_coords=torch.zeros(1, 5, num_apo, 3, 3),
            apo_frame_mask=torch.zeros(1, 5, num_apo, dtype=torch.bool),
            apo_uid=torch.tensor([[10, 10, 20, -1, 10]]),
            pad_mask=torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool),
            res_type=torch.nn.functional.one_hot(torch.zeros(1, 5, dtype=torch.long), 32),
        )
    )


def test_pair_taps_preserve_geometry_availability_and_original_reduction():
    net, f = _pair_encoder(), _pair_input()
    baseline = net(f)
    taps = {}
    actual = net(f, feature_taps=taps)
    assert torch.equal(actual, baseline)
    raw, mask = taps["apo_pair"], taps["apo_pair_mask"]
    assert raw.shape == (1, 2, 5, 5, 4)
    assert not raw.requires_grad
    assert mask[0, 0, 0, 1] and not mask[0, 1, 0, 1]
    assert not mask[0, :, 0, 2].any()  # Distinct independent apo groups.
    assert not mask[0, :, 3, :].any()  # Unassigned apo UID.
    assert not mask[0, :, 4, :].any()  # Padded token.
    assert torch.count_nonzero(raw[~mask]) == 0
    reduced = raw.sum(1) / mask.sum(1).clamp(min=1)[..., None]
    assert torch.equal(actual, net.linear_out(torch.relu(reduced)))


def test_zero_apo_sources_return_empty_taps_without_encoder_work():
    single, single_f = _single_encoder(), _single_input(num_apo=0)
    single_taps = {}
    result = single(single_f, feature_taps=single_taps)
    assert result.shape == (1, 5, 3) and torch.count_nonzero(result) == 0
    assert single_taps["apo_single"].shape == (1, 0, 5, 3)
    assert single_taps["apo_single_mask"].shape == (1, 0, 5)

    pair, pair_f = _pair_encoder(), _pair_input(num_apo=0)
    pair_taps = {}
    result = pair(pair_f, feature_taps=pair_taps)
    assert result.shape == (1, 5, 5, 3) and torch.count_nonzero(result) == 0
    assert pair_taps["apo_pair"].shape == (1, 0, 5, 5, 4)
    assert pair_taps["apo_pair_mask"].shape == (1, 0, 5, 5)
