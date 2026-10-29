"""Independent, zero-output residual set mixers over frozen apo branches."""

import itertools
import math
import random
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(kw_only=True)
class ApoReprMixConfig:
    enabled: bool = False
    hidden: int = 512
    chunk_size: int = 1024
    max_combinations: int = 3
    train_max_apo: int = 3

    def __post_init__(self):
        if (
            min(self.hidden, self.chunk_size, self.max_combinations, self.train_max_apo)
            < 1
        ):
            raise ValueError("Apo mixer sizes and combination limits must be positive")


class ResidualSetMixer(nn.Module):
    def __init__(self, channels, hidden, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
        self.phi = nn.Sequential(
            nn.LayerNorm(2 * channels), nn.Linear(2 * channels, hidden), nn.SiLU()
        )
        self.rho = nn.Sequential(nn.Linear(hidden + channels, hidden), nn.SiLU())
        self.out = nn.Linear(hidden, channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _chunk(self, base, peers, valid):
        h = self.phi(torch.cat((peers, peers - base[:, None]), dim=-1))
        weights = valid[:, :, None, None].to(h.dtype)
        pooled = (h * weights).sum(1) / weights.sum(1).clamp_min(1)
        delta = self.out(self.rho(torch.cat((base, pooled), dim=-1)))
        return delta * valid.any(1)[:, None, None]

    def forward(self, base, peers, valid):
        # base [B,...,C], peers [B,K,...,C], valid [B,K].
        shape = base.shape
        b = base.reshape(shape[0], -1, shape[-1])
        p = peers.reshape(shape[0], peers.shape[1], -1, shape[-1])
        deltas = []
        for start in range(0, b.shape[1], self.chunk_size):
            args = (
                b[:, start : start + self.chunk_size],
                p[:, :, start : start + self.chunk_size],
                valid,
            )
            if torch.is_grad_enabled() and self.training:
                delta = checkpoint(self._chunk, *args, use_reentrant=False)
            else:
                delta = self._chunk(*args)
            deltas.append(delta)
        return base + torch.cat(deltas, dim=1).reshape(shape).to(base.dtype)


class ApoRepresentationMixers(nn.Module):
    def __init__(self, channel_z, channel_s, config):
        super().__init__()
        self.structure_z = ResidualSetMixer(channel_z, config.hidden, config.chunk_size)
        self.confidence_z = ResidualSetMixer(channel_z, config.hidden, config.chunk_size)
        self.confidence_s = ResidualSetMixer(channel_s, config.hidden, config.chunk_size)

    def forward(self, z, s, branch_z, branch_s, valid):
        # Never share trainable parameters or pass confidence gradients to structure_z.
        return (
            self.structure_z(z, branch_z, valid),
            self.confidence_z(z, branch_z, valid),
            self.confidence_s(s, branch_s, valid),
        )


def sample_combinations(options, cap, rng):
    """Balanced random pairing, then unique mixed-radix draws; no Cartesian allocation."""
    if cap < 1:
        raise ValueError("max_combinations must be positive")
    if not options:
        return [()]
    options = [rng.sample(list(x), len(x)) for x in options]
    total = math.prod(map(len, options))
    wanted = min(cap, total)
    rows = []
    seen = set()
    for k in range(min(wanted, max(map(len, options)))):
        row = tuple(x[k % len(x)] for x in options)
        if row not in seen:
            rows.append(row)
            seen.add(row)
    # A bounded random draw plus lazy fallback avoids materializing huge products.
    for _ in range(10 * wanted):
        if len(rows) == wanted:
            break
        row = tuple(rng.choice(x) for x in options)
        if row not in seen:
            rows.append(row)
            seen.add(row)
    if len(rows) < wanted:
        for row in itertools.product(*options):
            if row not in seen:
                rows.append(row)
                seen.add(row)
            if len(rows) == wanted:
                break
    return rows


def make_apo_branches(f, cap, train_max_apo=None):
    """Slice all apo geometry/tokens together. Preserve correlated apo_uid groups.

    Chemical reference features, crop, atom layout, and ECSI priors are untouched.
    Returned validity masks exclude repeated padding branches in mixed-size batches.
    """
    seed = int(torch.randint(0, 2**31 - 1, (), device=f.device).item())
    rng = random.Random(seed)
    assignments = []
    for b in range(f.batch_size):
        groups = torch.unique(f.token.apo_uid[b][f.token.pad_mask[b]])
        groups = [int(g) for g in groups.tolist() if g >= 0]
        active, options = [], []
        for group in groups:
            members = (f.token.apo_uid[b] == group) & f.token.pad_mask[b]
            available = f.token.apo_center_mask[b, members].any(0)
            available |= f.token.apo_repr_mask[b, members].any(0)
            ids = available.nonzero().flatten().tolist()
            if ids:
                if train_max_apo is not None and len(ids) > train_max_apo:
                    ids = rng.sample(ids, train_max_apo)
                active.append(group)
                options.append(ids)
        rows = sample_combinations(options, cap, rng)
        assignments.append([dict(zip(active, row, strict=True)) for row in rows])
    count = max(map(len, assignments))
    valid = torch.zeros(f.batch_size, count, dtype=torch.bool, device=f.device)
    branches = []
    for k in range(count):
        token_indices = torch.zeros_like(f.token.apo_uid)
        seq_indices = torch.zeros_like(f.sequence.asym_id)
        for b, rows in enumerate(assignments):
            valid[b, k] = k < len(rows) and bool(rows[min(k, len(rows) - 1)])
            row = rows[min(k, len(rows) - 1)]
            for group, index in row.items():
                member = f.token.apo_uid[b] == group
                token_indices[b, member] = index
                chains = torch.unique(f.token.asym_id[b, member])
                for chain in chains:
                    seq_indices[b, f.sequence.asym_id[b] == chain] = index
        atom_indices = token_indices.gather(1, f.atom.token_index.clamp_min(0))

        def take(tensor, indices):
            idx = indices[..., None]
            for _ in range(tensor.ndim - 3):
                idx = idx.unsqueeze(-1)
            idx = idx.expand(*indices.shape, 1, *tensor.shape[3:])
            return tensor.gather(2, idx)

        token = replace(
            f.token,
            **{
                name: take(getattr(f.token, name), token_indices)
                for name in (
                    "apo_center_coords",
                    "apo_repr_coords",
                    "apo_frame_coords",
                    "apo_center_mask",
                    "apo_repr_mask",
                    "apo_frame_mask",
                )
            },
        )
        atom = replace(
            f.atom,
            **{
                name: take(getattr(f.atom, name), atom_indices)
                for name in ("apo_coords", "apo_mask")
            },
        )
        sequence = replace(
            f.sequence,
            **{
                name: take(getattr(f.sequence, name), seq_indices)
                for name in ("bb_struct_token_id", "fa_struct_token_id")
            },
        )
        branches.append(replace(f, token=token, atom=atom, sequence=sequence))
    return branches, valid, assignments


def mix_representations(
    model, f, s, lm, z, num_recycles, max_combinations=None, training=False
):
    """Keep the baseline RNG stream intact, including at zero initialization."""
    config = model.apo_repr_mix_config
    cap = config.max_combinations if max_combinations is None else max_combinations
    if cap < 1:
        raise ValueError("max_combinations must be positive")
    devices = (
        [f.device.index if f.device.index is not None else torch.cuda.current_device()]
        if f.device.type == "cuda"
        else []
    )
    zs, ss = [], []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        branches, valid, assignments = make_apo_branches(
            f, cap, config.train_max_apo if training else None
        )
        for branch in branches:
            # Every branch sees the same dropout stream, independent of branch order.
            with torch.random.fork_rng(devices=devices):
                _, branch_s, branch_z = model.run_trunk(branch, num_recycles)
            zs.append(branch_z.detach())
            ss.append(branch_s.detach())
    model.last_apo_mix_assignments = assignments
    return model.apo_repr_mix(
        z.detach(), lm.detach(), torch.stack(zs, 1), torch.stack(ss, 1), valid
    )
