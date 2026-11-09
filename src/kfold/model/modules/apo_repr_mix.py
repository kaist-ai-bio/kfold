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
    weighted_structure: bool = False
    relation_depth: int = 0
    relation_hidden: int = 64
    utility_rollout_fraction: float = 0.25
    utility_loss_weight: float = 0.01

    def __post_init__(self):
        if (
            min(self.hidden, self.chunk_size, self.max_combinations, self.train_max_apo)
            < 1
        ):
            raise ValueError("Apo mixer sizes and combination limits must be positive")
        if self.relation_depth < 0 or self.relation_hidden < 1:
            raise ValueError("Invalid apo relation depth/hidden size")
        if not 0 <= self.utility_rollout_fraction <= 1 or self.utility_loss_weight < 0:
            raise ValueError("Invalid utility rollout fraction/loss weight")


class ResidualSetMixer(nn.Module):
    def __init__(self, channels, hidden, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_is_compiled = False
        self.phi = nn.Sequential(
            nn.LayerNorm(2 * channels), nn.Linear(2 * channels, hidden), nn.SiLU()
        )
        self.rho = nn.Sequential(nn.Linear(hidden + channels, hidden), nn.SiLU())
        self.out = nn.Linear(hidden, channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def do_compile(self, **kwargs):
        # The full pair axis has hundreds of checkpointed chunks. Compile only
        # their shared work so Dynamo does not unroll the outer Python loop.
        self._chunk = torch.compile(self._chunk, **kwargs)
        self.chunk_is_compiled = True

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


class WeightedResidualSetMixer(ResidualSetMixer):
    """Learn a separate K-axis weight for each full-complex post-trunk pair."""

    def __init__(self, channels, hidden, chunk_size):
        super().__init__(channels, hidden, chunk_size)
        self.score = nn.Linear(hidden, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def _chunk(self, base, peers, valid):
        h = self.phi(torch.cat((peers, peers - base[:, None]), dim=-1))
        mask = valid[:, :, None].expand(*h.shape[:-1])
        logits = self.score(h).squeeze(-1).float().masked_fill(~mask, -torch.inf)
        # A null reference lets the model suppress all unhelpful branches.
        logits = torch.cat((logits, torch.zeros_like(logits[:, :1])), 1)
        weights = logits.softmax(1)[:, :-1].to(h.dtype)
        pooled = (h * weights[..., None]).sum(1)
        delta = self.out(self.rho(torch.cat((base, pooled), -1)))
        return delta * valid.any(1)[:, None, None]


class ApoRepresentationMixers(nn.Module):
    def __init__(self, channel_z, channel_s, config):
        super().__init__()
        initial_rng = torch.random.get_rng_state()
        mixer_cls = (
            WeightedResidualSetMixer if config.weighted_structure else ResidualSetMixer
        )
        self.structure_z = mixer_cls(channel_z, config.hidden, config.chunk_size)
        self.confidence_relation = None
        self.apo_selector = None
        if config.relation_depth:
            from kfold.model.modules.apo_relation import (
                ApoCandidateRelation,
                ApoUtilityHead,
            )

            self.confidence_z = self.confidence_s = None
            self.confidence_relation = ApoCandidateRelation(channel_z, channel_s, config)
            self.apo_selector = ApoUtilityHead(channel_s, config.relation_hidden)
        else:
            self.confidence_z = ResidualSetMixer(
                channel_z, config.hidden, config.chunk_size
            )
            self.confidence_s = ResidualSetMixer(
                channel_s, config.hidden, config.chunk_size
            )
        if config.relation_depth > 1:
            # Depth must not also change the random initialization of every
            # shared projection, selector, and generation mixer in the comparison.
            with torch.random.fork_rng(devices=[]):
                torch.random.set_rng_state(initial_rng)
                shallow = ApoRepresentationMixers(
                    channel_z, channel_s, replace(config, relation_depth=1)
                )
            state = self.state_dict()
            state.update(shallow.state_dict())
            self.load_state_dict(state, strict=True)

    def forward(self, z, s, branch_z, branch_s, valid):
        # Never share trainable parameters or pass confidence gradients to structure_z.
        if self.confidence_relation is not None:
            return self.structure_z(z, branch_z, valid), z, s
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
    if f.token.apo_repr_coords.shape[2] == 0:
        return (
            [f],
            torch.zeros(f.batch_size, 1, dtype=torch.bool, device=f.device),
            [[{}] for _ in range(f.batch_size)],
        )
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


def normalize_apo_feature_taps(capture, f, reference):
    """Keep a masked apo slot when an encoder or the entire pool is absent."""
    batch, length = f.token.pad_mask.shape
    if "apo_single" not in capture or capture["apo_single"].shape[1] == 0:
        capture["apo_single"] = reference.new_zeros(batch, 1, length, 2560)
        capture["apo_single_mask"] = torch.zeros(
            batch, 1, length, device=f.device, dtype=torch.bool
        )
    if capture["apo_pair"].shape[1] == 0:
        capture["apo_pair"] = capture["apo_pair"].new_zeros(
            batch, 1, length, length, capture["apo_pair"].shape[-1]
        )
        capture["apo_pair_mask"] = torch.zeros(
            batch, 1, length, length, device=f.device, dtype=torch.bool
        )
    return capture


@torch.no_grad()
def capture_pre_trunk_apo_features(model, f, reference):
    """Read raw apo evidence without a sequence LM or folding trunk call."""
    capture = {}
    get_module = getattr(model, "_get_model_module", lambda module: module)
    structure_encoder = get_module(model.prot_struct_encoder)
    if structure_encoder is not None:
        with model._encoder_on_device(structure_encoder.encoder, f.device):
            structure_encoder(f, feature_taps=capture)
    get_module(model.apo_module)(f, feature_taps=capture)
    return normalize_apo_feature_taps(capture, f, reference)


def mix_representations(
    model,
    f,
    lm,
    z,
    num_recycles,
    max_combinations=None,
    training=False,
    preselected=None,
    *,
    apo_trunk_recycles=None,
):
    """Run complete-complex apo trunks sequentially and mix their representations.

    Training inherits the main recycle count. Inference supplies its independent
    branch count. Extra feature/trunk work never advances the generation RNG.
    Each branch includes all chains and uses the same stochastic stream.
    """
    config = model.apo_repr_mix_config
    cap = config.max_combinations if max_combinations is None else max_combinations
    if cap < 1:
        raise ValueError("max_combinations must be positive")
    branch_recycles = num_recycles if apo_trunk_recycles is None else apo_trunk_recycles
    if branch_recycles < 0:
        raise ValueError("apo_trunk_recycles must be non-negative")
    devices = (
        [f.device.index if f.device.index is not None else torch.cuda.current_device()]
        if f.device.type == "cuda"
        else []
    )
    zs, ss, taps = [], [], []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        if preselected is None:
            branches, valid, assignments = make_apo_branches(
                f, cap, config.train_max_apo if training else None
            )
        else:
            if training:
                raise ValueError("Apo preselection is inference-only")
            branches = preselected["branches"]
            valid = preselected["valid"]
            assignments = preselected["assignments"]
        for branch in branches:
            # Share the dropout stream across complete-complex assignments.
            with torch.random.fork_rng(devices=devices):
                if config.relation_depth:
                    capture = {}
                    _, branch_s, branch_z = model.run_trunk(
                        branch, branch_recycles, feature_taps=capture
                    )
                    taps.append(normalize_apo_feature_taps(capture, branch, branch_s))
                else:
                    _, branch_s, branch_z = model.run_trunk(branch, branch_recycles)
            zs.append(branch_z.detach())
            ss.append(branch_s.detach())
    branch_z = torch.stack(zs, 1)
    branch_s = torch.stack(ss, 1)
    model.last_apo_mix_assignments = assignments
    if config.relation_depth:
        context = {
            name: torch.cat([tap[name] for tap in taps], 1)
            for name in ("apo_single", "apo_single_mask", "apo_pair", "apo_pair_mask")
        }
        with torch.no_grad():
            if f.token.apo_repr_coords.shape[2] == 0:
                coords = z.new_zeros(f.batch_size, len(branches), z.shape[1], 3)
                masks = torch.zeros(coords.shape[:-1], dtype=torch.bool, device=f.device)
            else:
                coords = torch.stack(
                    [branch.token.apo_repr_coords[:, :, 0] for branch in branches], 1
                )
                masks = torch.stack(
                    [branch.token.apo_repr_mask[:, :, 0] for branch in branches], 1
                )
            geom_mask = (
                context["apo_pair_mask"] & masks[..., :, None] & masks[..., None, :]
            )
            distance = torch.cdist(coords.float(), coords.float())
            context["apo_distance"] = distance.masked_fill(~geom_mask, 0).detach()
            context["apo_distance_mask"] = geom_mask
        context.update(
            valid=valid,
            branches=branches,
            branch_z=branch_z,
            branch_s=branch_s,
        )
        model.last_apo_mix_context = context
    return model.apo_repr_mix(z.detach(), lm.detach(), branch_z, branch_s, valid)
