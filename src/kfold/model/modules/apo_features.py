"""Prepare complete-complex apo assignments and their frozen feature evidence."""

import itertools
import math
import random
from dataclasses import dataclass, replace
from typing import TypedDict

import torch

from kfold.data.types.model_input import FoldingInput

APO_SINGLE_CHANNELS = 2560


class ApoEvidence(TypedDict):
    """Frozen evidence for K complete-complex combinations and N model tokens.

    Missing observations are zero-filled and excluded by the boolean masks.
    Padded combinations are excluded by valid, independently of token padding.
    """

    apo_single: torch.Tensor  # (B, K, N, D), raw structure encoder features
    apo_single_mask: torch.Tensor  # (B, K, N)
    apo_pair: torch.Tensor  # (B, K, N, N, C), raw apo module features
    apo_pair_mask: torch.Tensor  # (B, K, N, N), observed within apo groups
    valid: torch.Tensor  # (B, K)


class ApoContext(ApoEvidence):
    """Confidence evidence with observed apo distances in angstroms."""

    apo_distance: torch.Tensor  # (B, K, N, N), zero outside the mask
    apo_distance_mask: torch.Tensor  # (B, K, N, N)


@dataclass
class EncoderOutput:
    """Per-input encoder results shared by the base and apo trunks.

    Attributes
    ----------
    lm_single : torch.Tensor
        Combined sequence and averaged structure representation, shape (B, N, C).
    sequence_single : torch.Tensor
        Apo-independent protein and RNA representation, shape (B, N, C).
    apo_features : dict[str, torch.Tensor]
        Unaveraged structure features (B, K, N, D) and masks (B, K, N).
    """

    lm_single: torch.Tensor
    sequence_single: torch.Tensor
    apo_features: dict[str, torch.Tensor]


@dataclass
class ApoBranches:
    """Selected complete-complex assignments and their structure features."""

    inputs: list[FoldingInput]
    valid: torch.Tensor
    features: list[dict[str, torch.Tensor]]


def sample_combinations(
    options: list[list[int]], cap: int, rng: random.Random
) -> list[tuple[int, ...]]:
    """Sample distinct combinations without materializing the Cartesian product."""
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


def _sample_assignments(f: FoldingInput, cap: int, train_max_apo: int | None):
    """Choose a candidate per observed apo_uid group for each batch item."""
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
    return assignments


def _chain_candidate_indices(f, rows, asym_id):
    """Map selected apo groups to candidate indices on a sequence or token axis."""
    indices = torch.zeros_like(asym_id)
    for batch, row in enumerate(rows):
        for group, candidate in row.items():
            members = f.token.apo_uid[batch] == group
            chains = torch.unique(f.token.asym_id[batch, members])
            for chain in chains:
                indices[batch, asym_id[batch] == chain] = candidate
    return indices


def _gather_apo(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Select one apo per position from [B, N, K, ...], keeping a singleton K."""
    index = indices.reshape(*indices.shape, *([1] * (tensor.ndim - 2)))
    index = index.expand(*indices.shape, 1, *tensor.shape[3:])
    return tensor.gather(2, index)


def _select_branch_input(f: FoldingInput, rows: list[dict[int, int]]) -> FoldingInput:
    """Slice geometry by apo group and structure tokens by their full chain."""
    token_indices = torch.zeros_like(f.token.apo_uid)
    for batch, row in enumerate(rows):
        for group, candidate in row.items():
            token_indices[batch, f.token.apo_uid[batch] == group] = candidate
    atom_indices = token_indices.gather(1, f.atom.token_index.clamp_min(0))
    sequence_indices = _chain_candidate_indices(f, rows, f.sequence.asym_id)

    token = replace(
        f.token,
        **{
            name: _gather_apo(getattr(f.token, name), token_indices)
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
        apo_coords=_gather_apo(f.atom.apo_coords, atom_indices),
        apo_mask=_gather_apo(f.atom.apo_mask, atom_indices),
    )
    sequence = replace(
        f.sequence,
        bb_struct_token_id=_gather_apo(f.sequence.bb_struct_token_id, sequence_indices),
        fa_struct_token_id=_gather_apo(f.sequence.fa_struct_token_id, sequence_indices),
    )
    return replace(f, token=token, atom=atom, sequence=sequence)


def make_apo_branches(f: FoldingInput, cap: int, train_max_apo: int | None = None):
    """Select complete-complex apo combinations and materialize their inputs.

    Returns
    -------
    branches : list[FoldingInput]
        One input per combination, preserving correlated apo_uid groups.
    valid : torch.Tensor
        Boolean mask of shape (B, K). Repeated padding branches are invalid.
    assignments : list[list[dict[int, int]]]
        Per-batch combinations mapping apo_uid groups to candidate indices.
    """
    if f.token.apo_repr_coords.shape[2] == 0:
        return (
            [f],
            torch.zeros(f.batch_size, 1, dtype=torch.bool, device=f.device),
            [[{}] for _ in range(f.batch_size)],
        )
    assignments = _sample_assignments(f, cap, train_max_apo)
    count = max(map(len, assignments))
    valid = torch.zeros(f.batch_size, count, dtype=torch.bool, device=f.device)
    branches = []
    for index in range(count):
        # Repeat the final combination to keep a common branch count across B.
        rows = [batch[min(index, len(batch) - 1)] for batch in assignments]
        for batch, combinations in enumerate(assignments):
            valid[batch, index] = index < len(combinations) and bool(rows[batch])
        branches.append(_select_branch_input(f, rows))
    return branches, valid, assignments


def fill_missing_apo_features(capture, f, reference):
    """Fill missing features in-place with a single masked apo slot."""
    batch, length = f.token.pad_mask.shape
    if "apo_single" not in capture or capture["apo_single"].shape[1] == 0:
        capture["apo_single"] = reference.new_zeros(batch, 1, length, APO_SINGLE_CHANNELS)
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


def select_apo_structure_features(f, features, assignments, branch_index):
    """Gather each chain's selected candidate from the full structure encoding.

    Structure attention is restricted to individual chains, so each chain can
    reuse its encoding when other chains choose different apo candidates.
    """
    if not features or features["apo_single"].shape[1] == 0:
        return features
    rows = [batch[min(branch_index, len(batch) - 1)] for batch in assignments]
    indices = _chain_candidate_indices(f, rows, f.token.asym_id)[:, None, :]
    single = features["apo_single"].gather(
        1, indices[..., None].expand(-1, -1, -1, features["apo_single"].shape[-1])
    )
    mask = features["apo_single_mask"].gather(1, indices)
    return {"apo_single": single, "apo_single_mask": mask}
