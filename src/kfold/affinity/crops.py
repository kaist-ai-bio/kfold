"""Coordinate-free query windows matching the colleague's online crop policy.

Consumes full-input cached features. Consensus construction is separate and must
provide a profile aligned to this exact protein input using training evidence.
"""
from __future__ import annotations

import torch


def protein_distance_profile(payload, *, min_dist=2.0, max_dist=22.0):
    """Minimum expected ligand distance per protein token, from sparse logits.

    Open-ended bins use the finite representatives from the pinned K-Fold head
    convention. Validate saved edges before using these representatives.
    """
    p = payload["protein_mask"].bool()
    l = payload["ligand_mask"].bool()
    logits = payload["distogram_tri_logits"].float()
    edges = payload["bin_boundaries"].float()
    bins = logits.shape[-1]
    if bins < 2 or not max_dist > min_dist:
        raise ValueError("Invalid distance-bin contract")
    step = (max_dist - min_dist) / bins
    expected_edges = torch.linspace(min_dist + step, max_dist - step, bins - 1)
    if edges.shape != expected_edges.shape or not torch.allclose(edges, expected_edges, atol=1e-5, rtol=0):
        raise ValueError("Saved distance bins do not match the configured head")
    pair = payload["distogram_tri_indices"].long()
    cross = (p[pair[:, 0]] & l[pair[:, 1]]) | (l[pair[:, 0]] & p[pair[:, 1]])
    pair = pair[cross]
    centers = torch.linspace(min_dist + step / 2, max_dist - step / 2, bins)
    distances = (logits[cross].softmax(-1) * centers).sum(-1)
    protein_pos = torch.where(p[pair[:, 0]], pair[:, 0], pair[:, 1])
    ligand_pos = torch.where(l[pair[:, 0]], pair[:, 0], pair[:, 1])
    expected_count = int(p.sum()) * int(l.sum())
    if len(pair) != expected_count or len(torch.unique(protein_pos * len(p) + ligand_pos)) != expected_count:
        raise ValueError("Incomplete PL distogram coverage")
    result = torch.full((len(p),), float("inf"))
    result.scatter_reduce_(0, protein_pos, distances, reduce="amin", include_self=True)
    if not torch.isfinite(result[p]).all():
        raise ValueError("Nonfinite protein distance profile")
    return result[p]


def select_windows(payload, profile, *, max_protein_tokens, max_tokens, neighborhood_size=10):
    """Same stable ranking, window placement and stop-on-overflow as feat/affinity."""
    p = payload["protein_mask"].nonzero().flatten()
    l = payload["ligand_mask"].nonzero().flatten()
    profile = torch.as_tensor(profile).float()
    if profile.shape != p.shape or not torch.isfinite(profile).all():
        raise ValueError("Distance profile must align to every protein token")
    if min(max_protein_tokens, max_tokens, neighborhood_size) <= 0:
        raise ValueError("Invalid crop budgets")
    if len(l) >= max_tokens or len(p) < neighborhood_size:
        raise ValueError("No complete protein neighborhood fits")
    if len(p) > 1 and not torch.all(p[1:] == p[:-1] + 1):
        raise ValueError("Online reference requires a contiguous monomer block")
    asym = payload["token_mapping"].get("asym_id")
    if asym is not None and len(torch.unique(asym[p])) != 1:
        raise ValueError("Multi-chain protein needs a separate crop contract")
    selected, protein_selected = set(l.tolist()), set()
    for position in torch.argsort(profile, stable=True).tolist():
        left = min(max(0, position - neighborhood_size // 2), len(p) - neighborhood_size)
        new = set(p[left:left + neighborhood_size].tolist()) - protein_selected
        if not new:
            continue
        if len(selected) + len(new) > max_tokens or len(protein_selected) + len(new) > max_protein_tokens:
            break
        selected.update(new)
        protein_selected.update(new)
    if not protein_selected:
        raise ValueError("No complete protein neighborhood fits")
    return torch.tensor(sorted(selected), dtype=torch.long)


def query_views(payload):
    """Return the two approved query views and explicit unsupported reasons."""
    profile = protein_distance_profile(payload)
    views, unavailable = {}, {}
    for name, pmax, total in (("query_200", 200, 256), ("query_350", 350, 512)):
        try:
            views[name] = select_windows(payload, profile, max_protein_tokens=pmax, max_tokens=total)
        except ValueError as exc:
            unavailable[name] = str(exc)
    return views, unavailable, profile
