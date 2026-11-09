"""Lossless storage of overlapping, externally selected affinity crops.

This module does not choose consensus evidence or delete source shards. A missing
view is explicit; every available view keeps the complete ligand. Indices always
refer to the unpadded full-context source, never to a re-run cropped trunk.
"""
from __future__ import annotations

import copy

import torch

from .preparation import digest


VIEW_NAMES = ("query_200", "target_consensus_200", "query_350")
SCHEMA = "pli_crop_union_bf16_v1"


def _indices(value, n):
    value = torch.as_tensor(value)
    if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
        raise ValueError("Crop indices must be a one-dimensional integer tensor")
    value = value.cpu().long()
    if not len(value) or (value < 0).any() or (value >= n).any():
        raise ValueError("Empty or out-of-bounds crop")
    if len(torch.unique(value)) != len(value):
        raise ValueError("Duplicate crop indices")
    return value.sort().values


def _subset(payload, indices):
    """Remap sparse entries without allocating a full protein-protein tensor."""
    p = payload["protein_mask"].cpu().bool()
    l = payload["ligand_mask"].cpu().bool()
    n = len(p)
    if l.shape != p.shape or (p & l).any() or not (p | l).all():
        raise ValueError("Invalid source protein/ligand masks")
    indices = _indices(indices, n)
    reverse = torch.full((n,), -1, dtype=torch.long)
    reverse[indices] = torch.arange(len(indices))
    if (reverse[l] < 0).any() or not p[indices].any() or not l.any():
        raise ValueError("Every crop must retain protein and the entire ligand")
    out = {key: payload[key][indices].clone() for key in ("s_inputs", "s_lm")}
    if any(v.dtype != torch.bfloat16 or not torch.isfinite(v).all() for v in out.values()):
        raise ValueError("Expected finite BF16 singles")
    out.update(protein_mask=p[indices].clone(), ligand_mask=l[indices].clone(),
               bin_boundaries=payload["bin_boundaries"].clone(),
               token_mapping={k: v[indices].clone()
                              for k, v in payload["token_mapping"].items()})
    for ikey, vkey, triangular in (
        ("z_pair_indices", "z_pair_values", False),
        ("distogram_tri_indices", "distogram_tri_logits", True),
    ):
        pairs = payload[ikey].long().cpu()
        if pairs.ndim != 2 or pairs.shape[1] != 2 or (pairs < 0).any() or (pairs >= n).any():
            raise ValueError("Invalid source sparse indices")
        if len(payload[vkey]) != len(pairs):
            raise ValueError("Sparse values and indices disagree")
        mapped = reverse[pairs]
        keep = (mapped >= 0).all(dim=1)
        mapped = mapped[keep]
        values = payload[vkey][keep].clone()
        pp = out["protein_mask"]
        if (pp[mapped[:, 0]] & pp[mapped[:, 1]]).any():
            raise ValueError("PP entries are forbidden")
        if triangular and (mapped[:, 0] > mapped[:, 1]).any():
            raise ValueError("Distance indices must be upper triangular")
        np, nl = int(pp.sum()), int(out["ligand_mask"].sum())
        expected = np * nl + nl * (nl + 1) // 2 if triangular else 2 * np * nl + nl * nl
        codes = mapped[:, 0] * len(indices) + mapped[:, 1]
        if len(mapped) != expected or len(torch.unique(codes)) != expected:
            raise ValueError("Incomplete or duplicate PL/LP/LL coverage")
        if values.dtype != torch.bfloat16 or not torch.isfinite(values).all():
            raise ValueError("Expected finite BF16 feature values")
        out[ikey] = mapped.to(torch.int32)
        out[vkey] = values
    return out


def pack_crop_union(source, views, *, crop_receipt, unavailable=None):
    """Pack all available views once, with explicit unavailable-view reasons.

    ``crop_receipt`` pins the selector version/config and consensus provenance.
    It must be JSON-serializable. Scientific evidence eligibility is verified by
    the selector, not inferred here from a tensor or an affinity label.
    """
    if source.get("schema") != "pli_full_cross_bf16_v1":
        raise ValueError("Expected original full-cross source")
    unavailable = dict(unavailable or {})
    if set(views) & set(unavailable) or set(views) | set(unavailable) != set(VIEW_NAMES):
        raise ValueError("Each approved view needs indices or an unavailable reason")
    if not views or any(not isinstance(v, str) or not v.strip() for v in unavailable.values()):
        raise ValueError("At least one available view and explicit reasons required")
    if not crop_receipt.get("selector_version") or not crop_receipt.get("config"):
        raise ValueError("Selector version and configuration are required")
    n = len(source["protein_mask"])
    selected = {name: _indices(value, n) for name, value in views.items()}
    # Validate each view individually: a valid union can hide an incomplete view.
    ligand = source["ligand_mask"].cpu().bool()
    protein = source["protein_mask"].cpu().bool()
    for idx in selected.values():
        mask = torch.zeros(n, dtype=torch.bool)
        mask[idx] = True
        if not mask[ligand].all() or not protein[idx].any():
            raise ValueError("Each view must keep all ligand tokens and protein")
    union = torch.unique(torch.cat(list(selected.values())), sorted=True)
    out = _subset(source, union)
    reverse = torch.full((n,), -1, dtype=torch.long)
    reverse[union] = torch.arange(len(union))
    index_lists = {name: idx.tolist() for name, idx in selected.items()}
    receipt = dict(source_cache_identity_sha256=source["cache_identity_sha256"],
                   crop=copy.deepcopy(crop_receipt), source_view_indices=index_lists,
                   unavailable_views=unavailable, schema=SCHEMA)
    out.update(schema=SCHEMA, source_token_indices=union,
               view_indices={name: reverse[idx] for name, idx in selected.items()},
               unavailable_views=unavailable, crop_receipt=receipt,
               receipt=copy.deepcopy(source["receipt"]),
               cache_identity_sha256=digest(receipt))
    return out


def read_crop(compact, view):
    """Return one sparse head view, without substituting a missing consensus."""
    if compact.get("schema") not in (SCHEMA, "pli_historical_crop_union_bf16_v1"):
        raise ValueError("Unexpected compact schema")
    if view not in compact["view_indices"]:
        raise ValueError(f"Unavailable crop {view}: {compact['unavailable_views'].get(view, 'unknown view')}")
    idx = compact["view_indices"][view]
    out = _subset(compact, idx)
    out.update(schema="pli_crop_view_bf16_v1", view=view,
               source_token_indices=compact["source_token_indices"][idx].clone(),
               receipt=copy.deepcopy(compact["receipt"]))
    return out


def verify_against_source(source, compact):
    """Check every retained tensor and view before any source-retirement step.

    This is necessary, but not sufficient, for deletion: whole-shard coverage,
    durable publication, checksums, actual head reads, and scientific crop
    provenance must also pass. No file is deleted by this function.
    """
    if compact["crop_receipt"]["source_cache_identity_sha256"] != source["cache_identity_sha256"]:
        raise ValueError("Source identity changed")
    if compact["cache_identity_sha256"] != digest(compact["crop_receipt"]):
        raise ValueError("Compact identity changed")
    if compact["receipt"] != source["receipt"]:
        raise ValueError("Model/input/seed provenance changed")
    expected_views = compact["crop_receipt"]["source_view_indices"]
    if set(compact["view_indices"]) != set(expected_views):
        raise ValueError("View inventory changed")
    if compact["unavailable_views"] != compact["crop_receipt"]["unavailable_views"]:
        raise ValueError("Unavailable-view evidence changed")
    expected_union = sorted({i for ids in expected_views.values() for i in ids})
    if compact["source_token_indices"].tolist() != expected_union:
        raise ValueError("Union mapping changed")
    for name, expected_indices in compact["crop_receipt"]["source_view_indices"].items():
        actual = read_crop(compact, name)
        if actual["source_token_indices"].tolist() != expected_indices:
            raise ValueError("View mapping changed")
        expected = _subset(source, expected_indices)
        for key, value in expected.items():
            if isinstance(value, dict):
                if value.keys() != actual[key].keys():
                    raise ValueError("Token mapping keys changed")
                equal = all(torch.equal(v, actual[key][k]) for k, v in value.items())
            else:
                equal = torch.equal(value, actual[key])
            if not equal:
                raise ValueError(f"Retained tensor mismatch: {name}/{key}")
    return {"views_verified": len(compact["view_indices"]),
            "source_tokens": len(source["protein_mask"]),
            "union_tokens": len(compact["protein_mask"]),
            "source_deleted": False}
