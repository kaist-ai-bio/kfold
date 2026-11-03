"""Auditable identities, temporal eligibility and cache/sampling accounting.

Inputs must already carry the source's canonical chemistry/sequence identity.
This module deliberately does not silently normalize molecules or assign labels.
"""
from __future__ import annotations

import calendar
import hashlib
import json
import re
from collections import Counter
from datetime import date
from typing import Iterable, Mapping


def extend_system_registry(system_ids: Iterable[str], previous=None) -> dict[str, int]:
    """Append immutable, unique uint32 indices; assay replicates share a seed.

    Persist and reuse the returned registry when extending/reordering a corpus.
    Rebuilding an index after filtering is forbidden: it changes dropout views.
    """
    result = dict(previous or {})
    values = list(result.values())
    if any(not isinstance(k, str) or not k for k in result):
        raise ValueError("Registry keys must be nonempty system identities")
    if any(type(v) is not int or not 0 <= v < 2**32 for v in values):
        raise ValueError("Registry indices must be uint32 integers")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate registry indices")
    incoming = set(system_ids)
    if any(not isinstance(k, str) or not k for k in incoming):
        raise ValueError("Missing system identity")
    next_index = max(values, default=-1) + 1
    for key in sorted(incoming - result.keys()):
        if next_index >= 2**32:
            raise ValueError("Seed index space exhausted")
        result[key] = next_index
        next_index += 1
    return result


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def date_interval(value: str | None) -> tuple[date, date] | None:
    """Interpret partial source dates as intervals, never invented exact dates."""
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    if re.fullmatch(r"\d{4}", text):
        y = int(text)
        return date(y, 1, 1), date(y, 12, 31)
    if re.fullmatch(r"\d{4}-\d{2}", text):
        y, m = map(int, text.split("-"))
        return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        exact = date.fromisoformat(text)
        return exact, exact
    raise ValueError(f"Unsupported date format: {text!r}")


def temporal_status(value, cutoff: str) -> str:
    """Inclusive cutoff. Missing/ambiguous dates never pass a temporal split."""
    boundary = date.fromisoformat(cutoff)
    try:
        interval = date_interval(value)
    except (ValueError, TypeError):
        return "invalid_date"
    if interval is None:
        return "missing_date"
    lo, hi = interval
    if hi <= boundary:
        return "on_or_before_cutoff"
    if lo > boundary:
        return "after_cutoff"
    return "straddles_cutoff"


def full_cross_bytes(protein_tokens: int, ligand_tokens: int,
                     pair_channels=256, single_channels=768, bins=64) -> int:
    """BF16 tensor payload only: directed PL/LP/LL, triangular logits, singles.

    Excludes indices, metadata, padding, compression and replicas.
    """
    if any(type(x) is not int or x <= 0 for x in
           (protein_tokens, ligand_tokens, pair_channels, single_channels, bins)):
        raise ValueError("Token/channel counts must be positive integers")
    p, l = protein_tokens, ligand_tokens
    return 2 * (pair_channels * (2*p*l + l*l)
                + bins * (p*l + l*(l+1)//2) + single_channels * (p+l))


def exposure_budget(steps: int, global_batch: int, systems: int, labels: int) -> dict:
    if any(type(x) is not int or x <= 0 for x in (steps, global_batch, systems, labels)):
        raise ValueError("Counts must be positive integers")
    slots = steps * global_batch
    return dict(label_slots=slots, slots_per_system=slots/systems,
                slots_per_label=slots/labels,
                note="Arithmetic average only; use actual training-only sampler draws for exposure distribution")


def summarize_draws(system_ids: Iterable[str], draws: Iterable[str]) -> dict:
    """Count real sampler output, including zero-exposure training systems."""
    counts = dict.fromkeys(system_ids, 0)
    if not counts:
        raise ValueError("Empty training population")
    for sid in draws:
        if sid not in counts:
            raise ValueError(f"Sampled a non-training system: {sid}")
        counts[sid] += 1
    ordered = sorted(counts.values())
    def quantile(q):
        return ordered[round((len(ordered)-1)*q)]
    return dict(total_draws=sum(ordered), unique_seen=sum(v > 0 for v in ordered),
                never_seen=sum(v == 0 for v in ordered), mean=sum(ordered)/len(ordered),
                p50=quantile(.5), p90=quantile(.9), p99=quantile(.99), maximum=ordered[-1])


def audit_records(records: Iterable[Mapping], *, cutoff: str | None = None,
                  date_field: str = "publication_date") -> dict:
    """Inspect a joined label/system manifest without filtering or relabeling.

    Keep publication, database curation and backbone training dates separate.
    A date field must be chosen explicitly by the caller; never use file mtime.
    """
    if cutoff is not None:
        date.fromisoformat(cutoff)
    counts = {k: Counter() for k in ("origin", "endpoint", "relation", "split", "target_family")}
    temporal = Counter()
    systems, labels, ligand_splits, shapes = {}, set(), {}, {}
    duplicate_rows = total = 0
    for row in records:
        total += 1
        if not row.get("system_id") or not row.get("record_id"):
            raise ValueError("Empty system_id or record_id")
        sid, rid = str(row["system_id"]), str(row["record_id"])
        sequence, ligand = row.get("sequence"), row.get("canonical_smiles")
        if not sequence or not ligand:
            raise ValueError("Joined canonical sequence/SMILES required for identity audit")
        identity = (sequence, ligand)
        if sid in systems and systems[sid] != identity:
            raise ValueError(f"Conflicting chemistry/sequence for {sid}")
        systems[sid] = identity
        duplicate_rows += rid in labels
        labels.add(rid)
        for key, counter in counts.items():
            counter[str(row.get(key) or "unknown")] += 1
        split = row.get("split")
        if split:
            ligand_splits.setdefault(ligand, set()).add(split)
        if cutoff is not None:
            temporal[temporal_status(row.get(date_field), cutoff)] += 1
        p, l = row.get("protein_tokens"), row.get("ligand_tokens")
        if p is not None and l is not None:
            shape = (int(p), int(l))
            if sid in shapes and shapes[sid] != shape:
                raise ValueError(f"Conflicting token counts for {sid}")
            shapes[sid] = shape
    reverse = Counter(systems.values())
    return dict(rows=total, unique_record_ids=len(labels), duplicate_record_rows=duplicate_rows,
                unique_system_ids=len(systems), unique_sequence_smiles_pairs=len(reverse),
                identities_with_multiple_system_ids=sum(v > 1 for v in reverse.values()),
                composition={k: dict(v) for k, v in counts.items()},
                cutoff=cutoff, date_field=date_field, temporal_counts=dict(temporal),
                ligands_crossing_splits=sum(len(v) > 1 for v in ligand_splits.values()),
                token_counted_systems=len(shapes),
                estimated_bf16_payload_bytes=sum(full_cross_bytes(*x) for x in shapes.values()),
                production_ready=False)
