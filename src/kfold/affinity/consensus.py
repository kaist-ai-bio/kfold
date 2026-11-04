"""Train-only distance15 consensus, matched to recovered colleague code.

Evidence selection/entropy filtering is an upstream operation. This module
reports both stability gates; it does not convert a broad/unstable candidate into
an approved target view or infer a negative affinity label.
"""
from dataclasses import dataclass

import numpy as np

REFERENCE_REVISION = '7cd98dd728a95efe6560f0a59a741c7c66bd4c7e'


@dataclass(frozen=True)
class Evidence:
    system_id: str
    target_key: str
    ligand_identity: str
    candidate_rank: int
    distances: np.ndarray
    training_member: bool
    synthetic: bool


def _jaccard(left, right):
    return len(left & right) / len(left | right)


def _medoid(pockets):
    if len(pockets) == 1:
        return 0, 1.0
    scores = [sum(_jaccard(a, b) for j, b in enumerate(pockets) if i != j)
              / (len(pockets)-1) for i, a in enumerate(pockets)]
    best = max(range(len(scores)), key=scores.__getitem__)
    return best, float(scores[best])


def consensus_candidate(evidence):
    """Consume exactly ten valid distinct training ligands with frozen ranks.

Split halves use ORIGINAL candidate_rank parity, including refill gaps, not
the even/odd positions of the selected ten. This matches the recovered source.
"""
    if len(evidence) != 10:
        raise ValueError('Exactly ten valid distinct training ligands required')
    if any(e.training_member is not True or e.synthetic is not False for e in evidence):
        raise ValueError('Validation and synthetic evidence are forbidden')
    if any(type(e.candidate_rank) is not int or e.candidate_rank < 0 for e in evidence):
        raise ValueError('Frozen nonnegative candidate ranks required')
    ordered = sorted(evidence, key=lambda e: (e.candidate_rank, e.system_id))
    if any(not e.system_id or not e.target_key or not e.ligand_identity for e in ordered):
        raise ValueError('Missing evidence identity')
    if len({e.system_id for e in ordered}) != 10 or len({e.ligand_identity for e in ordered}) != 10:
        raise ValueError('Duplicate evidence system or ligand')
    if len({e.target_key for e in ordered}) != 1:
        raise ValueError('Mixed target/construct evidence')
    profiles = [np.asarray(e.distances, dtype=np.float32) for e in ordered]
    if any(p.ndim != 1 or not len(p) or not np.isfinite(p).all() or (p < 0).any() for p in profiles):
        raise ValueError('Finite nonnegative protein profiles required')
    if len({len(p) for p in profiles}) != 1:
        raise ValueError('Mismatched protein token layouts')
    pockets = [set(np.flatnonzero(p < 15.).tolist()) for p in profiles]
    if any(not p for p in pockets):
        raise ValueError('Empty pocket is not valid consensus evidence')
    medoid, mean = _medoid(pockets)
    halves = [[i for i,e in enumerate(ordered) if e.candidate_rank % 2 == parity] for parity in (0,1)]
    split = 0.
    if all(halves):
        representatives = [indices[_medoid([pockets[i] for i in indices])[0]] for indices in halves]
        split = _jaccard(*(pockets[i] for i in representatives))
    return dict(target_key=ordered[medoid].target_key,
        medoid_system_id=ordered[medoid].system_id,
        distance_profile=profiles[medoid].copy(), medoid_mean_jaccard=mean,
        split_half_jaccard=float(split), strict_stable=mean >= .5 and split >= .5,
        evidence_system_ids=[e.system_id for e in ordered],
        evidence_candidate_ranks=[e.candidate_rank for e in ordered],
        half_sizes=[len(x) for x in halves], reference_revision=REFERENCE_REVISION,
        thresholds=dict(distance_lt=15.,mean_jaccard_ge=.5,split_half_jaccard_ge=.5))
