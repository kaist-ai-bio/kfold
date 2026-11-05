"""CPU distogram profiles for consensus evidence, without constructing PP.

The entropy reduction follows recovered crop.py revision 7cd98dd: average over
all ligand atoms for each protein residue, then over residues whose minimum
expected ligand distance is below 15 A. This produces evidence, not labels.
"""
import math
import torch

from .crops import distance_bin_centers


def distogram_profile(payload):
    p = payload['protein_mask'].bool().cpu()
    l = payload['ligand_mask'].bool().cpu()
    if p.ndim != 1 or l.shape != p.shape or (p & l).any() or not (p | l).all():
        raise ValueError('Invalid protein/ligand masks')
    np, nl = int(p.sum()), int(l.sum())
    if not np or not nl:
        raise ValueError('Both protein and ligand required')
    pairs = payload['distogram_tri_indices'].cpu().long()
    logits = payload['distogram_tri_logits'].cpu().float()
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) != len(logits):
        raise ValueError('Invalid sparse distogram layout')
    if (pairs < 0).any() or (pairs >= len(p)).any() or not torch.isfinite(logits).all():
        raise ValueError('Invalid sparse distogram values')
    centers = distance_bin_centers(payload).cpu()
    cross = p[pairs[:, 0]] != p[pairs[:, 1]]
    pair = pairs[cross]
    protein = torch.where(p[pair[:, 0]], pair[:, 0], pair[:, 1])
    ligand = torch.where(l[pair[:, 0]], pair[:, 0], pair[:, 1])
    if len(pair) != np*nl or len(torch.unique(protein*len(p)+ligand)) != np*nl:
        raise ValueError('Incomplete or duplicate PL coverage')
    reverse = torch.full((len(p),), -1, dtype=torch.long)
    reverse[p] = torch.arange(np)
    rows = reverse[protein]
    prob = logits[cross].softmax(-1)
    expected = (prob*centers).sum(-1)
    entropy = -(prob*prob.clamp_min(1e-8).log()).sum(-1)/math.log(len(centers))
    distances = torch.full((np,), float('inf'))
    distances.scatter_reduce_(0, rows, expected, reduce='amin', include_self=True)
    mean_entropy = torch.zeros(np)
    mean_entropy.scatter_add_(0, rows, entropy)
    mean_entropy /= nl
    pocket = distances < 15.0
    return dict(status='passed', protein_min_expected_distance=distances,
                protein_mean_normalized_entropy=mean_entropy,
                pocket_residue_count=int(pocket.sum()),
                pocket_hlp_15a=float(mean_entropy[pocket].mean()) if pocket.any() else None)
