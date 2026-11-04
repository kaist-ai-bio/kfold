"""Materialize one small crop for the existing affinity head, never full PP z."""
import math

import torch

from .compact import read_crop
from .crops import distance_bin_centers


def compact_head_inputs(compact, view, *, contact_cutoff=8.0):
    """Return FP32 CPU kwargs; the caller controls device/autocast/batching.

    This follows the colleague's three-channel distogram convention. PP entries
    are zero placeholders, not recovered backbone PP representations. Production
    uses the cross-pair mask in AffinityPairformer.
    """
    if not math.isfinite(contact_cutoff) or contact_cutoff <= 0:
        raise ValueError("Invalid contact cutoff")
    crop = read_crop(compact, view)
    n = len(crop['protein_mask'])
    pairs = crop['z_pair_indices'].long()
    z = torch.zeros((n, n, crop['z_pair_values'].shape[-1]), dtype=torch.float32)
    z[pairs[:, 0], pairs[:, 1]] = crop['z_pair_values'].float()
    centers = distance_bin_centers(crop)
    probability = crop['distogram_tri_logits'].float().softmax(-1)
    values = torch.stack((probability[:, centers <= contact_cutoff].sum(-1),
                          (probability * centers).sum(-1),
                          -(probability * probability.clamp_min(1e-8).log()).sum(-1) / math.log(len(centers))), dim=-1)
    tri = crop['distogram_tri_indices'].long()
    distances = torch.zeros((n, n, 3), dtype=torch.float32)
    distances[tri[:, 0], tri[:, 1]] = values
    distances[tri[:, 1], tri[:, 0]] = values
    result = dict(s_inputs=crop['s_inputs'].float()[None], s_lm=crop['s_lm'].float()[None],
                  z=z[None], distogram_features=distances[None],
                  token_mask=torch.ones((1, n), dtype=torch.bool),
                  protein_mask=crop['protein_mask'][None], ligand_mask=crop['ligand_mask'][None])
    if any(not torch.isfinite(x).all() for x in result.values()):
        raise ValueError("Nonfinite head input")
    return result
