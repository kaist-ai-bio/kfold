"""Detached generated-quality targets for the optional pre-generation selector."""

from typing import Any

import torch

from kfold.data.types.model_input import FoldingInput


@torch.no_grad()
def apo_utility_quality(
    f_input: FoldingInput,
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    mask_gt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference interface lDDT of a generated sample, never apo--holo similarity.

    Return quality and observed-target masks of shape [B, Nsample]. Multichain
    examples use resolved interchain representative-atom contacts. Single-chain
    examples fall back to resolved intrachain contacts. A missing interface is
    an absent target, rather than a zero-quality observation.
    """
    b, n, _, _ = x_pred.shape
    indices = f_input.token.repr_index[:, None].expand(b, n, -1)
    xyz_index = indices[..., None].expand(-1, -1, -1, 3)
    predicted = x_pred.detach().float().gather(2, xyz_index)
    reference = x_gt.detach().float().gather(2, xyz_index)
    resolved = mask_gt.gather(2, indices) & f_input.token.pad_mask[:, None]
    predicted_distance = torch.cdist(predicted, predicted)
    reference_distance = torch.cdist(reference, reference)
    length = indices.shape[-1]
    token_pair = (
        f_input.token.pad_mask[:, :, None]
        & f_input.token.pad_mask[:, None, :]
        & ~torch.eye(length, device=x_pred.device, dtype=torch.bool)[None]
    )
    interchain = f_input.token.asym_id[:, :, None] != f_input.token.asym_id[:, None, :]
    has_partner = (interchain & token_pair).any(dim=(-2, -1))
    relation_mask = torch.where(has_partner[:, None, None], interchain, ~interchain)
    is_nucleic = f_input.token.is_rna | f_input.token.is_dna
    nucleic_pair = is_nucleic[:, :, None] | is_nucleic[:, None, :]
    cutoff = torch.where(nucleic_pair, 30.0, 15.0)[:, None]
    pair_mask = (
        resolved[..., :, None]
        & resolved[..., None, :]
        & token_pair[:, None]
        & relation_mask[:, None]
        & (reference_distance < cutoff)
    )
    differences = (predicted_distance - reference_distance).abs()
    agreement = (
        sum((differences < threshold).float() for threshold in (0.5, 1.0, 2.0, 4.0)) / 4
    )
    counts = pair_mask.sum(dim=(-2, -1))
    quality = (agreement * pair_mask).sum(dim=(-2, -1)) / counts.clamp_min(1)
    return quality, counts > 0


def compute_apo_utility_loss(
    utility_out: dict[str, Any],
    f_input: FoldingInput,
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    mask_gt: torch.Tensor,
    supervision_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Regress only the randomly observed branch's frozen-generator quality.

    The selector sees no candidate coordinates. Its labels come from that
    branch's actual mini-rollout, and are never copied to unobserved apos.
    Unsampled batches retain a zero-valued autograd path for DDP.
    """
    scores = utility_out["scores"].float()
    loss = scores.sum() * 0.0
    zero = scores.detach().sum() * 0.0
    metrics = {
        "apo_utility_loss": zero,
        "apo_utility_target": zero,
        "apo_utility_prediction": zero,
        "apo_utility_observed_fraction": zero,
    }
    branch_index = utility_out["branch_index"]
    if branch_index < 0:
        return loss, metrics
    quality, observed = apo_utility_quality(f_input, x_pred, x_gt, mask_gt)
    observation_count = observed.sum(-1)
    target = (quality * observed).sum(-1) / observation_count.clamp_min(1)
    eligible = (
        supervision_mask.bool()
        & utility_out["valid"][:, branch_index].bool()
        & (observation_count > 0)
    )
    prediction = scores[:, branch_index].sigmoid()
    weights = eligible.float()
    denominator = weights.sum().clamp_min(1)
    loss = loss + ((prediction - target).square() * weights).sum() / denominator
    metrics = {
        "apo_utility_loss": loss.detach(),
        "apo_utility_target": (target * weights).sum().detach() / denominator,
        "apo_utility_prediction": (prediction * weights).sum().detach() / denominator,
        "apo_utility_observed_fraction": weights.mean().detach(),
    }
    return loss, metrics
