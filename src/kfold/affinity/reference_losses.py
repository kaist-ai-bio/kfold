"""Recovered colleague loss reference at7cd98dd; unchanged below."""
"""Joint absolute-regression and within-assay ranking objectives."""


from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AffinityLossOutput:
    loss: torch.Tensor
    regression: torch.Tensor
    ranking: torch.Tensor
    active_ranking_groups: int
    pairwise_groups: int
    pearson_groups: int
    ranking_replicate_records: int
    difference_pairs: int = 0


def _pearson_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    centered_prediction = prediction - prediction.mean()
    centered_target = target - target.mean()
    denominator = centered_prediction.norm() * centered_target.norm()
    if denominator <= torch.finfo(prediction.dtype).eps:
        return None
    return 1.0 - (centered_prediction * centered_target).sum() / denominator


def grouped_affinity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    group_index: torch.Tensor,
    ligand_index: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    huber_delta: float = 0.5,
    ranking_weight: float = 0.5,
    pairwise_temperature: float = 0.2,
    near_tie_delta: float = 0.1,
) -> AffinityLossOutput:
    """Compute Huber plus conditional Pearson/pairwise within-assay loss."""
    if (
        prediction.shape != target.shape
        or prediction.shape != group_index.shape
        or prediction.shape != ligand_index.shape
    ):
        raise ValueError(
            "prediction, target, group_index, and ligand_index must share shape [B]."
        )
    if valid_mask.shape != prediction.shape:
        raise ValueError("valid_mask must have shape [B].")
    if not valid_mask.any():
        raise ValueError("At least one valid affinity label is required.")

    regression = F.huber_loss(
        prediction[valid_mask], target[valid_mask], delta=huber_delta
    )
    group_losses: list[torch.Tensor] = []
    pairwise_groups = 0
    pearson_groups = 0
    ranking_replicate_records = 0
    for group in torch.unique(group_index[valid_mask]).tolist():
        mask = valid_mask & (group_index == group)
        pred_group = prediction[mask]
        target_group = target[mask]
        ligand_group = ligand_index[mask]
        unique_ligands = torch.unique(ligand_group)
        ranking_replicate_records += len(pred_group) - len(unique_ligands)
        collapsed_prediction = torch.stack(
            [pred_group[ligand_group == ligand].mean() for ligand in unique_ligands]
        )
        collapsed_target = torch.stack(
            [target_group[ligand_group == ligand].mean() for ligand in unique_ligands]
        )
        if len(collapsed_prediction) == 2:
            delta = collapsed_target[0] - collapsed_target[1]
            if delta.abs() < near_tie_delta:
                continue
            signed_prediction = torch.sign(delta) * (
                collapsed_prediction[0] - collapsed_prediction[1]
            )
            group_losses.append(F.softplus(-signed_prediction / pairwise_temperature))
            pairwise_groups += 1
        elif len(collapsed_prediction) >= 3:
            loss = _pearson_loss(collapsed_prediction, collapsed_target)
            if loss is None:
                continue
            group_losses.append(loss)
            pearson_groups += 1

    if group_losses:
        ranking = torch.stack(group_losses).mean()
    else:
        ranking = regression.detach() * 0.0
    return AffinityLossOutput(
        loss=regression + ranking_weight * ranking,
        regression=regression,
        ranking=ranking,
        active_ranking_groups=len(group_losses),
        pairwise_groups=pairwise_groups,
        pearson_groups=pearson_groups,
        ranking_replicate_records=ranking_replicate_records,
    )


def boltz2_continuous_affinity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    group_index: torch.Tensor,
    ligand_index: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    huber_delta: float = 0.5,
    difference_weight: float = 0.9,
    absolute_weight: float = 0.1,
) -> AffinityLossOutput:
    """Boltz-2-style continuous supervision on logical same-assay groups.

    Exact numeric labels have no censor branch.  Raw labels contribute to the
    absolute Huber term; duplicate canonical ligands are collapsed before
    difference pairs are formed, preventing artificial replicate ranking.
    """
    if (
        prediction.shape != target.shape
        or prediction.shape != group_index.shape
        or prediction.shape != ligand_index.shape
    ):
        raise ValueError(
            "prediction, target, group_index, and ligand_index must share shape [B]."
        )
    if valid_mask.shape != prediction.shape:
        raise ValueError("valid_mask must have shape [B].")
    if not valid_mask.any():
        raise ValueError("At least one valid affinity label is required.")
    if difference_weight < 0 or absolute_weight < 0:
        raise ValueError("Boltz2 loss weights must be non-negative.")
    if difference_weight + absolute_weight <= 0:
        raise ValueError("At least one Boltz2 loss weight must be positive.")

    absolute = F.huber_loss(
        prediction[valid_mask], target[valid_mask], delta=huber_delta
    )
    difference_losses: list[torch.Tensor] = []
    active_groups = 0
    difference_pairs = 0
    replicate_records = 0
    for group in torch.unique(group_index[valid_mask]).tolist():
        mask = valid_mask & (group_index == group)
        pred_group = prediction[mask]
        target_group = target[mask]
        ligands = ligand_index[mask]
        unique_ligands = torch.unique(ligands)
        replicate_records += len(pred_group) - len(unique_ligands)
        if len(unique_ligands) < 2:
            continue
        collapsed_prediction = torch.stack(
            [pred_group[ligands == ligand].mean() for ligand in unique_ligands]
        )
        collapsed_target = torch.stack(
            [target_group[ligands == ligand].mean() for ligand in unique_ligands]
        )
        pairs = torch.triu_indices(
            len(unique_ligands),
            len(unique_ligands),
            offset=1,
            device=prediction.device,
        )
        prediction_difference = (
            collapsed_prediction[pairs[0]] - collapsed_prediction[pairs[1]]
        )
        target_difference = collapsed_target[pairs[0]] - collapsed_target[pairs[1]]
        difference_losses.append(
            F.huber_loss(prediction_difference, target_difference, delta=huber_delta)
        )
        active_groups += 1
        difference_pairs += prediction_difference.numel()
    if difference_losses:
        difference = torch.stack(difference_losses).mean()
    else:
        difference = absolute.detach() * 0.0
    return AffinityLossOutput(
        loss=absolute_weight * absolute + difference_weight * difference,
        regression=absolute,
        ranking=difference,
        active_ranking_groups=active_groups,
        pairwise_groups=active_groups,
        pearson_groups=0,
        ranking_replicate_records=replicate_records,
        difference_pairs=difference_pairs,
    )
