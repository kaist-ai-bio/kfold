"""Execute apo-joint structure adaptation or confidence-only inference."""

from dataclasses import dataclass

import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.modules.apo_features import (
    ApoBranches,
    ApoContext,
    EncoderOutput,
    fill_missing_apo_features,
    make_apo_branches,
    select_apo_structure_features,
)
from kfold.model.modules.apo_selection import SelectedApoCombinations


@dataclass
class ApoRollout:
    """Frozen branch inputs retained together for training utility targets."""

    inputs: list[FoldingInput]  # K complete-complex inputs
    pair: torch.Tensor  # (B, K, N, N, C)
    valid: torch.Tensor  # (B, K)


@dataclass
class ApoOutput:
    """Representations and evidence consumed by diffusion, confidence and utility."""

    structure_pair: torch.Tensor
    confidence_pair: torch.Tensor
    confidence_single: torch.Tensor
    context: ApoContext | None  # Present only for candidate-conditioned confidence.
    assignments: list[list[dict[int, int]]] | None = None
    rollout: ApoRollout | None = None  # Present only for relation-adapter training.


@torch.no_grad()
def _build_apo_context(f, branches, valid, taps, reference) -> ApoContext:
    """Assemble shared confidence evidence and selected apo geometry."""
    features = {
        name: torch.cat([tap[name] for tap in taps], 1)
        for name in ("apo_single", "apo_single_mask", "apo_pair", "apo_pair_mask")
    }
    if f.token.apo_repr_coords.shape[2] == 0:
        coords = reference.new_zeros(f.batch_size, len(branches), reference.shape[1], 3)
        masks = torch.zeros(coords.shape[:-1], dtype=torch.bool, device=f.device)
    else:
        coords = torch.stack(
            [branch.token.apo_repr_coords[:, :, 0] for branch in branches], 1
        )
        masks = torch.stack(
            [branch.token.apo_repr_mask[:, :, 0] for branch in branches], 1
        )
    geom_mask = features["apo_pair_mask"] & masks[..., :, None] & masks[..., None, :]
    distance = torch.cdist(coords.float(), coords.float())
    return ApoContext(
        apo_single=features["apo_single"],
        apo_single_mask=features["apo_single_mask"],
        apo_pair=features["apo_pair"],
        apo_pair_mask=features["apo_pair_mask"],
        apo_distance=distance.masked_fill(~geom_mask, 0).detach(),
        apo_distance_mask=geom_mask,
        valid=valid,
    )


def run_adapter(
    model,
    f: FoldingInput,
    encoded: EncoderOutput,
    z: torch.Tensor,
    branches: ApoBranches,
    devices: list[int],
    num_recycles: int,
    *,
    training: bool,
) -> ApoOutput:
    """Run each apo trunk sequentially and mix its output into the base result."""
    pair_outputs, single_outputs, taps = [], [], []
    use_relation = model.apo_repr_mix_config.relation_depth > 0
    with torch.no_grad():
        for branch, features in zip(branches.inputs, branches.features, strict=True):
            s_lm = encoded.sequence_single
            if "apo_single" in features:
                raw = features["apo_single"]
                structure = (
                    raw.sum(1)
                    / features["apo_single_mask"].sum(1).clamp_min(1)[..., None]
                )
                s_lm = s_lm + model.prot_struct_to_s_lm(structure)
            capture = dict(features) if use_relation else None
            with torch.random.fork_rng(devices=devices):
                _, branch_s, branch_z = model.run_trunk(
                    branch, num_recycles, s_lm, feature_taps=capture
                )
            if use_relation:
                taps.append(fill_missing_apo_features(capture, branch, branch_s))
            pair_outputs.append(branch_z.detach())
            single_outputs.append(branch_s.detach())
        branch_z = torch.stack(pair_outputs, 1)
        branch_s = torch.stack(single_outputs, 1)
    context = None
    if use_relation:
        context = _build_apo_context(f, branches.inputs, branches.valid, taps, z)
    structure, confidence, single = model.apo_repr_mix(
        z.detach(), encoded.lm_single.detach(), branch_z, branch_s, branches.valid
    )
    retain_branches = training and use_relation
    return ApoOutput(
        structure_pair=structure,
        confidence_pair=confidence,
        confidence_single=single,
        context=context,
        rollout=ApoRollout(branches.inputs, branch_z, branches.valid)
        if retain_branches
        else None,
    )


@torch.no_grad()
def run_single(
    model,
    f: FoldingInput,
    encoded: EncoderOutput,
    z: torch.Tensor,
    branches: ApoBranches,
    devices: list[int],
) -> ApoOutput:
    """Keep the base prediction and prepare apo confidence/utility evidence."""
    taps = []
    for branch, features in zip(branches.inputs, branches.features, strict=True):
        with torch.random.fork_rng(devices=devices):
            taps.append(model.collect_apo_features(branch, encoded.lm_single, features))
    context = _build_apo_context(f, branches.inputs, branches.valid, taps, z)
    return ApoOutput(z.detach(), z.detach(), encoded.lm_single.detach(), context)


def run_apo_joint(
    model,
    f: FoldingInput,
    encoded: EncoderOutput,
    z: torch.Tensor,
    num_recycles: int,
    max_combinations: int | None = None,
    *,
    training: bool = False,
    preselected: SelectedApoCombinations | None = None,
    apo_adapter: bool = True,
) -> ApoOutput:
    """Select apo assignments and dispatch to the adapter ON or OFF path.

    Training inherits the main recycle count. Inference uses two recycles.
    Extra feature/trunk work never advances the generation RNG.
    """
    config = model.apo_repr_mix_config
    cap = config.max_combinations if max_combinations is None else max_combinations
    if cap < 1:
        raise ValueError("max_combinations must be positive")
    if training and not apo_adapter:
        raise ValueError("Skipping apo trunks is inference-only")
    if not apo_adapter and not config.relation_depth:
        return ApoOutput(z.detach(), z.detach(), encoded.lm_single.detach(), None)
    devices = (
        [f.device.index if f.device.index is not None else torch.cuda.current_device()]
        if f.device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        with torch.no_grad():
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
            branch_features = [
                select_apo_structure_features(f, encoded.apo_features, assignments, index)
                for index in range(len(branches))
            ]
        selected = ApoBranches(branches, valid, branch_features)
        if apo_adapter:
            result = run_adapter(
                model,
                f,
                encoded,
                z,
                selected,
                devices,
                num_recycles if training else 2,
                training=training,
            )
        else:
            result = run_single(model, f, encoded, z, selected, devices)
        result.assignments = assignments
        return result
