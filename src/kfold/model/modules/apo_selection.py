"""Score apo combinations using pre-trunk evidence, before expensive generation."""

import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.modules.apo_repr_mix import make_apo_branches


def select_scored_combinations(scored: dict, top_k: int) -> dict | None:
    """Keep top-K finite B=1 assignments; None requests the normal fallback."""
    if top_k < 1:
        raise ValueError("apo_selection_top_k must be positive")
    valid = scored["valid"][0] & torch.isfinite(scored["scores"][0])
    available = valid.nonzero().flatten()
    if not available.numel():
        return None
    order = scored["scores"][0, available].argsort(descending=True, stable=True)
    chosen = available[order[:top_k]]
    indices = chosen.tolist()
    return {
        "branches": [scored["branches"][i] for i in indices],
        "valid": scored["valid"][:, chosen],
        "assignments": [[scored["assignments"][0][i] for i in indices]],
        "indices": chosen[None],
    }


@torch.no_grad()
def score_apo_combinations(model, f: FoldingInput, max_combinations: int = 3) -> dict:
    """Return utility logits and pre-trunk context for a batched B=1 input.

    This inference-only helper reads molecular inputs and available apo features.
    It does not run a sequence LM, folding trunk, diffusion, or confidence head,
    and never reads native structures or generated candidates. Scores are utility
    logits, not calibrated probabilities. Invalid combinations have score -inf.
    """
    if f.batch_size != 1:
        raise ValueError("Apo combination scoring currently requires batch_size=1")
    if max_combinations < 1:
        raise ValueError("max_combinations must be positive")
    mixers = getattr(model, "apo_repr_mix", None)
    selector = getattr(mixers, "apo_selector", None)
    if selector is None:
        raise ValueError("This model does not contain a trained apo selector")

    devices = (
        [f.device.index if f.device.index is not None else torch.cuda.current_device()]
        if f.device.type == "cuda"
        else []
    )
    # Pool sampling and any feature-module dropout must not consume the caller's
    # generation RNG stream. No training-time routing is performed here.
    with torch.random.fork_rng(devices=devices):
        if f.token.apo_repr_coords.shape[2] == 0:
            # A zero-width apo axis cannot be indexed by make_apo_branches.
            branches = [f]
            valid = torch.zeros(1, 1, device=f.device, dtype=torch.bool)
            assignments = [[{}]]
        else:
            branches, valid, assignments = make_apo_branches(f, max_combinations)
        s_inputs, _ = model.input_embedder(f)
        s_inputs = s_inputs.detach()
        taps = []
        get_module = getattr(model, "_get_model_module", lambda module: module)
        structure_encoder = get_module(model.prot_struct_encoder)
        apo_module = get_module(model.apo_module)
        for branch in branches:
            capture = {}
            if structure_encoder is not None:
                with model._encoder_on_device(structure_encoder.encoder, f.device):
                    structure_encoder(branch, feature_taps=capture)
            apo_module(branch, feature_taps=capture)
            b, length = branch.token.pad_mask.shape
            if "apo_single" not in capture or capture["apo_single"].shape[1] == 0:
                # Geometry-only apos remain usable by the selector through the
                # pair validity path; absent structural singles stay unavailable.
                capture["apo_single"] = s_inputs.new_zeros(b, 1, length, 2560)
                capture["apo_single_mask"] = torch.zeros(
                    b, 1, length, device=f.device, dtype=torch.bool
                )
            if capture["apo_pair"].shape[1] == 0:
                capture["apo_pair"] = capture["apo_pair"].new_zeros(
                    b, 1, length, length, apo_module.channel_apo
                )
                capture["apo_pair_mask"] = torch.zeros(
                    b, 1, length, length, device=f.device, dtype=torch.bool
                )
            taps.append(capture)

        context = {
            name: torch.cat([tap[name] for tap in taps], dim=1).detach()
            for name in ("apo_single", "apo_single_mask", "apo_pair", "apo_pair_mask")
        }
        observed = context["apo_single_mask"].any(-1)
        observed |= context["apo_pair_mask"].any(-1).any(-1)
        valid = valid & observed
        context["valid"] = valid
        scores = selector(f, context, s_inputs).masked_fill(~valid, -torch.inf)
    return {
        "scores": scores,
        "branches": branches,
        "valid": valid,
        "assignments": assignments,
        "context": context,
        "s_inputs": s_inputs,
    }
