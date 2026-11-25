# Copyright 2026 Korea Advanced Institute of Science and Technology (KAIST)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the Ratio745 affinity head on the v1.1 trunk representations."""

import math
from pathlib import Path

import torch
from torch import Tensor

from kfold.constants import ChainType
from kfold.data.types.model_input import FoldingInput
from kfold.model import KFold
from kfold.model.model import AFFINITY_BUNDLE_FORMAT
from kfold.model.modules.affinity_pairformer import JointAffinityHead


def load_affinity_head(
    path: str | Path, device: torch.device | str = "cpu"
) -> JointAffinityHead:
    """Load the cross-only joint head and verify all parameter keys and shapes."""
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return _affinity_head_from_state(state, device)


def _affinity_head_from_state(
    state: dict, device: torch.device | str
) -> JointAffinityHead:
    if not isinstance(state, dict):
        raise ValueError("Expected a saved joint affinity checkpoint.")
    if state.get("format") == AFFINITY_BUNDLE_FORMAT:
        state = state["affinity"]
    config = state.get("config", {})
    if state.get("task") != "joint" or config.get("variant") != "colleague_cross":
        raise ValueError("Expected a joint colleague_cross affinity checkpoint.")
    head = JointAffinityHead(
        channel_s=config["channel_s"],
        channel_z=config["channel_z"],
        blocks=config["blocks"],
        heads=config["heads"],
        dropout=config["dropout"],
    )
    head.load_state_dict(state["model"], strict=True)
    return head.requires_grad_(False).eval().to(device)


def load_affinity_models(
    checkpoint: str | Path,
    model_path: str | Path,
    *,
    device: torch.device,
    cache_dir: str | Path | None,
    use_rna_encoder: bool,
    cpu_offload: bool,
    kernel_backend: str,
) -> tuple[KFold, JointAffinityHead, str]:
    """Resolve the backbone and affinity head for a CLI inference run."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    head = _affinity_head_from_state(state, device)
    model_options = {
        "device": device,
        "cache_dir": cache_dir,
        "use_rna_encoder": use_rna_encoder,
        "cpu_offload": cpu_offload,
        "kernel_backend": kernel_backend,
    }
    if state.get("format") == AFFINITY_BUNDLE_FORMAT:
        backbone = state["backbone"]
        model = KFold.from_weights(
            backbone["model"], backbone["config_yaml"], **model_options
        )
        reference = backbone["reference"]
    else:
        model = KFold.from_pretrained(model_path, **model_options)
        reference = str(model_path)
    return model, head, reference


def _feature_precision(tensor: Tensor) -> Tensor:
    # Match the BF16 storage and FP32 feature construction used during training.
    return tensor.to(device="cpu", dtype=torch.bfloat16).float()


def _query_positions(protein: Tensor, ligand: Tensor, distances: Tensor) -> Tensor:
    """Select the query_200 view by adding complete 10-token neighborhoods."""
    window = 10
    budget = min(200, 256 - len(ligand))
    if len(protein) < window or budget < window:
        raise ValueError("A complete protein neighborhood does not fit in query_200.")
    if not torch.isfinite(distances).all():
        raise ValueError("Nonfinite distogram distances.")

    selected = set()
    for center in torch.argsort(distances, stable=True).tolist():
        start = min(max(0, center - window // 2), len(protein) - window)
        added = set(range(start, start + window)) - selected
        if len(selected) + len(added) > budget:
            break
        selected.update(added)
    protein_crop = protein[sorted(selected)]
    return torch.cat((protein_crop, ligand)).sort().values


def query_200_head_inputs(
    s_inputs: Tensor,
    s_lm: Tensor,
    z: Tensor,
    distogram: dict[str, Tensor],
    *,
    protein: Tensor,
    ligand: Tensor,
) -> dict[str, Tensor]:
    """Construct the inputs used by the head from one unbatched dense trunk."""
    if not len(protein) or not len(ligand):
        raise ValueError("Affinity prediction requires protein and ligand tokens.")

    logits = distogram["logits"]
    edges = distogram["bin_boundaries"].float().cpu()
    bins = logits.shape[-1]
    if bins < 2:
        raise ValueError("Affinity prediction needs at least two distance bins.")
    step = 20.0 / bins
    expected_edges = torch.linspace(2.0 + step, 22.0 - step, bins - 1)
    if edges.shape != expected_edges.shape or not torch.allclose(
        edges, expected_edges, atol=1e-5, rtol=0
    ):
        raise ValueError("Distogram bins do not match the affinity head.")
    centers = torch.linspace(2.0 + step / 2, 22.0 - step / 2, bins)
    protein, ligand = protein.to(z.device), ligand.to(z.device)
    cross = _feature_precision(logits[protein[:, None], ligand[None, :]])
    distances = (cross.softmax(-1) * centers).sum(-1).amin(-1)
    crop = _query_positions(protein.cpu(), ligand.cpu(), distances)
    ligand_mask = torch.isin(crop, ligand.cpu())
    active = ligand_mask[:, None] | ligand_mask[None, :]
    crop_device = crop.to(z.device)

    rows, cols = active.nonzero(as_tuple=True)
    pair_values = _feature_precision(
        z[crop_device[rows.to(z.device)], crop_device[cols.to(z.device)]]
    )
    crop_z = pair_values.new_zeros((len(crop), len(crop), z.shape[-1]))
    crop_z[rows, cols] = pair_values

    rows, cols = active.triu().nonzero(as_tuple=True)
    probs = _feature_precision(
        logits[crop_device[rows.to(z.device)], crop_device[cols.to(z.device)]]
    ).softmax(-1)
    summaries = torch.stack(
        (
            probs[:, centers <= 8.0].sum(-1),
            (probs * centers).sum(-1),
            -(probs * probs.clamp_min(1e-8).log()).sum(-1) / math.log(bins),
        ),
        dim=-1,
    )
    crop_distogram = summaries.new_zeros((len(crop), len(crop), 3))
    crop_distogram[rows, cols] = summaries
    crop_distogram[cols, rows] = summaries
    features = {
        "s_inputs": _feature_precision(s_inputs[crop_device])[None],
        "s_lm": _feature_precision(s_lm[crop_device])[None],
        "z": crop_z[None],
        "distogram_features": crop_distogram[None],
        "token_mask": torch.ones(1, len(crop), dtype=torch.bool),
        "protein_mask": (~ligand_mask)[None],
        "ligand_mask": ligand_mask[None],
    }
    if any(not torch.isfinite(value).all() for value in features.values()):
        raise ValueError("Nonfinite affinity head input.")
    return features


@torch.inference_mode()
def predict_affinity(
    model: KFold,
    head: JointAffinityHead,
    f_input: FoldingInput,
    *,
    seed: int,
    num_recycles: int = 3,
) -> dict[str, float]:
    """Run the trunk and both affinity readouts for one protein–ligand query."""
    device = model.device
    f_input = f_input.to(device)
    if not f_input.is_batched:
        f_input = f_input.add_batch_dim()
    if f_input.batch_size != 1:
        raise ValueError("Affinity inference expects one complex.")

    valid = f_input.token.pad_mask[0].bool()
    chain = f_input.token.chain_type[0]
    asym = f_input.token.asym_id[0]
    protein_mask = valid & (chain == ChainType.PROTEIN)
    ligand_mask = valid & (chain == ChainType.LIGAND)
    if (
        not torch.equal(protein_mask | ligand_mask, valid)
        or asym[protein_mask].unique().numel() != 1
        or asym[ligand_mask].unique().numel() != 1
    ):
        raise ValueError("Affinity inference requires one protein chain and one ligand.")
    protein = protein_mask.nonzero().flatten()
    ligand = ligand_mask.nonzero().flatten()

    cuda = device.type == "cuda"
    with torch.random.fork_rng(devices=[device] if cuda else []):
        torch.random.default_generator.manual_seed(seed)
        if cuda:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        # The head was trained on the single trunk before apo-representation mixing.
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda):
            encoded = model.encode_input(f_input)
            s_inputs, s_lm, z = model.run_trunk(f_input, num_recycles, encoded.lm_single)
        with torch.autocast(device.type, enabled=False):
            dist = model.distogram_head(z.float())

    features = query_200_head_inputs(
        s_inputs[0],
        s_lm[0],
        z[0],
        {"logits": dist["logits"][0], "bin_boundaries": dist["bin_boundaries"]},
        protein=protein,
        ligand=ligand,
    )
    del z, dist
    features = {name: value.to(device) for name, value in features.items()}
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda):
        prediction = head(**features)
    logit = prediction["logits"].float()
    scores = {
        "p_activity": prediction["value"].float().item(),
        "binary_logit": logit.item(),
        "binding_probability": logit.sigmoid().item(),
    }
    if not all(math.isfinite(score) for score in scores.values()):
        raise ValueError("Nonfinite affinity prediction.")
    return scores
