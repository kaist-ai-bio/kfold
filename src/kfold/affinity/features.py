"""One indexed stochastic trunk view per system; no diffusion or PP persistence.

This extraction boundary consumes an already prepared/tokenized FoldingInput.
Large-scale shard publication and source-data adapters are separate work.
"""
from __future__ import annotations

import random
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import numpy as np
import torch

import kfold.constants as C
from .preparation import digest


@dataclass(frozen=True)
class FeatureContract:
    checkpoint_sha256: str
    config_sha256: str
    code_revision: str
    registry_sha256: str
    num_recycles: int = 3

    def __post_init__(self):
        for value in (self.checkpoint_sha256, self.config_sha256, self.registry_sha256):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("Immutable SHA256 receipts are required")
        if not self.code_revision or type(self.num_recycles) is not int or self.num_recycles < 0:
            raise ValueError("Code revision and nonnegative recycle count required")


@contextmanager
def indexed_rng(system_index: int):
    """Use the unique system index directly as seed; restore caller RNG states.

    Serialized execution only: global RNG isolation is not thread safe. Each
    process owns one GPU and executes one unpadded system at a time. A fixed
    seed alone does not guarantee identical output across kernels/hardware.
    """
    if type(system_index) is not int or not 0 <= system_index < 2**32:
        raise ValueError("system_index must be uint32")
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        try:
            random.seed(system_index)
            np.random.seed(system_index)
            torch.manual_seed(system_index)
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


def pack_full_cross(s_inputs, s_lm, z, logits, *, protein_mask, ligand_mask,
                    bin_boundaries, token_mapping):
    """Slice on device before host transfer; preserve directional z values."""
    p, l = protein_mask.bool(), ligand_mask.bool()
    if p.ndim != 1 or l.shape != p.shape or (p & l).any() or not p.any() or not l.any():
        raise ValueError("Disjoint nonempty protein and ligand masks required")
    if not (p | l).all():
        raise ValueError("Only unpadded protein/ligand tokens are supported")
    n = len(p)
    if z.ndim != 3 or z.shape[:2] != (n, n) or logits.ndim != 3 or logits.shape[:2] != (n, n):
        raise ValueError("Expected one full system [N,N,C]")
    if bin_boundaries.ndim != 1 or len(bin_boundaries) != logits.shape[-1]-1:
        raise ValueError("Distance bin metadata does not match logits")
    if not torch.all(bin_boundaries[1:] > bin_boundaries[:-1]):
        raise ValueError("Distance boundaries must increase")
    if any(x.ndim != 2 or x.shape[0] != n for x in (s_inputs, s_lm)):
        raise ValueError("Single representations do not align with tokens")
    active = (p[:, None] & l[None, :]) | (l[:, None] & (p | l)[None, :])
    pairs = active.nonzero()
    tri = (active & torch.ones_like(active).triu()).nonzero()
    values = z[pairs[:, 0], pairs[:, 1]]
    dist = logits[tri[:, 0], tri[:, 1]]
    reverse = logits[tri[:, 1], tri[:, 0]]
    if not torch.allclose(dist.float(), reverse.float(), atol=1e-5, rtol=0):
        raise ValueError("Distogram is not symmetric; cannot triangular-pack")
    if any(not torch.isfinite(x).all() for x in (s_inputs, s_lm, values, dist, bin_boundaries)):
        raise ValueError("Nonfinite feature payload")
    def bf16_cpu(x):
        value = x.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        if not torch.isfinite(value).all():
            raise ValueError("BF16 overflow")
        return value
    mapping = {}
    for key, value in token_mapping.items():
        if value.shape[0] != n:
            raise ValueError(f"Token mapping {key} has incompatible length")
        mapping[key] = value.detach().cpu().clone()
    return dict(schema="pli_full_cross_bf16_v1", s_inputs=bf16_cpu(s_inputs),
                s_lm=bf16_cpu(s_lm), z_pair_indices=pairs.to(device="cpu", dtype=torch.int32),
                z_pair_values=bf16_cpu(values), distogram_tri_indices=tri.to(device="cpu", dtype=torch.int32),
                distogram_tri_logits=bf16_cpu(dist),
                bin_boundaries=bin_boundaries.detach().float().cpu().clone(),
                protein_mask=p.cpu().clone(), ligand_mask=l.cpu().clone(), token_mapping=mapping)


def extract_indexed_features(model, f_input, *, system_id: str, system_index: int,
                             input_sha256: str, contract: FeatureContract):
    """Extract from full input with native inference LM dropout still enabled.

    The input digest identifies the exact apo/MSA/conformer/tokenization bundle,
    not just sequence+SMILES. No label or split is part of the stochastic seed.
    """
    if not system_id or not re.fullmatch(r"[0-9a-f]{64}", input_sha256):
        raise ValueError("System identity and exact input digest required")
    if not 0 < float(model.lm_dropout) < 1:
        raise ValueError("Native LM-to-pair dropout must be enabled")
    mask = f_input.token.pad_mask
    if mask.ndim != 2 or mask.shape[0] != 1 or not mask.all():
        raise ValueError("Extract one unpadded system at a time for seed stability")
    flags = [(m, m.training) for m in model.modules()]
    try:
        model.eval()  # run_trunk deliberately calls LM dropout with training=True.
        with indexed_rng(system_index), torch.inference_mode(), torch.autocast(
                device_type=f_input.device.type, dtype=torch.bfloat16):
            singles, lm, z = model.run_trunk(f_input, contract.num_recycles)
            dist = model.distogram_head(z)
            chain = f_input.token.chain_type[0]
            names = ("chain_type", "entity_id", "asym_id", "apo_uid", "sym_id",
                     "token_index", "org_token_index", "residue_index", "seq_token_index",
                     "center_index", "repr_index", "frame_index", "num_atoms")
            mapping = {k: getattr(f_input.token, k)[0] for k in names}
            result = pack_full_cross(singles[0], lm[0], z[0], dist["logits"][0],
                protein_mask=chain == C.ChainType.PROTEIN.value,
                ligand_mask=chain == C.ChainType.LIGAND.value,
                bin_boundaries=dist["bin_boundaries"], token_mapping=mapping)
    finally:
        for module, training in flags:
            module.training = training
    receipt = dict(system_id=system_id, system_index=system_index, seed=system_index,
                   input_sha256=input_sha256, contract=asdict(contract),
                   lm_dropout=float(model.lm_dropout), torch_version=str(torch.__version__),
                   device=str(f_input.device), diffusion_executed=False, pp_stored=False)
    result["receipt"] = receipt
    result["cache_identity_sha256"] = digest(receipt)
    return result
