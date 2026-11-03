import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kfold.affinity.features import FeatureContract, extract_indexed_features, indexed_rng, pack_full_cross
from kfold.affinity.preparation import (
    audit_records, extend_system_registry, full_cross_bytes, exposure_budget,
    summarize_draws, temporal_status,
)
import kfold.constants as C


def test_registry_survives_reorder_filter_extension_and_replicates():
    first = extend_system_registry(["b", "a", "b"])
    assert first == {"a": 0, "b": 1}
    assert extend_system_registry(["b", "c"], first) == {"a": 0, "b": 1, "c": 2}
    with pytest.raises(ValueError):
        extend_system_registry([], {"a": 0, "b": 0})


@pytest.mark.parametrize("value,expected", [
    ("2024-06-30", "on_or_before_cutoff"), ("2024-07-01", "after_cutoff"),
    ("2024", "straddles_cutoff"), ("2024-06", "on_or_before_cutoff"),
    (None, "missing_date"), ("2024-02-30", "invalid_date"),
])
def test_conservative_temporal_boundaries(value, expected):
    assert temporal_status(value, "2024-06-30") == expected


def test_manifest_audit_catches_cross_assay_leakage_and_preserves_unknown_dates():
    base = dict(sequence="AAAA", canonical_smiles="CC", origin="SAIR", endpoint="IC50")
    records = [dict(base, record_id="r1", system_id="s1", split="train", publication_date="2024"),
               dict(base, record_id="r2", system_id="s1", split="validation")]
    r = audit_records(records, cutoff="2024-06-30")
    assert r["unique_system_ids"] == 1 and r["rows"] == 2
    assert r["ligands_crossing_splits"] == 1
    assert r["temporal_counts"] == {"straddles_cutoff": 1, "missing_date": 1}
    records[1]["sequence"] = "BBBB"
    with pytest.raises(ValueError):
        audit_records(records)


def test_budget_and_nonuniform_exposures():
    assert full_cross_bytes(600, 32) == 23681024
    assert exposure_budget(20000, 512, 100, 200)["label_slots"] == 10240000
    r = summarize_draws(["a", "b", "c"], ["a"]*10 + ["b"])
    assert r["never_seen"] == 1 and r["maximum"] == 10
    with pytest.raises(ValueError):
        summarize_draws(["a"], ["heldout"])


def test_seed_replay_and_rng_restoration_on_exception():
    torch.manual_seed(19); np.random.seed(19); random.seed(19)
    expected = (torch.rand(3), np.random.rand(), random.random())
    torch.manual_seed(19); np.random.seed(19); random.seed(19)
    with indexed_rng(7):
        first = torch.rand(3)
    with pytest.raises(RuntimeError), indexed_rng(7):
        assert torch.equal(first, torch.rand(3))
        raise RuntimeError()
    assert torch.equal(expected[0], torch.rand(3))
    assert expected[1:] == (np.random.rand(), random.random())


def payload():
    z = torch.arange(4*4*2).reshape(4, 4, 2).float()
    d = z + z.transpose(0, 1)
    return dict(s_inputs=torch.ones(4, 3), s_lm=torch.ones(4, 3), z=z, logits=d,
                protein_mask=torch.tensor([1, 1, 0, 0], dtype=torch.bool),
                ligand_mask=torch.tensor([0, 0, 1, 1], dtype=torch.bool),
                bin_boundaries=torch.tensor([8.]), token_mapping={"residue_index": torch.arange(4)})


def test_pack_keeps_directions_excludes_pp_and_roundtrips_logits():
    x = payload(); r = pack_full_cross(**x)
    pairs = r["z_pair_indices"].long(); tri = r["distogram_tri_indices"].long()
    assert len(pairs) == 12 and len(tri) == 7
    assert not ((pairs[:, 0] < 2) & (pairs[:, 1] < 2)).any()
    assert torch.equal(r["z_pair_values"].float(), x["z"][pairs[:, 0], pairs[:, 1]])
    decoded = torch.zeros_like(x["logits"])
    decoded[tri[:, 0], tri[:, 1]] = r["distogram_tri_logits"].float()
    decoded[tri[:, 1], tri[:, 0]] = r["distogram_tri_logits"].float()
    assert torch.equal(decoded[pairs[:, 0], pairs[:, 1]], x["logits"][pairs[:, 0], pairs[:, 1]])
    x["logits"][0, 2, 0] += 1
    with pytest.raises(ValueError):
        pack_full_cross(**x)


def test_feature_extractor_replays_seed_and_restores_model_mode():
    class ToyTrunk(torch.nn.Module):
        lm_dropout = .5
        def run_trunk(self, inputs, num_recycles):
            assert not self.training and num_recycles == 3
            z = torch.nn.functional.dropout(torch.ones(1, 4, 4, 8), p=.5, training=True)
            return torch.ones(1, 4, 3), torch.ones(1, 4, 3), z
        def distogram_head(self, z):
            return dict(logits=z + z.transpose(1, 2), bin_boundaries=torch.arange(1, 8).float())
    fields = "entity_id asym_id apo_uid sym_id token_index org_token_index residue_index seq_token_index center_index repr_index frame_index num_atoms".split()
    token = SimpleNamespace(**{k: torch.arange(4)[None] for k in fields})
    token.pad_mask = torch.ones(1, 4, dtype=torch.bool)
    token.chain_type = torch.tensor([[C.ChainType.PROTEIN.value]*2 + [C.ChainType.LIGAND.value]*2])
    inp = SimpleNamespace(token=token, device=torch.device("cpu"))
    contract = FeatureContract("a"*64, "b"*64, "revision", "c"*64)
    model = ToyTrunk().train()
    def extract(index):
        return extract_indexed_features(model, inp, system_id="s", system_index=index,
                                        input_sha256="d"*64, contract=contract)
    a, b, c = extract(1), extract(1), extract(2)
    assert torch.equal(a["z_pair_values"], b["z_pair_values"])
    assert not torch.equal(a["z_pair_values"], c["z_pair_values"])
    assert model.training and not a["receipt"]["pp_stored"]
    token.pad_mask[0, -1] = False
    with pytest.raises(ValueError):
        extract(1)


@pytest.mark.parametrize("cross_only", [True, False])
def test_colleague_head_runs_on_current_main_and_backpropagates(cross_only):
    from kfold.model.modules.affinity_pairformer import AffinityPairformer
    head = AffinityPairformer(AffinityPairformer.Config(
        channel_s=16, channel_z=16, num_heads_attn=4, num_blocks=1, dropout=0.0,
        cross_pair_only=cross_only))
    mask = torch.ones(1, 4, dtype=torch.bool)
    protein = torch.tensor([[True, True, False, False]])
    result = head(s_inputs=torch.randn(1, 4, 16), s_lm=torch.randn(1, 4, 16),
                  z=torch.randn(1, 4, 4, 16), distogram_features=torch.rand(1, 4, 4, 3),
                  token_mask=mask, protein_mask=protein, ligand_mask=~protein)
    assert result.numel() == 1 and torch.isfinite(result).all()
    (result - 1).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
