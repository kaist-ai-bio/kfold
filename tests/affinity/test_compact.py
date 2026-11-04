import copy

import pytest
import torch

from kfold.affinity.compact import pack_crop_union, read_crop, verify_against_source
from kfold.affinity.features import pack_full_cross
from kfold.affinity.shards import ShardStore


def fixture():
    # Ligands are interleaved to detect assumptions about token block order.
    p = torch.tensor([1, 0, 1, 1, 0, 1, 1, 1], dtype=torch.bool)
    n = len(p)
    z = torch.arange(n * n * 2).reshape(n, n, 2).float()
    logits = z + z.transpose(0, 1)
    source = pack_full_cross(torch.arange(n * 3).reshape(n, 3).float(),
                             torch.arange(n * 4).reshape(n, 4).float(), z, logits,
                             protein_mask=p, ligand_mask=~p,
                             bin_boundaries=torch.tensor([8.]),
                             token_mapping={"residue_index": torch.arange(n) * 10})
    source.update(cache_identity_sha256="a" * 64,
                  receipt={"system_id": "sample", "system_index": 73, "seed": 73})
    views = dict(query_200=[0, 1, 2, 4], target_consensus_200=[1, 3, 4, 5],
                 query_350=[0, 1, 2, 4, 6])
    recipe = dict(selector_version="fixture-v1", config={"window": 10})
    return source, views, recipe


def test_union_roundtrip_and_directionality(tmp_path):
    source, views, recipe = fixture()
    compact = pack_crop_union(source, views, crop_receipt=recipe)
    assert compact["source_token_indices"].tolist() == list(range(7))
    store = ShardStore(tmp_path, "compact-test")
    store.add("sample", compact)
    store.seal()
    restored = ShardStore(tmp_path, "compact-test").get("sample")
    assert verify_against_source(source, restored)["views_verified"] == 3
    view = read_crop(restored, "query_200")
    pairs = view["z_pair_indices"].tolist()
    pl = view["z_pair_values"][pairs.index([0, 1])]
    lp = view["z_pair_values"][pairs.index([1, 0])]
    assert not torch.equal(pl, lp)
    assert len(compact["z_pair_indices"]) == 2 * 5 * 2 + 2 * 2
    assert sum(len(read_crop(compact, v)["z_pair_indices"]) for v in views) > len(compact["z_pair_indices"])
    # Caller-owned source remains untouched and still contains the discarded token.
    assert len(source["protein_mask"]) == 8
    assert restored["receipt"]["seed"] == 73


@pytest.mark.parametrize("bad_indices", [[0, 2, 4], [0, 1, 1, 4], [0, 1, 4, 99]])
def test_reject_missing_ligand_duplicate_and_out_of_bounds(bad_indices):
    source, views, recipe = fixture()
    views["query_200"] = bad_indices
    with pytest.raises(ValueError):
        pack_crop_union(source, views, crop_receipt=recipe)


def test_explicit_unavailable_consensus_is_not_query_fallback():
    source, views, recipe = fixture()
    del views["target_consensus_200"]
    with pytest.raises(ValueError):
        pack_crop_union(source, views, crop_receipt=recipe)
    compact = pack_crop_union(source, views, crop_receipt=recipe,
                             unavailable={"target_consensus_200": "no training evidence"})
    assert verify_against_source(source, compact)["views_verified"] == 2
    with pytest.raises(ValueError, match="no training evidence"):
        read_crop(compact, "target_consensus_200")


def test_tampering_and_recipe_changes_are_detected():
    source, views, recipe = fixture()
    compact = pack_crop_union(source, views, crop_receipt=recipe)
    altered_recipe = copy.deepcopy(recipe)
    altered_recipe["config"]["window"] = 20
    other = pack_crop_union(source, views, crop_receipt=altered_recipe)
    assert compact["cache_identity_sha256"] != other["cache_identity_sha256"]
    compact["z_pair_values"][0, 0] += 8
    with pytest.raises(ValueError, match="mismatch"):
        verify_against_source(source, compact)


def test_missing_sparse_entry_fails_before_publication():
    source, views, recipe = fixture()
    source["z_pair_indices"] = source["z_pair_indices"][1:]
    source["z_pair_values"] = source["z_pair_values"][1:]
    with pytest.raises(ValueError, match="coverage"):
        pack_crop_union(source, views, crop_receipt=recipe)
