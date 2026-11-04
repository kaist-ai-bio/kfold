import pytest
import torch

from kfold.affinity.crops import protein_distance_profile, query_views, select_windows


def payload(np=400, nl=3):
    n = np + nl
    p = torch.arange(n) < np
    cross = torch.cartesian_prod(torch.arange(np), torch.arange(np, n))
    # Distances increase with protein index, so the first residues are selected.
    logits = torch.full((len(cross), 64), -20.)
    logits[torch.arange(len(cross)), (cross[:, 0] * 63 // max(1, np - 1))] = 20.
    return dict(protein_mask=p, ligand_mask=~p, distogram_tri_indices=cross,
                distogram_tri_logits=logits.to(torch.bfloat16),
                bin_boundaries=torch.linspace(2 + 20/64, 22 - 20/64, 63),
                token_mapping={"asym_id": torch.cat((torch.zeros(np), torch.ones(nl)))})


def test_expected_distance_and_approved_nested_windows():
    d = payload()
    views, missing, profile = query_views(d)
    assert not missing
    assert profile[0].item() == pytest.approx(2 + 10/64)
    assert profile[-1].item() == pytest.approx(22 - 10/64)
    assert len(views["query_200"]) == 203
    assert len(views["query_350"]) == 353
    assert set(views["query_200"].tolist()) <= set(views["query_350"].tolist())
    assert views["query_200"][-3:].tolist() == [400, 401, 402]


def test_large_ligand_kept_and_small_view_unavailable():
    views, missing, _ = query_views(payload(400, 250))
    assert "query_200" in missing and "query_350" in views
    assert set(range(400, 650)) <= set(views["query_350"].tolist())
    assert len(views["query_350"]) <= 512


def test_mismatched_bins_and_missing_distograms_fail():
    d = payload(20)
    d["bin_boundaries"][0] += .1
    with pytest.raises(ValueError, match="bins"):
        protein_distance_profile(d)
    d = payload(20)
    d["distogram_tri_indices"] = d["distogram_tri_indices"][1:]
    d["distogram_tri_logits"] = d["distogram_tri_logits"][1:]
    with pytest.raises(ValueError, match="coverage"):
        protein_distance_profile(d)


def test_multichain_is_not_silently_treated_as_monomer():
    d = payload(20)
    d["token_mapping"]["asym_id"][10:20] = 3
    with pytest.raises(ValueError, match="Multi-chain"):
        select_windows(d, torch.arange(20), max_protein_tokens=200, max_tokens=256)
