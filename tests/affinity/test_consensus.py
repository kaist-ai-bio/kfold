from dataclasses import replace

import numpy as np
import pytest

from kfold.affinity.consensus import Evidence, consensus_candidate


def evidence():
    return [Evidence(f's{i:02}', 'target', f'ligand{i}', i,
                     np.array([5., 5., 20., 20.]), True, False) for i in range(10)]


def test_small_protein_different_sites_do_not_collapse_to_full_protein():
    rows = evidence()
    for i in range(5):
        rows[i] = replace(rows[i], distances=np.array([20.,20.,5.,5.]))
    result = consensus_candidate(rows)
    assert result['medoid_system_id'] == 's00'
    assert result['medoid_mean_jaccard'] == pytest.approx(4/9)
    assert not result['strict_stable']
    reversed_result = consensus_candidate(rows[::-1])
    assert reversed_result['medoid_system_id'] == result['medoid_system_id']


def test_refill_gaps_keep_original_rank_parity():
    rows = [replace(e, candidate_rank=2*i) for i,e in enumerate(evidence())]
    result = consensus_candidate(rows)
    assert result['half_sizes'] == [10,0]
    assert result['medoid_mean_jaccard'] == 1.
    assert result['split_half_jaccard'] == 0.
    assert not result['strict_stable']


@pytest.mark.parametrize('change', [dict(training_member=False), dict(synthetic=True)])
def test_heldout_and_synthetic_profiles_cannot_define_consensus(change):
    rows=evidence();rows[0]=replace(rows[0], **change)
    with pytest.raises(ValueError, match='forbidden'):
        consensus_candidate(rows)


def test_exact_cutoff_and_duplicate_ligand_rejection():
    rows=evidence();rows[0]=replace(rows[0],distances=np.full(4,15.))
    with pytest.raises(ValueError,match='Empty pocket'):
        consensus_candidate(rows)
    rows=evidence();rows[0]=replace(rows[0],ligand_identity=rows[1].ligand_identity)
    with pytest.raises(ValueError,match='Duplicate'):
        consensus_candidate(rows)
