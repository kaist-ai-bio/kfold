import pytest

from kfold.affinity.evidence import freeze_candidates, select_evidence, evidence_rejection


def row(i, **changes):
    return dict(dict(system_id=f's{i:02}', protein_key='target',
                     canonical_smiles=f'ligand{i}', p_activity=7.,
                     eligible_for_training=True, split='train'), **changes)


def test_validation_and_ineligible_labels_cannot_change_gate_or_order():
    rows = [row(i) for i in range(12)]
    baseline = freeze_candidates(rows)
    rows += [row(0, split='val', p_activity=1.),
             row(0, eligible_for_training=False, p_activity=2.),
             row(99, split='val')]
    assert freeze_candidates(rows[::-1]) == baseline
    assert baseline[0]['target_training_ligands'] == 12
    with pytest.raises(ValueError, match='Synthetic'):
        freeze_candidates([row(0, synthetic=True)])


def test_measured_training_repeats_aggregate_without_duplicate_ligand():
    result = freeze_candidates([row(0, system_id='z'), row(0, system_id='a', p_activity=5.)])
    assert len(result) == 1
    assert result[0]['source_system_id'] == 'a'
    assert result[0]['min_p_activity'] == 5.
    assert result[0]['train_label_count'] == 2
    assert select_evidence(result, {})['status'] == 'unsupported'


def profile(entropy=.5, count=5):
    return dict(status='passed', pocket_hlp_15a=entropy, pocket_residue_count=count)


def test_late_earlier_profile_blocks_completion_and_refill_keeps_ranks():
    candidates = freeze_candidates([row(i) for i in range(12)])
    profiles = {i: profile() for i in range(1,12)}
    assert select_evidence(candidates, profiles)['status'] == 'pending'
    profiles[0] = profile(.8)
    result = select_evidence(candidates, profiles)
    assert result['status'] == 'ready'
    assert [c['candidate_rank'] for c in result['selected']] == list(range(1,11))
    assert result['rejected'] == [dict(candidate_rank=0, reason='strong_high_entropy')]


def test_entropy_boundary_and_weak_training_evidence():
    candidate = freeze_candidates([row(0)])[0]
    assert evidence_rejection(candidate, profile(.7)) is None
    assert evidence_rejection(candidate, profile(.7001)) == 'strong_high_entropy'
    assert evidence_rejection(dict(candidate, min_p_activity=6.), profile(.99)) is None
    assert evidence_rejection(candidate, profile(None,0)) == 'empty_15a_pocket'
    with pytest.raises(ValueError, match='Invalid'):
        evidence_rejection(candidate, profile(float('nan')))


def test_exhaustion_distinct_from_not_yet_generated():
    candidates = freeze_candidates([row(i) for i in range(10)])
    profiles = {i: profile() for i in range(10)}
    profiles[2] = dict(status='excluded', reason='verified_oom')
    result = select_evidence(candidates, profiles)
    assert result['status'] == 'unsupported'
    assert result['reason'] == 'fewer_than_ten_valid_profiles'
    with pytest.raises(ValueError, match='contiguous'):
        select_evidence(candidates[1:], profiles)
