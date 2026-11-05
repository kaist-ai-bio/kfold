import copy

import pytest

from kfold.affinity.conversion import convert_sealed_shard, file_sha
from kfold.affinity.preparation import digest
from kfold.affinity.shards import ShardStore
from test_compact import fixture


def prepared(tmp_path):
    store = ShardStore(tmp_path / 'source', 'source-contract')
    records = {}
    for i in range(2):
        source, views, recipe = fixture()
        source['receipt']['system_id'] = str(i)
        source['cache_identity_sha256'] = digest(source['receipt'])
        store.add(str(i), source)
        records[str(i)] = dict(status='final', source_cache_identity_sha256=source['cache_identity_sha256'],
            views=views, crop_receipt=recipe, unavailable={})
    store.seal()
    path = next((tmp_path / 'source').glob('*.sqlite'))
    plan = dict(purpose='integration_pilot', source_sha256=file_sha(path), source_contract_id='source-contract',
        selector_contract='fixture-crops', records=records)
    return path, plan


def run(source, dest, plan, quota=lambda: None, validator=lambda s, c: None):
    return convert_sealed_shard(source, dest, plan, check_quota=quota, head_validator=validator)


def test_interrupted_conversion_resumes_without_source_mutation(tmp_path):
    source, plan = prepared(tmp_path)
    checked = []
    def interrupt(s, compact):
        checked.append(s['receipt']['system_id'])
        raise RuntimeError('simulated loader failure')
    with pytest.raises(RuntimeError):
        run(source, tmp_path/'out', plan, validator=interrupt)
    assert checked == ['0'] and not (tmp_path/'out/complete.json').exists()
    result = run(source, tmp_path/'out', plan)
    assert result['records'] == 2 and result['system_ids'] == ['0', '1']
    assert len(result['shards']) == 1 and file_sha(source) == plan['source_sha256']
    assert run(source, tmp_path/'out', plan) == result


@pytest.mark.parametrize('defect', ['missing', 'extra', 'pending', 'source_hash'])
def test_incomplete_or_unverified_plan_never_creates_output(tmp_path, defect):
    source, plan = prepared(tmp_path)
    if defect == 'missing': del plan['records']['0']
    if defect == 'extra': plan['records']['extra'] = plan['records']['0']
    if defect == 'pending': plan['records']['0']['status'] = 'pending'
    if defect == 'source_hash': plan['source_sha256'] = '0'*64
    with pytest.raises(ValueError): run(source, tmp_path/'out', plan)
    assert not (tmp_path/'out').exists()


def test_changed_plan_and_corrupted_output_cannot_be_reused(tmp_path):
    source, plan = prepared(tmp_path)
    result = run(source, tmp_path/'out', plan)
    changed = copy.deepcopy(plan)
    changed['records']['0']['crop_receipt']['config']['window'] = 20
    with pytest.raises(ValueError, match='changed on resume'): run(source, tmp_path/'out', changed)
    shard = result['shards'][0]['path']
    with open(shard, 'r+b') as stream:
        stream.seek(-10, 2); stream.write(b'corruption')
    with pytest.raises(ValueError, match='modified'): run(source, tmp_path/'out', plan)
    assert file_sha(source) == plan['source_sha256']


def test_quota_failure_blocks_writes_without_deleting_source(tmp_path):
    source, plan = prepared(tmp_path)
    def full(): raise RuntimeError('quota reserve')
    with pytest.raises(RuntimeError, match='quota'): run(source, tmp_path/'out', plan, quota=full)
    assert not (tmp_path/'out').exists() and file_sha(source) == plan['source_sha256']


def test_pilot_cannot_be_silently_promoted_to_training_cache(tmp_path):
    source, plan = prepared(tmp_path)
    plan['purpose'] = 'training_cache'
    with pytest.raises(ValueError, match='scientific crop receipts'):
        run(source, tmp_path/'out', plan)
    assert not (tmp_path/'out').exists()
