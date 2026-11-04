import importlib.util
from pathlib import Path

import pytest

from kfold.affinity.preparation import digest
from kfold.affinity.shards import ShardStore

spec = importlib.util.spec_from_file_location('feature_campaign',
    Path(__file__).resolve().parents[2]/'scripts/affinity/feature_campaign.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def test_decoy_interval_covers_only_new_indices_without_tail_overrun():
    c = dict(systems=2796376,first_system_index=1459112,last_system_index=2796376,chunk_size=128)
    bounds = [campaign.chunk_interval(c,i) for i in range((1337265+127)//128)]
    assert bounds[0][0] == 1459112
    assert bounds[-1][0]+bounds[-1][1]-1 == 2796376
    assert sum(size for _,size in bounds) == 1337265
    assert all(a+n==b for (a,n),(b,_) in zip(bounds,bounds[1:]))
    with pytest.raises(ValueError,match='outside'):
        campaign.chunk_interval(c,len(bounds))
    assert campaign.chunk_interval(dict(systems=259,chunk_size=128),2) == (257,3)


@pytest.mark.parametrize('first,last', [(0,3),(4,3),(1,11),(True,3)])
def test_invalid_range_is_rejected(first,last):
    with pytest.raises(ValueError):
        campaign.index_interval(dict(systems=10,first_system_index=first,last_system_index=last))


def test_pilot_reuse_allows_bucket_but_rejects_seed_apo_and_contract_changes():
    task=dict(input_id='pair',system_index=1459112,total_tokens=42,apo_sha256='apo')
    contract=dict(checkpoint='checkpoint')
    receipt=dict(system_id='pair',system_index=1459112,seed=1459112,contract=contract,
                 actual_tokens=42,lm_dropout=.25,diffusion_executed=False,pp_stored=False)
    payload=dict(source_input=dict(task,bucket='1-256'),receipt=receipt,cache_identity_sha256=digest(receipt))
    campaign.validate_cached_record(payload,task,contract)
    for changed in [dict(task,apo_sha256='different'),dict(task,system_index=1459113)]:
        with pytest.raises(ValueError):
            campaign.validate_cached_record(payload,changed,contract)
    with pytest.raises(ValueError):
        campaign.validate_cached_record(payload,task,dict(checkpoint='different'))
    payload['cache_identity_sha256']='wrong'
    with pytest.raises(ValueError,match='hash'):
        campaign.validate_cached_record(payload,task,contract)


def test_sealed_pilot_readback_preserves_identity_and_rejects_manifest_changes(tmp_path):
    import json
    task=dict(input_id='pair',system_index=1459112,total_tokens=42)
    contract=dict(checkpoint='checkpoint')
    receipt=dict(system_id='pair',system_index=1459112,seed=1459112,contract=contract,
                 actual_tokens=42,lm_dropout=.25,diffusion_executed=False,pp_stored=False)
    payload=dict(source_input=dict(task,bucket='1-256'),receipt=receipt,cache_identity_sha256=digest(receipt))
    store=ShardStore(tmp_path/'pilot','pilot-contract')
    store.add('pair',payload);store.seal()
    path=store.entries['pair'][0]
    sealed=json.loads(path.with_suffix('.receipt.json').read_text())
    entry=dict(shard_path=str(path),shard_sha256=sealed['sha256'],
               cache_identity_sha256=digest(receipt),system_index=1459112)
    assert campaign.read_pilot_record(entry,task,contract)==payload
    with pytest.raises(ValueError,match='receipt changed'):
        campaign.read_pilot_record(dict(entry,shard_sha256='wrong'),task,contract)
    with pytest.raises(ValueError,match='manifest identity'):
        campaign.read_pilot_record(dict(entry,cache_identity_sha256='wrong'),task,contract)


def test_group_quota_uses_limits_instead_of_filesystem_capacity():
    text='/scratch 17681426756 85899345920 85899345920 - 3892890 5000000 5000000 -'
    result=campaign.quota_headroom(text,'/scratch')
    assert result['free_bytes']==(85899345920-17681426756)*1024
    assert result['free_files']==1107110
    with pytest.raises(ValueError):
        campaign.quota_headroom(text,'/appsdata')
    with pytest.raises(ValueError):
        campaign.quota_headroom('/scratch 1 0 0 - 1 0 0 -','/scratch')


def test_quota_guard_pauses_on_low_space_or_unavailable_output(monkeypatch):
    from types import SimpleNamespace
    policy=dict(mode='group_filesystem_reserve',group='bs0247',
        reserve_bytes_per_filesystem=3_000_000_000_000,inflight_margin_bytes=500_000_000_000)
    def run(args,**kwargs):
        fs=args[-1]
        return SimpleNamespace(stdout=f'{fs} 1000000000 10000000000 10000000000 - 1 5000000 5000000 -')
    monkeypatch.setattr(campaign.subprocess,'run',run)
    assert campaign.StorageReserveGuard(policy).check()['allowed']
    monkeypatch.setattr(campaign.subprocess,'run',lambda args,**k: SimpleNamespace(
        stdout=f'{args[-1]} 8000000000 10000000000 10000000000 - 1 5000000 5000000 -'))
    report=campaign.StorageReserveGuard(policy).check()
    assert not report['allowed'] and 'error' not in report
    monkeypatch.setattr(campaign.subprocess,'run',lambda *a,**k: SimpleNamespace(stdout=''))
    assert 'error' in campaign.StorageReserveGuard(policy).check()
