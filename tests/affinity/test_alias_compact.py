import copy

import pytest
import torch

from test_compact import fixture
from kfold.affinity.alias_compact import (pack_alias_union, representation_id,
    resolve_historical_view, verify_alias_against_source)
from kfold.affinity.compact import pack_crop_union, read_crop
from kfold.affinity.head_inputs import compact_head_inputs
from kfold.affinity.shards import encode, decode, SHUFFLE_CODEC


def setup():
    source, views, receipt = fixture()
    source['bin_boundaries'] = torch.tensor([12.])
    plans = {}
    for key in ('b'*64, 'c'*64):
        plans[key] = dict(status='final', source_cache_identity_sha256=source['cache_identity_sha256'],
            views=copy.deepcopy(views), unavailable={}, crop_receipt=dict(receipt,
            historical_target_key=key, target_input_key='d'*64, training_ready=True,
            consensus_quality='stable', evidence_system_ids=[key]))
    plans['c'*64]['views']['target_consensus_200'] = [1, 4, 6, 7]
    return source, plans


def test_two_consensuses_share_one_tensor_union_and_exact_head_inputs():
    source, plans = setup()
    packed = decode(*encode(pack_alias_union(source, plans), SHUFFLE_CODEC)[:3])
    assert verify_alias_against_source(source, packed)['views_verified'] == 4
    assert len(packed['protein_mask']) == 8
    assert len(packed['crop_receipt']['representation_keys']) == 2
    for key, plan in plans.items():
        single = pack_crop_union(source, plan['views'], crop_receipt=plan['crop_receipt'])
        for view in plan['views']:
            actual = compact_head_inputs(packed, view, historical_key=key)
            expected = compact_head_inputs(single, view)
            assert all(torch.equal(actual[k], expected[k]) for k in actual)
    with pytest.raises(ValueError, match='Unavailable'):
        read_crop(packed, 'target_consensus_200')
    assert representation_id('sample','b'*64) != representation_id('sample','c'*64)


def test_unavailable_is_preserved_per_historical_key():
    source, plans = setup()
    plan = plans['c'*64]
    del plan['views']['target_consensus_200']
    plan['unavailable']['target_consensus_200'] = 'fewer_than_ten_valid_profiles'
    plan['crop_receipt']['consensus_quality'] = 'unavailable'
    packed = pack_alias_union(source, plans)
    verify_alias_against_source(source, packed)
    compact_head_inputs(packed, 'target_consensus_200', historical_key='b'*64)
    with pytest.raises(ValueError, match='fewer_than_ten'):
        compact_head_inputs(packed, 'target_consensus_200', historical_key='c'*64)
    compact_head_inputs(packed, 'query_350', historical_key='c'*64)


@pytest.mark.parametrize('mutation', ['query','identity','key','target','pending'])
def test_reject_inconsistent_plans(mutation):
    source, plans = setup(); plan = plans['c'*64]
    if mutation == 'query': plan['views']['query_200'] = [1,3,4]
    if mutation == 'identity': plan['source_cache_identity_sha256'] = 'e'*64
    if mutation == 'key': plan['crop_receipt']['historical_target_key'] = 'e'*64
    if mutation == 'target': plan['crop_receipt']['target_input_key'] = 'e'*64
    if mutation == 'pending': plan['status'] = 'pending'
    with pytest.raises(ValueError): pack_alias_union(source, plans)


def test_label_routing_rejects_wrong_source_unknown_key_and_tampering():
    source, plans = setup(); packed = pack_alias_union(source, plans)
    with pytest.raises(ValueError, match='different physical'):
        resolve_historical_view(packed, 'b'*64, 'query_200', source_system_id='other')
    with pytest.raises(ValueError, match='Unknown historical'):
        resolve_historical_view(packed, 'e'*64, 'target_consensus_200')
    packed['crop_receipt']['historical_views']['b'*64]['target_consensus_200'] = 'query_200'
    with pytest.raises(ValueError): verify_alias_against_source(source, packed)
    with pytest.raises(ValueError): resolve_historical_view(packed,'b'*64,'target_consensus_200')


def test_tensor_or_view_mapping_corruption_is_detected():
    source, plans = setup(); packed = pack_alias_union(source, plans)
    packed['z_pair_values'][0,0] += 8
    with pytest.raises(ValueError, match='mismatch'):
        verify_alias_against_source(source, packed)


def test_alias_bundle_publication_readback_and_resume(tmp_path):
    from test_bundles import source as make_source, CONTRACT
    from kfold.affinity.bundles import BundleStore, read_payload, file_sha
    from kfold.affinity.bundle_conversion import append_source
    from kfold.affinity.partial_bundles import PartialBundleCatalog, selection
    from kfold.affinity.alias_compact import POLICY, SCHEMA
    import json
    path, full = make_source(tmp_path,'raw',['a','deferred'])
    original_sha=file_sha(path)
    base=full['records']['a'];plans={}
    for key in ('b'*64,'c'*64):
        plan=copy.deepcopy(base)
        plan['crop_receipt'].update(training_ready=True,historical_target_key=key,target_input_key='d'*64)
        plans[key]=plan
    plans['c'*64]['views']['target_consensus_200']=[1,4,6,7]
    full['records']={'a':dict(status='final',schema=SCHEMA,source_cache_identity_sha256=base['source_cache_identity_sha256'],
        historical_plans=plans,crop_receipt={'training_ready':True})}
    full['selection']=selection(['a','deferred'],['a'])
    contract=dict(CONTRACT,coverage_scope='record_subset_v1',historical_alias_policy=POLICY)
    cat=PartialBundleCatalog(tmp_path/'catalog.sqlite');store=BundleStore(tmp_path/'bundle.sqlite',contract,16*1024**2)
    job=dict(receipt=json.loads(path.with_suffix('.receipt.json').read_text()),selection=full['selection'],identity_keys={'a':list(plans)})
    cat.claim('alias',store.path,contract,store.max_bytes,[job])
    def check(raw,compact):
        verify_alias_against_source(raw,compact)
        for key in plans: compact_head_inputs(compact,'target_consensus_200',historical_key=key)
    append_source(store,path,full,head_validator=check)
    assert append_source(store,path,full,head_validator=check)['reused']
    sealed=store.seal();cat.publish('alias',sealed)
    loaded=read_payload(cat.path,'a')
    assert loaded['schema']==SCHEMA and len(loaded['crop_receipt']['representation_keys'])==2
    from kfold.affinity.alias_compact import load_historical_head_inputs
    for key in plans:load_historical_head_inputs(cat.path,'a',key,'target_consensus_200')
    with pytest.raises(KeyError):read_payload(cat.path,'deferred')
    assert file_sha(path)==original_sha
    store.close();cat.close()
