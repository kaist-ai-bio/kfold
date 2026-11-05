import copy
import hashlib
import pytest

from kfold.affinity.crop_plan import stable_record_plan
from kfold.affinity.preparation import digest
from test_crops import payload


def fixture():
    source=payload(400,3)
    source['source_input']=dict(input_id='query',system_index=99,sequence='A'*400,smiles='CCC',apo_sha256='a'*64)
    source['receipt']=dict(system_id='query',system_index=99,seed=99,contract={},lm_dropout=.25,diffusion_executed=False,pp_stored=False)
    source['cache_identity_sha256']=digest(source['receipt'])
    sha=lambda x:hashlib.sha256(x.encode()).hexdigest()
    sequence=sha('A'*400);target=sha(sequence+':'+'a'*64)
    rows=[dict(input_id=str(i),ligand_sha=str(i),candidate_rank=i,target_training_ligands=10,
        protein_key='historical',target_input_key=target,input_sha=sequence,apo_sha='a'*64,min_p_activity=7.) for i in range(10)]
    profile=dict(status='passed',pocket_residue_count=50,pocket_hlp_15a=.2,
        protein_min_expected_distance=[20.]*350+[5.]*50,target_input_key=target,
        input_sha=sequence,apo_sha='a'*64,protein_token_mapping={'asym_id':[0.]*400})
    return source,rows,{str(i):copy.deepcopy(profile) for i in range(10)}


def plan(source,rows,profiles,keys=('historical',),retain_unstable=False):
    return stable_record_plan(source,rows,profiles,historical_keys=keys,feature_contract={},
        candidate_provenance=dict(candidate_manifest_sha256='b'*64,frontier_sha256='c'*64),
        retain_unstable=retain_unstable)


def test_exact_training_consensus_attaches_without_using_query_label():
    source,rows,profiles=fixture()
    result=plan(source,rows,profiles)
    assert result['status']=='final' and result['crop_receipt']['training_ready']
    assert result['views']['target_consensus_200'] != result['views']['query_200']
    # A different ligand/provenance reuses the same frozen protein pocket.
    source['source_input']['smiles']='NNN'
    source['source_input']['synthetic']=True
    other=plan(source,rows,profiles)
    assert other['views']['target_consensus_200']==result['views']['target_consensus_200']
    assert other['crop_receipt']['consensus_sha256']==result['crop_receipt']['consensus_sha256']


def test_missing_earlier_rank_and_aliases_stay_pending():
    source,rows,profiles=fixture()
    assert plan(source,rows,profiles,keys=('historical','other'))['reason']=='historical_target_alias_review'
    del profiles['0']
    assert plan(source,rows,profiles)['status']=='pending'


def test_unstable_consensus_is_not_silently_used_or_discarded():
    source,rows,profiles=fixture()
    for i in range(1,10,2):
        profiles[str(i)]['protein_min_expected_distance']=[5.]*50+[20.]*350
    assert plan(source,rows,profiles)['reason']=='unstable_consensus_policy_review'
    retained=plan(source,rows,profiles,retain_unstable=True)
    assert retained['status']=='final'
    assert 'target_consensus_200' in retained['views']
    receipt=retained['crop_receipt']
    assert receipt['strict_stable'] is False
    assert receipt['consensus_quality']=='unstable'
    assert receipt['unstable_consensus_policy']=='retain_flagged'
    assert receipt['training_inclusion_decided'] is False
    assert receipt['medoid_mean_jaccard'] < .5 or receipt['split_half_jaccard'] < .5


@pytest.mark.parametrize('defect',['apo','mapping','seed'])
def test_mismatched_apo_mapping_or_dropout_seed_rejected(defect):
    source,rows,profiles=fixture()
    if defect=='apo':source['source_input']['apo_sha256']='d'*64
    if defect=='mapping':profiles['0']['protein_token_mapping']['asym_id'][0]=2.
    if defect=='seed':source['receipt']['seed']=100
    with pytest.raises(ValueError):plan(source,rows,profiles)
