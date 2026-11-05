import pytest
import hashlib
import torch
from kfold.affinity.preparation import digest
from kfold.affinity.profile_corpus import consensus_from_prefix, validate_profile_input

def candidates(n=12,total=100):
    return [dict(candidate_rank=i,target_training_ligands=total,protein_key='source-target',
                 target_input_key='exact-target',input_id=str(i),ligand_sha=str(i),min_p_activity=7.) for i in range(n)]

def profiles(n=12,bad=()):
    return {str(i):dict(status='passed',pocket_residue_count=2,pocket_hlp_15a=.9 if i in bad else .2,
                       protein_min_expected_distance=[3.,4.,18.]) for i in range(n)}

def test_missing_earlier_evidence_cannot_be_skipped():
    p=profiles();del p['2']
    r=consensus_from_prefix(candidates(),p)
    assert r['status']=='pending' and r['next_rank']==2

def test_prefix_exhaustion_is_pending_not_unsupported():
    r=consensus_from_prefix(candidates(),profiles(bad=(0,1,2)))
    assert r['status']=='pending' and r['reason']=='extend_frozen_prefix'
    r=consensus_from_prefix(candidates(total=12),profiles(bad=(0,1,2)))
    assert r['status']=='unsupported'

def test_frozen_rank_parity_survives_refill_and_reordering():
    r=consensus_from_prefix(list(reversed(candidates())),profiles(bad=(1,3)))
    assert r['status']=='ready'
    assert r['consensus']['evidence_candidate_ranks']==[0,2,4,5,6,7,8,9,10,11]
    assert r['consensus']['half_sizes']==[6,4]

def test_mixed_target_prefix_rejected():
    c=candidates();c[-1]['target_input_key']='other'
    with pytest.raises(ValueError):consensus_from_prefix(c,profiles())

@pytest.mark.parametrize('changed',['sequence','smiles','apo_sha256','system_index'])
def test_raw_input_identity_mismatch_cannot_enter_consensus(changed):
    sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
    c=dict(input_id='id',system_index=17,input_sha=sha('AA'),ligand_sha=sha('CC'),
           apo_sha='a'*64,protein_tokens=2)
    c['target_input_key']=sha(c['input_sha']+':'+c['apo_sha'])
    receipt=dict(system_id='id',system_index=17,seed=17,contract={},lm_dropout=.25,
                 diffusion_executed=False,pp_stored=False)
    payload=dict(source_input=dict(input_id='id',system_index=17,sequence='AA',smiles='CC',apo_sha256='a'*64),
                 receipt=receipt,cache_identity_sha256=digest(receipt),protein_mask=torch.tensor([True,True,False]))
    validate_profile_input(payload,c,{})
    payload['source_input'][changed]=18 if changed=='system_index' else 'changed'
    with pytest.raises(ValueError):validate_profile_input(payload,c,{})
