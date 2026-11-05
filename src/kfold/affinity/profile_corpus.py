"""Identity checks and prefix-safe consensus assembly for cached profiles."""
import hashlib
import numpy as np

from .consensus import Evidence, consensus_candidate
from .evidence import evidence_rejection
from .preparation import digest


def validate_profile_input(payload, candidate, contract):
    source, receipt = payload['source_input'], payload['receipt']
    index = candidate['system_index']
    if source['input_id'] != candidate['input_id'] or source['system_index'] != index:
        raise ValueError('Source identity mismatch')
    if any(receipt.get(k) != v for k,v in dict(system_id=candidate['input_id'],
            system_index=index,seed=index,contract=contract,lm_dropout=.25,
            diffusion_executed=False,pp_stored=False).items()):
        raise ValueError('Feature contract mismatch')
    if payload['cache_identity_sha256'] != digest(receipt):
        raise ValueError('Receipt digest mismatch')
    sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
    if (sha(source['sequence']) != candidate['input_sha'] or
            sha(source['smiles']) != candidate['ligand_sha'] or
            source['apo_sha256'] != candidate['apo_sha']):
        raise ValueError('Sequence, ligand or apo mismatch')
    if sha(candidate['input_sha']+':'+candidate['apo_sha']) != candidate['target_input_key']:
        raise ValueError('Exact target key mismatch')
    if int(payload['protein_mask'].sum()) != candidate['protein_tokens']:
        raise ValueError('Protein layout length mismatch')


def consensus_from_prefix(candidates, profiles):
    """A finite scheduling prefix never becomes an unsupported scientific target."""
    if not candidates:
        raise ValueError('Missing frozen candidate prefix')
    ordered=sorted(candidates,key=lambda c:c['candidate_rank'])
    total=ordered[0]['target_training_ligands']
    if (len({c['protein_key'] for c in ordered}) != 1 or
            len({c['target_input_key'] for c in ordered}) != 1 or
            any(c['target_training_ligands'] != total for c in ordered) or
            [c['candidate_rank'] for c in ordered] != list(range(len(ordered))) or len(ordered)>total):
        raise ValueError('Mixed or noncontiguous frozen prefix')
    chosen=[];rejected=[]
    for c in ordered:
        p=profiles.get(c['input_id'])
        if p is None:
            return dict(status='pending',reason='profile_not_available',next_rank=c['candidate_rank'])
        reason=evidence_rejection(c,p)
        if reason:
            rejected.append(dict(rank=c['candidate_rank'],reason=reason));continue
        chosen.append(Evidence(c['input_id'],c['protein_key'],c['ligand_sha'],c['candidate_rank'],
                               np.asarray(p['protein_min_expected_distance'],dtype=np.float32),True,False))
        if len(chosen)==10:
            result=consensus_candidate(chosen)
            result['distance_profile']=result['distance_profile'].tolist()
            return dict(status='ready',target_input_key=c['target_input_key'],rejected=rejected,consensus=result)
    if len(ordered)<total:
        return dict(status='pending',reason='extend_frozen_prefix',next_rank=len(ordered),rejected=rejected)
    return dict(status='unsupported',reason='fewer_than_ten_valid_profiles',rejected=rejected)
