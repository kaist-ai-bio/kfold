#!/usr/bin/env python3
"""Version two explicit stereochemistry corrections, retaining source identities."""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from audit_validation_overlap import chemical_keys


def system_id(sequence, graph):
    value=f'{len(sequence)}:{sequence}:{len(graph)}:{graph}'
    return 'valsys-'+hashlib.sha256(value.encode()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ['primary','references','identity_audit','training_keys','manifest','out']:
        p.add_argument(name,type=Path)
    a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    RDLogger.DisableLog('rdApp.*')
    refs={r['source_name']:r for r in json.loads(a.references.read_text())}
    records=pq.read_table(a.primary).to_pylist()
    drugs={r['drug_name']:r['exact_graph'] for r in records}
    supported={r['source_name'] for r in json.loads(a.identity_audit.read_text())['drugs'] if r['independent_graph_and_name_support']}
    assert set(drugs)<=supported|set(refs)
    changes={}
    for name,ref in refs.items():
        mol=Chem.MolFromSmiles(ref['canonical_smiles'])
        assert mol is not None and Chem.MolToInchiKey(mol)==ref['standard_inchi_key']
        new=chemical_keys(ref['canonical_smiles']);old=chemical_keys(drugs[name])
        assert new['parent_no_stereo']==old['parent_no_stereo'], 'Non-stereo parent change needs separate adjudication'
        if new['exact_graph']!=old['exact_graph']:
            changes[name]=dict(ref,old_exact_graph=old['exact_graph'],new_exact_graph=new['exact_graph'],
                               old_inchikey=Chem.MolToInchiKey(Chem.MolFromSmiles(drugs[name])))
    assert set(changes)=={'Abrocitinib','Pralsetinib'}
    db=duckdb.connect();db.execute('SET threads=2');db.execute("SET memory_limit='2GB'")
    db.read_parquet(str(a.training_keys)).create_view('training_keys')
    db.read_parquet(str(a.manifest)).create_view('training_labels')
    db.register('corrections',pa.Table.from_pylist(list(changes.values())))
    exact_seen={r[0] for r in db.execute('SELECT DISTINCT k.exact_graph FROM training_keys k JOIN corrections c ON k.exact_graph=c.new_exact_graph').fetchall()}
    pairs=set(db.execute('SELECT DISTINCT l.protein_uniprot,k.exact_graph FROM training_labels l JOIN training_keys k USING(canonical_smiles) JOIN corrections c ON k.exact_graph=c.new_exact_graph').fetchall())
    corrected=0
    for row in records:
        name=row['drug_name']
        for key in ['exact_graph','candidate_system_id','drug_exact_seen','uniprot_drug_exact_seen']:
            row['source_'+key]=row[key]
        row['input_chemistry_corrected']=name in changes
        row['input_identity_review']='independent_chembl_graph_and_name_support'
        row['input_identity_reference']=None
        if name in refs:
            row['input_identity_reference']=refs[name]['url']
            row['input_identity_review']='reviewed_named_chembl_reference'
        if name in changes:
            graph=changes[name]['new_exact_graph']
            row['exact_graph']=graph
            row['candidate_system_id']=system_id(row['reference_sequence'],graph)
            row['drug_exact_seen']=graph in exact_seen
            row['uniprot_drug_exact_seen']=(row['uniprot'],graph) in pairs
            corrected+=1
        row['canonical_smiles']=row['exact_graph']
    # Non-stereo parent and associated overlap flags are unchanged by construction.
    # Source chemical evidence remains in source_* fields and original packages.
    assert len(records)==len({r['observation_id'] for r in records})
    assert len(records)==len({r['candidate_system_id'] for r in records})
    labels=a.out/'kinase-primary-labels.parquet'
    pq.write_table(pa.Table.from_pylist(records),labels,compression='zstd')
    db.read_parquet(str(labels)).create_view('corrected')
    db.sql('SELECT candidate_system_id,reference_sequence AS "sequence",canonical_smiles,drug_name,target_name,uniprot,input_identity_review,input_identity_reference,input_chemistry_corrected FROM corrected').write_parquet(str(a.out/'kinase-primary-systems.parquet'),compression='zstd')
    summary={'labels':len(records),'corrected_label_and_system_rows':corrected,'corrections':changes,
             'training_changed':False,'experimental_values_changed':False,'source_catalog_changed':False,
             'source_scope':'Primary cohort only; original full panel and source identities preserved. Parent/tautomer overlap unchanged; exact graph and target-drug overlap recomputed against the full training-candidate manifest.',
             'remaining':'Reference protein constructs and final feature tokenization/crops remain unfinalized.',
             'files':[{'name':x.name,'bytes':x.stat().st_size,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in sorted(a.out.glob('*.parquet'))]}
    (a.out/'kinase-identity-correction.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('IDENTITY_CORRECTIONS_VERIFIED',len(records),corrected,flush=True)


if __name__=='__main__':
    main()
