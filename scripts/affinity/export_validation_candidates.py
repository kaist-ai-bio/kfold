#!/usr/bin/env python3
"""Export reviewable evaluation labels/systems without changing training data.

Source rows, repeated assays, bounds and input-QC flags remain explicit. These
are candidate identities, not finalized feature-cache identities or RNG seeds.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import duckdb
import pyarrow as pa
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, FilterCatalog
from audit_validation_overlap import rows
from build_validation_tautomer_index import tautomer_key


def main():
    p=argparse.ArgumentParser()
    p.add_argument('chembl',type=Path)
    p.add_argument('kinase',type=Path)
    p.add_argument('drugs',type=Path)
    p.add_argument('tautomers',type=Path)
    p.add_argument('out',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    RDLogger.DisableLog('rdApp.*')
    db=duckdb.connect();db.execute("SET threads=4");db.execute("SET memory_limit='4GB'")
    for name,path in [('c',a.chembl),('k',a.kinase),('drugs',a.drugs),('train_tautomers',a.tautomers)]:
        db.read_parquet(str(path)).create_view(name)
    # Select metadata candidates only. The unfiltered source remains archived.
    db.execute("""CREATE TEMP VIEW requested_chembl AS SELECT * FROM c
        WHERE document_indexed_after_35 AND NOT drug_parent_seen AND uniprot_seen
        AND (core_candidate OR (
            standard_type IN ('Ki','Kd','IC50') AND assay_type='B'
            AND confidence_score=9 AND target_type='SINGLE PROTEIN'
            AND standard_flag=1 AND data_validity_comment IS NULL AND parse_ok
            AND core_exclusion_reasons NOT LIKE '%annotated_variant%'
            AND standard_relation IN ('<','<=','>','>=')
            AND standard_units='nM' AND try_cast(standard_value AS DOUBLE)>0))""")
    parents=db.execute("SELECT parent_no_stereo,any_value(exact_graph) FROM (SELECT parent_no_stereo,exact_graph FROM requested_chembl UNION SELECT parent_no_stereo,exact_graph FROM drugs) GROUP BY 1").fetchall()
    keys=[tautomer_key(graph,parent) for parent,graph in parents]
    db.register('query_tautomers',pa.Table.from_pylist(keys, schema=pa.schema([('parent_no_stereo',pa.string()),('tautomer_hash',pa.string())])))
    db.execute("""CREATE TEMP TABLE query_overlap AS SELECT q.*,
        EXISTS(SELECT 1 FROM train_tautomers t WHERE t.tautomer_hash=q.tautomer_hash) tautomer_seen
        FROM query_tautomers q""")
    params=FilterCatalog.FilterCatalogParams();params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog=FilterCatalog.FilterCatalog(params)
    graphs=[x[0] for x in db.execute("SELECT DISTINCT exact_graph FROM requested_chembl WHERE exact_graph IS NOT NULL UNION SELECT DISTINCT exact_graph FROM drugs WHERE exact_graph IS NOT NULL").fetchall()]
    qc=[]
    for graph in graphs:
        mol=Chem.MolFromSmiles(graph)
        assert mol is not None
        alerts=sorted({m.GetDescription() for m in catalog.GetMatches(mol)})
        qc.append({'exact_graph':graph,'molecular_weight':Descriptors.MolWt(mol),
                   'source_fragment_count':len(Chem.GetMolFrags(mol)),
                   'source_formal_charge':Chem.GetFormalCharge(mol),
                   'ligand_heavy_atoms_qc':mol.GetNumHeavyAtoms(),
                   'pains_alerts':';'.join(alerts),'pains_alert_count':len(alerts)})
    db.register('chemistry_qc',pa.Table.from_pylist(qc))
    cid="CASE WHEN nullif(sequence,'') IS NOT NULL THEN 'valsys-' || sha256(length(sequence)::VARCHAR || ':' || sequence || ':' || length(exact_graph)::VARCHAR || ':' || exact_graph) END"
    db.execute(f"""CREATE TEMP VIEW chembl_candidate AS SELECT c.*,
        'chembl37_activity_'||activity_id::VARCHAR record_id,
        {cid} candidate_system_id,
        q.tautomer_hash,q.tautomer_seen,
        x.* EXCLUDE(exact_graph),
        molecular_weight<=1250 AND pains_alert_count=0 AND protein_tokens_full_sequence<=2000 sair_library_flags_pass,
        count(*) OVER(PARTITION BY assay_chembl_id,protein_uniprot,exact_graph,standard_type) same_assay_group_rows,
        min(pchembl_value) OVER(PARTITION BY assay_chembl_id,protein_uniprot,exact_graph,standard_type) same_assay_group_p_min,
        max(pchembl_value) OVER(PARTITION BY assay_chembl_id,protein_uniprot,exact_graph,standard_type) same_assay_group_p_max
        FROM requested_chembl c JOIN query_overlap q USING(parent_no_stereo)
        JOIN chemistry_qc x USING(exact_graph)
        WHERE q.tautomer_hash IS NOT NULL AND NOT q.tautomer_seen""")
    def export(name,query):
        path=a.out/name
        db.sql(query).write_parquet(str(path),compression='zstd')
        return path
    exports=[]
    exports.append(export('chembl-validation-candidates.parquet',"SELECT * FROM chembl_candidate WHERE core_candidate"))
    exports.append(export('chembl-censored-candidates.parquet',"SELECT * FROM chembl_candidate WHERE NOT core_candidate"))
    exports.append(export('chembl-validation-systems.parquet',"""SELECT candidate_system_id,sequence,exact_graph canonical_smiles,
        count(*) source_label_rows, list(DISTINCT protein_uniprot) uniprot_ids,
        list(DISTINCT protein_class_group) protein_classes, bool_and(sequence_seen) reference_sequence_seen
        FROM chembl_candidate WHERE core_candidate AND candidate_system_id IS NOT NULL GROUP BY 1,2,3"""))
    db.execute("""CREATE TEMP VIEW kinase_candidate AS SELECT k.*,
        d.exact_graph,d.parent_no_stereo,d.identity_resolution_level,d.structure_status,
        CASE WHEN nullif(reference_sequence,'') IS NOT NULL THEN 'valsys-'||sha256(length(reference_sequence)::VARCHAR||':'||reference_sequence||':'||length(d.exact_graph)::VARCHAR||':'||d.exact_graph) END candidate_system_id,
        q.tautomer_hash,q.tautomer_seen,x.* EXCLUDE(exact_graph),
        CASE WHEN measurement_status='measured' THEN value<50 ELSE NULL END hit_at_1um_lt50
        FROM k JOIN drugs d USING(drug_record_id) JOIN query_overlap q USING(parent_no_stereo)
        JOIN chemistry_qc x USING(exact_graph)""")
    assert db.execute("SELECT count(*) FROM kinase_candidate").fetchone()[0]==db.execute("SELECT count(*) FROM k").fetchone()[0]
    assert db.execute("SELECT count(*) FROM kinase_candidate WHERE measurement_status='measured' AND (value IS NULL OR dose_uM!=1 OR unit!='percent')").fetchone()[0]==0
    exports.append(export('kinase-selectivity-labels.parquet',"SELECT * FROM kinase_candidate WHERE measurement_status='measured'"))
    exports.append(export('kinase-selectivity-missing.parquet',"SELECT * FROM kinase_candidate WHERE measurement_status!='measured'"))
    exports.append(export('kinase-selectivity-systems.parquet',"""SELECT candidate_system_id,reference_sequence AS "sequence",exact_graph canonical_smiles,
        list(DISTINCT drug_name) drug_names,list(DISTINCT target_name) target_names,
        list(DISTINCT uniprot) uniprot_ids,count(*) measured_label_rows,
        bool_and(reference_sequence_seen) reference_sequence_seen
        FROM kinase_candidate WHERE measurement_status='measured' AND candidate_system_id IS NOT NULL GROUP BY 1,2,3"""))
    summary={
        'status':'candidate_metadata_exported_not_feature_ready',
        'chembl':rows(db,"""SELECT core_candidate,count(*) labels,count(DISTINCT candidate_system_id) systems,
            count(DISTINCT exact_graph) ligands,count(DISTINCT protein_uniprot) proteins,
            sum(sair_library_flags_pass::INT) sair_library_pass_labels,
            sum((same_assay_group_rows>1)::INT) rows_in_repeated_assay_groups,
            sum((same_assay_group_p_max-same_assay_group_p_min>1)::INT) rows_in_assay_groups_with_gt1log_range
            FROM chembl_candidate GROUP BY 1"""),
        'chembl_by_family':rows(db,"SELECT protein_class_group,count(*) labels,count(DISTINCT candidate_system_id) systems FROM chembl_candidate WHERE core_candidate GROUP BY 1 ORDER BY 1"),
        'chembl_by_endpoint':rows(db,"SELECT standard_type,count(*) labels FROM chembl_candidate WHERE core_candidate GROUP BY 1 ORDER BY 1"),
        'chembl_by_document_release':rows(db,"SELECT document_first_chembl_release,count(*) labels,count(DISTINCT document_chembl_id) documents FROM chembl_candidate WHERE core_candidate GROUP BY 1 ORDER BY 1"),
        'chembl_input_size_proxy':rows(db,"""SELECT count(*) systems,
            quantile_cont(p,[0.5,0.9,0.99]) protein_length_quantiles,max(p) protein_max,
            quantile_cont(l,[0.5,0.9,0.99]) ligand_heavy_atom_quantiles,max(l) ligand_max,
            quantile_cont(p+l,[0.5,0.9,0.99]) combined_size_quantiles,
            sum((p+l>1024)::INT) combined_gt1024,sum((p+l>2048)::INT) combined_gt2048,
            sum((p+l>4096)::INT) combined_gt4096,
            sum(2*p::BIGINT*l+l::BIGINT*l) directional_pl_lp_ll_pair_positions
            FROM (SELECT DISTINCT candidate_system_id,protein_tokens_full_sequence p,ligand_heavy_atoms_qc l
                  FROM chembl_candidate WHERE core_candidate AND candidate_system_id IS NOT NULL)"""),
        'input_size_scope':'Full reference sequence length and ligand heavy atoms; not actual K-Fold tokenization/crops. Pair positions exclude PP and are not a storage-byte estimate.',
        'tautomer_overlap_removed':rows(db,"SELECT c.core_candidate,count(*) labels,count(DISTINCT c.exact_graph) ligands FROM requested_chembl c JOIN query_overlap q USING(parent_no_stereo) WHERE q.tautomer_seen GROUP BY 1"),
        'kinase':rows(db,"SELECT measurement_status,count(*) row_count,count(DISTINCT candidate_system_id) systems,count(DISTINCT drug_record_id) drugs,count(DISTINCT target_record_id) targets,sum(hit_at_1um_lt50::INT) hits,sum((candidate_system_id IS NULL)::INT) missing_input_identity FROM kinase_candidate GROUP BY 1"),
        'kinase_seen_strata':rows(db,"SELECT drug_parent_seen,tautomer_seen,uniprot_drug_parent_seen,uniprot_seen,count(*) pairs FROM kinase_candidate WHERE measurement_status='measured' GROUP BY ALL"),
        'source_policy':'ChEMBL documents first indexed36/37; no publication-year or SAIR-release-date cut. Global ligand-overlap screening against full training candidate population, including added BindingDB. Kinase training records retained.',
        'pending':['Source identity/assay duplicate adjudication','Final target aliases and experimental constructs','Final feature-input canonicalization and tokenizer checks','Frozen evaluation versus final-test allocation'],
        'identity_policy':'candidate_system_id hashes a length-delimited full sequence and exact RDKit graph; not a production cache ID or seed. Input chemistry unchanged. Tautomer keys are conservative overlap flags only.',
    }
    for path in exports:
        db.read_parquet(str(path)).create_view('check_output',replace=True)
        if 'labels' in path.name or path.name=='chembl-validation-candidates.parquet':
            key='observation_id' if path.name.startswith('kinase') else 'activity_id'
            n,unique=db.execute(f'SELECT count(*),count(DISTINCT {key}) FROM check_output').fetchone()
            assert n==unique
    summary['files']=[{'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in exports]
    (a.out/'validation-package.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
