#!/usr/bin/env python3
"""Label-only coverage and metric eligibility; does not select a trained model."""
import argparse
import json
from pathlib import Path

import duckdb
from audit_validation_overlap import rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('chembl',type=Path)
    p.add_argument('kinase',type=Path)
    p.add_argument('out',type=Path)
    a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    db=duckdb.connect();db.execute('SET threads=2');db.execute("SET memory_limit='2GB'")
    db.read_parquet(str(a.chembl)).create_view('c')
    db.read_parquet(str(a.kinase)).create_view('k')
    db.execute("""CREATE TEMP VIEW assay_groups AS SELECT assay_chembl_id,protein_uniprot,standard_type,
        count(*) label_rows,count(DISTINCT exact_graph) ligands,min(pchembl_value) p_min,max(pchembl_value) p_max
        FROM c GROUP BY 1,2,3""")
    db.execute("""CREATE TEMP VIEW drug_groups AS SELECT drug_record_id,drug_name,count(*) measured_pairs,
        sum(hit_at_1um_lt50::INT) hits,sum((NOT hit_at_1um_lt50)::INT) nonhits,
        sum(uniprot_drug_parent_seen::INT) training_pair_overlap
        FROM k GROUP BY 1,2""")
    db.sql('SELECT * FROM assay_groups ORDER BY ligands DESC').write_parquet(str(a.out/'chembl-assay-groups.parquet'),compression='zstd')
    db.sql('SELECT *,hits>0 AND nonhits>0 auroc_defined FROM drug_groups ORDER BY drug_name').write_parquet(str(a.out/'kinase-per-drug-coverage.parquet'),compression='zstd')
    summary={
        'chembl_core':rows(db,'SELECT count(*) labels,count(DISTINCT candidate_system_id) systems,count(DISTINCT exact_graph) ligands,count(DISTINCT protein_uniprot) targets FROM c')[0],
        'chembl_sair_library_flags':rows(db,'SELECT sair_library_flags_pass,count(*) labels,count(DISTINCT candidate_system_id) systems FROM c GROUP BY 1'),
        'chembl_assay_groups':rows(db,'SELECT count(*) group_count,sum((ligands>=5 AND p_max>p_min)::INT) variable_groups_ge5,sum((ligands>=10 AND p_max>p_min)::INT) variable_groups_ge10,sum(CASE WHEN ligands>=5 AND p_max>p_min THEN label_rows ELSE 0 END) rows_in_variable_groups_ge5 FROM assay_groups')[0],
        'kinase_primary':rows(db,'SELECT count(*) drugs,sum((hits>0 AND nonhits>0)::INT) drugs_with_defined_auroc,sum(measured_pairs) measured_pairs,sum(hits) hits,sum(nonhits) nonhits FROM drug_groups')[0],
        'kinase_undefined_auroc':rows(db,'SELECT * FROM drug_groups WHERE hits=0 OR nonhits=0 ORDER BY drug_name'),
        'kinase_drugs':rows(db,'SELECT * FROM drug_groups ORDER BY drug_name'),
        'limits':['No model scores are used here. No threshold or checkpoint has been fitted.', 'Hit means residual activity<50% at1uM, not binding probability.', 'ChEMBL assay groups keep endpoint separate; repeated source rows remain for adjudication.', 'All structures, final tokenization/crops and final validation/test allocation remain downstream.'],
    }
    (a.out/'validation-readiness.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('VALIDATION_COVERAGE',json.dumps({k:summary[k] for k in ['chembl_core','chembl_assay_groups','kinase_primary']}),flush=True)


if __name__=='__main__':
    main()
