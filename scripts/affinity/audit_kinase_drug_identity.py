#!/usr/bin/env python3
"""Cross-check stored screen drug identities against official ChEMBL records.

Independent database support is recorded; it does not certify an assay stock or
replace any source molecular structure. This is a bounded network metadata task.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
from pathlib import Path
import re
import urllib.parse
from fetch_chembl_validation import BASE, request


def normalized(name):
    return re.sub(r'[^A-Z0-9]', '', name.upper())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('drugs_tsv', type=Path)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    inventory = json.loads(args.inventory.read_text())
    ids = {r['drug_record_id'] for r in inventory['wt_drug_records'] if r['n'] > 0}
    with args.drugs_tsv.open() as stream:
        drugs = [r for r in csv.DictReader(stream, delimiter='\t') if r['drug_record_id'] in ids]
    assert len(drugs) == len(ids) == 92
    before = request(BASE+'status.json')
    def fetch(drug):
        key = drug['inchikey']
        assert re.fullmatch(r'[A-Z]{14}-[A-Z]{10}-[A-Z]',key)
        url = BASE+'molecule.json?'+urllib.parse.urlencode({'molecule_structures__standard_inchi_key': key, 'limit': 1000})
        path = args.out/(drug['drug_record_id']+'.json')
        if path.exists():
            result = json.loads(path.read_text())
        else:
            result = request(url)
            path.write_text(json.dumps(result,indent=2)+'\n')
        records = result['molecules']
        assert len(records) == result['page_meta']['total_count']
        assert all((r.get('molecule_structures') or {}).get('standard_inchi_key') == key for r in records)
        matches = []
        for record in records:
            names = [record.get('pref_name') or '']
            for synonym in record.get('molecule_synonyms',[]):
                names.extend(v for v in synonym.values() if isinstance(v,str))
            if any(normalized(n)==normalized(drug['source_name']) for n in names):
                matches.append(record['molecule_chembl_id'])
        row = {k:drug[k] for k in ['drug_record_id','source_name','inchikey','pubchem_cid','structure_source','identity_resolution_level']}
        row.update(chembl_exact_inchikey_records=[r['molecule_chembl_id'] for r in records],exact_source_name_matches=matches,
                   independent_graph_and_name_support=bool(matches),url=url,raw_file=str(path),raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        print('DRUG_IDENTITY',drug['source_name'],len(records),len(matches),flush=True)
        return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(fetch,drugs))
    after = request(BASE+'status.json')
    assert after == before
    summary = {'api_status':after,'drugs':rows,'count':len(rows),'supported':sum(r['independent_graph_and_name_support'] for r in rows),
               'unconfirmed':[r['source_name'] for r in rows if not r['independent_graph_and_name_support']],
               'scope':'Exact stored InChIKey and normalized exact source drug name supported in an independent ChEMBL record. This does not establish experimental stock identity, assay construct, or equality of salt/protonation preparations. Source input chemistry unchanged.'}
    (args.out/'drug-identity-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('DRUG_IDENTITY_COMPLETE',summary['supported'],summary['count'],flush=True)


if __name__ == '__main__':
    main()
