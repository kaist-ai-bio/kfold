"""Allocated-CPU preflight only: verify 56 pilot reuses and write a disabled draft.

This never submits, extracts new features, deletes sources, or enables execution.
The coordinator must resolve storage and current scheduler gates before release.
"""
import argparse
import json
import sqlite3
from pathlib import Path

from feature_campaign import index_interval, read_pilot_record
from run_feature_pilot import sha, save
from run_feature_worker import tasks_for_chunk
from kfold.affinity.preparation import digest

CHILD_SHA = '47fc40a3b66d10cd91d8f3778e6c4e66d01566bfe24e1149f1b7c8715ff54401'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-campaign', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--pilot-results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    if sha(a.registry) != CHILD_SHA:
        raise ValueError('Child registry checksum mismatch')
    c = json.loads(a.base_campaign.read_text())
    c['registry'] = str(a.registry.resolve())
    c['registry_bytes'] = a.registry.stat().st_size
    c.update(systems=2796376,first_system_index=1459112,last_system_index=2796376,
             execution_authorized=False,projected_total_bytes=55_546_530_000_000,
             source_scope='Synthetic assumed-negative appended interval only; original registry/campaign unchanged')
    c['feature_contract']['registry_sha256'] = CHILD_SHA
    pilot_contract = json.loads((a.pilot_results/'contract.json').read_text())
    if pilot_contract['feature'] != c['feature_contract']:
        raise ValueError('Pilot feature contract differs')
    c['storage_contract_id'] = digest(pilot_contract)
    final = json.loads((a.pilot_results/'final.json').read_text())
    # Final schema is retained as provenance; exact count/readback below is decisive.
    c['pilot_final_sha256'] = sha(a.pilot_results/'final.json')
    first,last = index_interval(c)
    db = sqlite3.connect(a.registry.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    try:
        if db.execute('SELECT max(system_index) FROM systems').fetchone()[0] != last:
            raise ValueError('Unexpected registry append state')
        if db.execute('SELECT count(*) FROM systems WHERE system_index BETWEEN ? AND ?', (first,last)).fetchone()[0] != last-first+1:
            raise ValueError('Missing decoy registry indices')
        if db.execute("SELECT value FROM metadata WHERE key='recipe_sha256'").fetchone()[0] != c['recipe_sha256']:
            raise ValueError('Input recipe changed')
        reuse, shards = {}, []
        for receipt_path in sorted((a.pilot_results/'shards').glob('shard-*.receipt.json')):
            receipt = json.loads(receipt_path.read_text())
            path = receipt_path.parent/receipt['shard']
            if path.suffix != '.sqlite' or sha(path) != receipt['sha256']:
                raise ValueError('Pilot sealed SHA mismatch')
            shard = sqlite3.connect(path.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
            try:
                if shard.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                    raise ValueError('Pilot SQLite integrity error')
                if shard.execute("SELECT value FROM metadata WHERE key='contract_id'").fetchone()[0] != c['storage_contract_id']:
                    raise ValueError('Pilot storage contract differs')
                records = shard.execute('SELECT system_id,identity FROM records').fetchall()
            finally:
                shard.close()
            if len(records) != receipt['records']:
                raise ValueError('Pilot receipt record count differs')
            for key,identity in records:
                row = db.execute('SELECT system_index FROM systems WHERE input_id=?',(key,)).fetchone()
                if row is None or not first <= row[0] <= last or key in reuse:
                    raise ValueError('Duplicate/non-decoy pilot input')
                task = tasks_for_chunk(db,row[0],1)[0]
                entry=dict(system_index=row[0],shard_path=str(path.resolve()),
                           shard_sha256=receipt['sha256'],cache_identity_sha256=identity)
                payload = read_pilot_record(entry,task,c['feature_contract'])
                del payload
                reuse[key] = entry
            shards.append(receipt)
    finally:
        db.close()
    if len(reuse) != 56:
        raise ValueError('Expected exactly 56 original passed pilot inputs')
    a.out.mkdir(exist_ok=False)
    save(a.out/'reuse-manifest.json',reuse)
    c['reuse_manifest'] = str((a.out/'reuse-manifest.json').resolve())
    c['reuse_manifest_sha256'] = sha(a.out/'reuse-manifest.json')
    save(a.out/'campaign.draft.json',c)
    report=dict(execution_authorized=False,preflight_passed=True,reused_pilot_inputs=56,
                decoy_inputs=last-first+1,new_gpu_inputs=last-first+1-56,
                original_raw_unchanged=True,shards=shards,registry_sha256=CHILD_SHA,
                draft_sha256=sha(a.out/'campaign.draft.json'),
                remaining_gate='User storage staged-plan decision and live96GPU/5jobs budget',
                feature_source_revision=c['feature_contract']['code_revision'],
                pilot_final=final)
    save(a.out/'preparation-receipt.json',report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
