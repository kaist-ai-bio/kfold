"""Operational range and prior-pilot reuse checks; no feature math changes."""
import json
import sqlite3
from pathlib import Path

from kfold.affinity.preparation import digest
from kfold.affinity.shards import decode


def index_interval(campaign):
    maximum = campaign['systems']  # Frozen registry max index, not cohort size.
    first = campaign.get('first_system_index', 1)
    last = campaign.get('last_system_index', maximum)
    if any(type(x) is not int for x in (first, last, maximum)) or not 1 <= first <= last <= maximum:
        raise ValueError('Invalid inclusive campaign index interval')
    return first, last


def chunk_interval(campaign, number):
    first, last = index_interval(campaign)
    size = campaign['chunk_size']
    if type(size) is not int or size <= 0 or type(number) is not int or number < 0:
        raise ValueError('Invalid chunk size/number')
    start = first + number * size
    if start > last:
        raise ValueError('Chunk outside selected campaign interval')
    return start, min(size, last - start + 1)


def validate_cached_record(payload, task, contract):
    source = payload['source_input']
    # Pilot length-bucket annotation is the sole permitted extra source field.
    if set(source) - set(task) - {'bucket'} or any(source.get(k) != v for k,v in task.items()):
        raise ValueError('Cached source input differs from current registry task')
    receipt = payload['receipt']
    expected = dict(system_id=task['input_id'], system_index=task['system_index'],
                    seed=task['system_index'], contract=contract,
                    actual_tokens=task['total_tokens'], lm_dropout=.25,
                    diffusion_executed=False, pp_stored=False)
    if any(receipt.get(k) != v for k,v in expected.items()):
        raise ValueError('Cached feature contract/seed/identity mismatch')
    if payload['cache_identity_sha256'] != digest(receipt):
        raise ValueError('Cached receipt identity hash mismatch')


def read_pilot_record(entry, task, contract):
    """Read a coordinator-pinned sealed shard without modifying the pilot store."""
    path = Path(entry['shard_path'])
    if path.suffix != '.sqlite':
        raise ValueError('Only sealed pilot shards can be reused')
    receipt = json.loads(path.with_suffix('.receipt.json').read_text())
    if receipt['sha256'] != entry['shard_sha256'] or path.stat().st_size != receipt['bytes']:
        raise ValueError('Pilot shard receipt changed')
    # Full-file SHA must also be checked by coordinator before publication/submission.
    db = sqlite3.connect(path.resolve().as_uri()+'?mode=ro&immutable=1', uri=True)
    try:
        row = db.execute('SELECT payload,blob_sha,raw_sha FROM records WHERE system_id=?',
                         (task['input_id'],)).fetchone()
    finally:
        db.close()
    if row is None:
        raise ValueError('Pilot input missing from pinned shard')
    payload = decode(*row)
    validate_cached_record(payload, task, contract)
    if payload['cache_identity_sha256'] != entry['cache_identity_sha256']:
        raise ValueError('Pilot reuse manifest identity mismatch')
    return payload
