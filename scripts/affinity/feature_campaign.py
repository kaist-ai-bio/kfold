"""Operational range and prior-pilot reuse checks; no feature math changes."""
import json
import sqlite3
import subprocess
import time
from pathlib import Path

from kfold.affinity.preparation import digest
from kfold.affinity.shards import decode


def quota_headroom(output, filesystem):
    """Parse numeric `lfs quota -g GROUP -q MOUNT`; never use df capacity."""
    rows = [line.split() for line in output.splitlines() if line.split() and line.split()[0] == filesystem]
    if len(rows) != 1 or len(rows[0]) != 9:
        raise ValueError('Missing or ambiguous group quota row')
    fields = rows[0]
    used, soft, hard, files, file_soft, file_hard = [int(fields[i].rstrip('*')) for i in (1,2,3,5,6,7)]
    byte_limits = [x for x in (soft,hard) if x > 0]
    file_limits = [x for x in (file_soft,file_hard) if x > 0]
    if not byte_limits or not file_limits:
        raise ValueError('Enforced group quota must be available')
    return dict(free_bytes=(min(byte_limits)-used)*1024,
                free_files=min(file_limits)-files, used_kib=used, limit_kib=min(byte_limits))


class StorageReserveGuard:
    """Bounded live group checks, cached per worker for five minutes.

The extra in-flight margin supplements the user's3TB reserve. Other group users
can consume space between probes; coordinator forecasts remain required.
"""
    def __init__(self, policy):
        if policy.get('mode') != 'group_filesystem_reserve':
            raise ValueError('Unsupported storage policy')
        if policy.get('reserve_bytes_per_filesystem') != 3_000_000_000_000:
            raise ValueError('Expected explicit3TB group reserve')
        self.policy = policy
        self.checked = None
        self.last = None

    def check(self):
        now = time.monotonic()
        if self.checked is not None and now-self.checked < 300:
            return self.last
        rows = {}
        try:
            for fs in ('/scratch','/appsdata'):
                result = subprocess.run(['lfs','quota','-g',self.policy['group'],'-q',fs],
                    capture_output=True,text=True,check=True,timeout=20)
                rows[fs] = quota_headroom(result.stdout,fs)
            needed = self.policy['reserve_bytes_per_filesystem'] + self.policy['inflight_margin_bytes']
            allowed = all(x['free_bytes'] >= needed and x['free_files'] >= 1000 for x in rows.values())
            self.last = dict(allowed=allowed,observed_unix=time.time(),filesystems=rows,
                             reserve_bytes=self.policy['reserve_bytes_per_filesystem'],
                             inflight_margin_bytes=self.policy['inflight_margin_bytes'])
        except (OSError,ValueError,subprocess.SubprocessError) as exc:
            self.last = dict(allowed=False,observed_unix=time.time(),error=repr(exc),filesystems=rows)
        self.checked = now
        return self.last


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
