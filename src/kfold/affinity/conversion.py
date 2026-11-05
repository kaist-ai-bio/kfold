"""Resumable lossless conversion of one sealed source shard.

The caller supplies a complete, frozen crop plan and a live quota check. This
module does not choose pockets, substitute pending views, delete sources, or
decide which filesystem should receive the next shard.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3

from .compact import pack_crop_union, verify_against_source
from .preparation import digest
from .shards import ShardStore, decode, LEGACY_CODEC, validate_codec


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path, value):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    temporary.replace(path)


def convert_sealed_shard(source, destination, plan, *, check_quota, head_validator):
    """Validate every record and publish a location-aware manifest last.

    A plan contains source_sha256, source_contract_id, selector_contract and an
    exact records mapping. Each record pins source_cache_identity_sha256, views,
    unavailable and crop_receipt. Only status='final' may be converted. Pending
    targets and alias decisions must be resolved by the caller before invoking
    this function; unsupported views need explicit final reasons.

    check_quota runs before new writes and per record; exceptions preserve any
    partial output for an identical-plan resume. head_validator receives source
    and restored compact payloads and must raise on a loader mismatch.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source.suffix != '.sqlite' or not source.is_file():
        raise ValueError('Only an existing sealed source shard may be converted')
    if destination == source.parent or source.is_relative_to(destination):
        raise ValueError('Destination must not contain or overwrite the source')
    receipt = json.loads(source.with_suffix('.receipt.json').read_text())
    if (receipt['sha256'] != plan['source_sha256'] or
            receipt['contract_id'] != plan['source_contract_id'] or
            receipt['bytes'] != source.stat().st_size or file_sha(source) != receipt['sha256']):
        raise ValueError('Sealed source receipt or bytes changed')
    if plan.get('purpose') not in ('integration_pilot', 'training_cache'):
        raise ValueError('Explicit conversion purpose required')
    if not plan.get('selector_contract') or not plan.get('records'):
        raise ValueError('Missing frozen selector contract or records')
    if any(row.get('status') != 'final' for row in plan['records'].values()):
        raise ValueError('Pending crop decisions cannot be published')
    if plan['purpose'] == 'training_cache' and any(
            row['crop_receipt'].get('training_ready') is not True
            for row in plan['records'].values()):
        raise ValueError('Training cache requires finalized scientific crop receipts')
    codec = validate_codec(plan.get('storage_codec', LEGACY_CODEC))
    contract = digest(plan)
    db = sqlite3.connect(source.as_uri() + '?mode=ro&immutable=1', uri=True)
    try:
        if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('Source database integrity failed')
        if db.execute("SELECT value FROM metadata WHERE key='contract_id'").fetchone()[0] != receipt['contract_id']:
            raise ValueError('Source database contract mismatch')
        identities = [row[0] for row in db.execute('SELECT system_id FROM records ORDER BY system_id')]
        if set(identities) != set(plan['records']) or len(identities) != receipt['records']:
            raise ValueError('Crop plan must cover the entire source shard exactly')
        check_quota()
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / 'conversion.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            identity_path = destination / 'conversion-identity.json'
            identity = dict(plan_sha256=contract, source=str(source), source_sha256=receipt['sha256'])
            if identity_path.exists():
                if json.loads(identity_path.read_text()) != identity:
                    raise ValueError('Conversion plan or source changed on resume')
            else:
                if any(destination.glob('shards/*')) or (destination / 'complete.json').exists():
                    raise ValueError('Unidentified existing conversion output')
                _atomic_json(identity_path, identity)
            # Validate sealed destination bytes before ShardStore can resume.
            for shard in (destination / 'shards').glob('*.sqlite'):
                rpath = shard.with_suffix('.receipt.json')
                if rpath.exists():
                    r = json.loads(rpath.read_text())
                    if r['contract_id'] != contract or r['bytes'] != shard.stat().st_size or file_sha(shard) != r['sha256']:
                        raise ValueError('Existing compact shard was modified')
            store = ShardStore(destination / 'shards', contract, codec=codec)
            if set(store.entries) - set(identities):
                raise ValueError('Unexpected records in conversion output')
            views = {}
            try:
                for system_id, blob, bsha, rsha in db.execute('SELECT system_id,payload,blob_sha,raw_sha FROM records ORDER BY system_id'):
                    check_quota()
                    source_payload = decode(blob, bsha, rsha)
                    row = plan['records'][system_id]
                    if (source_payload['receipt']['system_id'] != system_id or
                            digest(source_payload['receipt']) != source_payload['cache_identity_sha256'] or
                            source_payload['cache_identity_sha256'] != row['source_cache_identity_sha256']):
                        raise ValueError('Source record identity or receipt changed')
                    compact = pack_crop_union(source_payload, row['views'],
                        crop_receipt=row['crop_receipt'], unavailable=row.get('unavailable', {}))
                    verify_against_source(source_payload, compact)
                    store.add(system_id, compact)
                    restored = store.get(system_id)
                    verify_against_source(source_payload, restored)
                    head_validator(source_payload, restored)
                    views[system_id] = dict(available=sorted(restored['view_indices']),
                        unavailable=restored['unavailable_views'], cache_identity_sha256=restored['cache_identity_sha256'])
                store.seal()
            finally:
                if store.db is not None:
                    store.db.close()
                    store.db = None
            if set(store.entries) != set(identities):
                raise ValueError('Final compact coverage mismatch')
            shards = []
            for shard in sorted((destination / 'shards').glob('*.sqlite')):
                r = json.loads(shard.with_suffix('.receipt.json').read_text())
                if file_sha(shard) != r['sha256'] or r['contract_id'] != contract:
                    raise ValueError('Final compact checksum mismatch')
                shards.append(dict(path=str(shard), **r))
            if sum(r['records'] for r in shards) != len(identities):
                raise ValueError('Sealed receipt coverage mismatch')
            result = dict(schema='pli_compact_conversion_v1', **identity,
                purpose=plan['purpose'], storage_codec=codec, training_ready=plan['purpose'] == 'training_cache',
                records=len(identities), system_ids=identities, views=views, shards=shards,
                bytes=sum(r['bytes'] for r in shards), source_deleted=False,
                retained_tensors_verified=True, readback_verified=True, head_loader_verified=True)
            check_quota()
            _atomic_json(destination / 'complete.json', result)
            return result
    finally:
        db.close()
