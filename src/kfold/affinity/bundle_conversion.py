"""Verified source conversion into a multi-source bundle; no quota RPCs here."""
import json
from pathlib import Path

from .bundles import file_sha, readonly, digest
from .compact import pack_crop_union, verify_against_source
from .shards import decode, encode, validate_codec


def append_source(store, source_path, plan, *, head_validator):
    source = Path(source_path).resolve()
    if source.suffix != '.sqlite' or not source.is_file():
        raise ValueError('Sealed source required')
    if source == store.path or source == store.partial:
        raise ValueError('Cannot overwrite source')
    receipt = json.loads(source.with_suffix('.receipt.json').read_text())
    if (plan['source_sha256'] != receipt['sha256'] or
        plan['source_contract_id'] != receipt['contract_id'] or
        source.stat().st_size != receipt['bytes'] or file_sha(source) != receipt['sha256']):
        raise ValueError('Source SHA/receipt mismatch')
    codec = validate_codec(plan['storage_codec'])
    if codec != store.contract['storage_codec'] or plan['purpose'] != store.contract['purpose']:
        raise ValueError('Storage/science contract mismatch')
    if plan['purpose'] not in ('training_cache','integration_pilot') or not plan.get('selector_contract'):
        raise ValueError('Explicit selector and purpose required')
    if any(row.get('status') != 'final' for row in plan['records'].values()):
        raise ValueError('Pending plan cannot be converted')
    if plan['purpose'] == 'training_cache' and any(row['crop_receipt'].get('training_ready') is not True for row in plan['records'].values()):
        raise ValueError('Training-ready science receipts required')
    db = readonly(source)
    try:
        if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('Source quick_check failed')
        if db.execute("SELECT value FROM metadata WHERE key='contract_id'").fetchone()[0] != receipt['contract_id']:
            raise ValueError('Source contract mismatch')
        ids = {r[0] for r in db.execute('SELECT system_id FROM records')}
        subset = plan.get('selection')
        if subset is not None:
            from .partial_bundles import validate_selection, SCOPE
            if store.contract.get('coverage_scope') != SCOPE:
                raise ValueError('Subset plan requires explicit subset store')
            validate_selection(subset, ids, plan['records'])
        elif store.contract.get('coverage_scope') == 'record_subset_v1':
            raise ValueError('Subset store requires frozen selection')
        if not ids or (subset is None and ids != set(plan['records'])) or len(ids) != receipt['records']:
            raise ValueError('Whole source exact coverage required')
        complete = store.pin_source(plan, dict(path=str(source), receipt=receipt))
        if complete:
            return dict(source_sha=receipt['sha256'],records=len(plan['records']),reused=True)
        head_views = 0
        with store.source_transaction(receipt['sha256']):
            for key,blob,bsha,rsha in db.execute('SELECT system_id,payload,blob_sha,raw_sha FROM records ORDER BY system_id'):
                if subset is not None and key not in plan['records']:
                    continue
                original = decode(blob,bsha,rsha)
                row = plan['records'][key]
                if (original['receipt']['system_id'] != key or
                    digest(original['receipt']) != original['cache_identity_sha256'] or
                    original['cache_identity_sha256'] != row['source_cache_identity_sha256']):
                    raise ValueError('Source identity/seed/receipt changed')
                verifier = verify_against_source
                if row.get('schema') == 'pli_historical_crop_union_bf16_v1':
                    from .alias_compact import pack_alias_union, verify_alias_against_source, POLICY
                    if store.contract.get('historical_alias_policy') != POLICY:
                        raise ValueError('Explicit historical alias store contract required')
                    compact = pack_alias_union(original, row['historical_plans'])
                    verifier = verify_alias_against_source
                else:
                    compact = pack_crop_union(original,row['views'],crop_receipt=row['crop_receipt'],unavailable=row.get('unavailable',{}))
                verifier(original,compact)
                encoded = encode(compact,codec)
                store.add_encoded(key,compact['cache_identity_sha256'],*encoded,receipt['sha256'])
                restored = decode(*store.encoded(key))
                verifier(original,restored)
                head_validator(original,restored)
                head_views += len(restored['view_indices'])
            if file_sha(source) != receipt['sha256']:
                raise ValueError('Source modified during conversion')
            store.finish_source(plan,dict(source_sha_verified=True,retained_tensors_verified=True,
                readback_verified=True,head_loader_verified=True,head_views=head_views,source_deleted=False,
                selected_coverage_verified=subset is not None,
                whole_source_coverage=subset is None or set(plan['records']) == ids))
        return dict(source_sha=receipt['sha256'],records=len(plan['records']),head_views=head_views,reused=False)
    finally:
        db.close()
