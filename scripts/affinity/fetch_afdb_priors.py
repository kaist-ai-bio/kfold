#!/usr/bin/env python3
"""Fetch AFDB priors for exact input sequences; never silently crop or mutate.

API presence/sequence matching is separated from coordinate parsing/QC. Source
metadata, exact versioned URLs, hashes and unresolved targets remain auditable.
"""
import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def request(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=40) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            time.sleep(min(60, max(2**attempt, int(exc.headers.get('Retry-After', '5')))))
        except (TimeoutError, urllib.error.URLError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(data, separators=(',', ':')) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('targets', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--metadata-only', action='store_true', help='Write selection plan without fetching coordinates')
    a = p.parse_args()
    started = time.monotonic()
    a.out.mkdir(parents=True, exist_ok=True)
    for name in ['metadata', 'structures']:
        (a.out / name).mkdir(exist_ok=True)
    targets = json.loads(a.targets.read_text())
    accessions = sorted({x for row in targets for x in row['accessions']
                         if isinstance(x, str) and re.fullmatch(r'[A-Z0-9]+(?:-\d+)?', x)})
    if a.limit:
        # Stable dispersed sample, independent of labels and target family.
        accessions = sorted(accessions, key=lambda x: hashlib.sha256(x.encode()).hexdigest())[:a.limit]
    selected = set(accessions)
    metadata = {}

    def fetch(accession):
        path = a.out / 'metadata' / (accession + '.json')
        if path.exists():
            result = json.loads(path.read_text())
        else:
            url = 'https://alphafold.ebi.ac.uk/api/prediction/' + accession
            try:
                raw = request(url)
                result = {'accession': accession, 'url': url,
                          'fetched_utc': datetime.now(timezone.utc).isoformat(),
                          'status': 'not_found' if raw is None else 'ok',
                          'models': [] if raw is None else json.loads(raw)}
                assert isinstance(result['models'], list)
                write_json(path, result)
            except Exception as exc:
                result = {'status': 'fetch_error', 'accession': accession,
                          'error': type(exc).__name__ + ': ' + str(exc), 'models': []}
        return accession, result

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for accession, result in pool.map(fetch, accessions):
            metadata[accession] = result
            if len(metadata) % 100 == 0:
                print('METADATA', len(metadata), len(accessions), flush=True)

    rows, downloads = [], {}
    for target in targets:
        ids = set(target['accessions']) & selected
        if not ids:
            continue
        matching = {}
        for accession in ids:
            for model in metadata[accession]['models']:
                sequence = model.get('sequence') or model.get('uniprotSequence')
                if (sequence == target['sequence'] and not model.get('isComplex', False)
                        and str(model.get('toolUsed', '')).startswith('AlphaFold Monomer')):
                    url = model.get('cifUrl', '')
                    if url.startswith('https://alphafold.ebi.ac.uk/files/'):
                        matching[url] = model
        row = dict(target, status='no_exact_afdb_sequence_match', metadata_accessions=sorted(ids))
        if matching:
            # Deterministic label-independent choice; no ensemble expansion.
            url = sorted(matching, key=lambda u: (-int(matching[u].get('latestVersion', 0)), u))[0]
            model = matching[url]
            row.update(status='selected_pending_coordinate_qc', url=url,
                       entry_id=model.get('entryId'), version=model.get('latestVersion'),
                       tool_used=model.get('toolUsed'), exact_match_candidates=len(matching))
            downloads[url] = model
        elif any(metadata[x]['status'] == 'fetch_error' for x in ids):
            row['status'] = 'metadata_fetch_error'
        rows.append(row)

    tag = 'all' if not a.limit else 'sample' + str(a.limit)
    write_json(a.out / ('target-selection-' + tag + '.json'), rows)
    if a.metadata_only:
        print(json.dumps({'target_status': dict(Counter(x['status'] for x in rows)),
                          'coordinate_urls': len(downloads), 'metadata_only': True}), flush=True)
        return

    def download(url):
        name = url.rsplit('/', 1)[1]
        assert re.fullmatch(r'[A-Za-z0-9_.-]+\.cif', name)
        path = a.out / 'structures' / name
        try:
            if path.exists():
                raw = path.read_bytes()
            else:
                raw = request(url)
                if raw is None:
                    raise FileNotFoundError(url)
                assert raw.lstrip().startswith(b'data_'), 'Expected mmCIF text'
                tmp = path.with_suffix('.cif.partial');tmp.write_bytes(raw);tmp.replace(path)
            return url, {'path': str(path), 'bytes': len(raw),
                         'sha256': hashlib.sha256(raw).hexdigest(), 'download_status': 'ok'}
        except Exception as exc:
            return url, {'download_status': 'error', 'download_error': str(exc)}

    downloaded = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for url, result in pool.map(download, downloads):
            downloaded[url] = result
            if len(downloaded) % 100 == 0:
                print('COORDINATES', len(downloaded), len(downloads), flush=True)
    for row in rows:
        if 'url' in row:
            row.update(downloaded[row['url']])
            if row['download_status'] != 'ok':
                row['status'] = 'coordinate_fetch_error'
    write_json(a.out / ('target-priors-' + tag + '.json'), rows)
    summary = {'observed_utc': datetime.now(timezone.utc).isoformat(),
               'source_targets_sha256': hashlib.sha256(a.targets.read_bytes()).hexdigest(),
               'scope': tag, 'queried_accessions': len(accessions), 'target_sequences_considered': len(rows),
               'metadata_status': dict(Counter(x['status'] for x in metadata.values())),
               'target_status': dict(Counter(x['status'] for x in rows)),
               'distinct_coordinate_files': sum(x['download_status'] == 'ok' for x in downloaded.values()),
               'coordinate_bytes': sum(x.get('bytes', 0) for x in downloaded.values()),
               'elapsed_seconds': time.monotonic() - started, 'workers': a.workers,
               'coordinate_qc_complete': False, 'atlasfold_or_feature_gpu_started': False}
    write_json(a.out / ('summary-' + tag + '.json'), summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
