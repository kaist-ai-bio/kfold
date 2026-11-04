"""One GPU worker over immutable registry chunks; safe across Slurm restarts.

The coordinator must publish an approved campaign.json after the GPU pilot.
Each chunk is protected by an OS lock, has transactional shards and an explicit
terminal record for every registry input. An unexpected error stops dispatch.
"""
import argparse
import fcntl
import gc
import json
import os
import signal
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

import torch
from kfold.affinity.features import FeatureContract, extract_indexed_features, indexed_rng, hash_prepared_input
from kfold.affinity.preparation import digest
from kfold.affinity.shards import ShardStore
from kfold.data.types.ccd import CCD
from kfold.inference.query import Query, ProteinSequence, LigandSequence
from kfold.inference.runner import KFoldRunner
from kfold.model import KFold
from run_feature_pilot import sha, save
from feature_campaign import index_interval, chunk_interval, validate_cached_record, read_pilot_record

STOP = False


def signal_stop(signum, frame):
    global STOP
    STOP = True


def tasks_for_chunk(db, start, size):
    keys = ('system_index', 'input_id', 'sequence', 'smiles', 'apo_path',
            'apo_sha256', 'total_tokens', 'protein_tokens', 'ligand_tokens', 'source_metadata')
    rows = db.execute('''SELECT s.system_index,s.input_id,p.sequence,l.smiles,p.path,
        p.apo_sha,s.total_tokens,s.protein_tokens,s.ligand_tokens,p.metadata
        FROM systems s JOIN proteins p ON s.protein_sha=p.source_sha
        JOIN ligands l USING(ligand_sha) WHERE system_index>=? AND system_index<?
        ORDER BY system_index''', (start, start + size))
    result = []
    for row in rows:
        task = dict(zip(keys, row))
        task['source_metadata'] = json.loads(task['source_metadata'])
        result.append(task)
    return result


def extract_one(task, runner, model, contract, checked_apos):
    if task['apo_path'] not in checked_apos:
        assert sha(task['apo_path']) == task['apo_sha256']
        checked_apos[task['apo_path']] = task['apo_sha256']
    assert checked_apos[task['apo_path']] == task['apo_sha256']
    start = time.perf_counter()
    query = Query(name='pli_' + str(task['system_index']), sequences=[
        ProteinSequence(id=['A'], sequence=task['sequence'], apo=[Path(task['apo_path'])]),
        LigandSequence(id=['B'], smiles=task['smiles'])])
    torch.cuda.reset_peak_memory_stats()
    with indexed_rng(task['system_index']):
        apos, priors = runner.load_apo_and_prior(query)
        item = runner.build_input(query, task['system_index'], num_samples=1, apos=apos, priors=priors)
    input_sha = hash_prepared_input(item.f_input)
    f = item.f_input.add_batch_dim().to('cuda:0')
    torch.cuda.synchronize()
    prep = time.perf_counter() - start
    start = time.perf_counter()
    result = extract_indexed_features(model, f, system_id=task['input_id'],
        system_index=task['system_index'], input_sha256=input_sha, contract=contract)
    torch.cuda.synchronize()
    assert result['receipt']['actual_tokens'] == task['total_tokens']
    metrics = dict(preparation_seconds=prep, trunk_and_pack_seconds=time.perf_counter()-start,
        actual_tokens=result['receipt']['actual_tokens'], padded_tokens=result['receipt']['padded_tokens'],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    atom = item.f_input.atom
    valid = atom.pad_mask
    result['atom_mapping'] = {k: getattr(atom, k)[valid].cpu().clone() for k in (
        'atom_type','token_index','atom_index','ref_charge','ref_pos','ref_mask','ref_space_uid')}
    result['atom_mapping']['name_chars'] = atom.ref_atom_name_chars[valid].argmax(-1).to(torch.uint8)
    result['atom_mapping']['element'] = atom.ref_element[valid].argmax(-1).to(torch.uint8)
    result['source_input'] = task
    return result, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', required=True)
    p.add_argument('--max-chunks', type=int, default=0)
    p.add_argument('--max-records', type=int, default=0)
    a = p.parse_args()
    cpath = Path(a.campaign)
    c = json.loads(cpath.read_text())
    assert c['pilot_gate_passed'] is True and c['projected_total_bytes'] <= 60_000_000_000_000
    first_index, last_index = index_interval(c)
    reuse = {}
    if c.get('reuse_manifest'):
        assert sha(c['reuse_manifest']) == c['reuse_manifest_sha256']
        reuse = json.loads(Path(c['reuse_manifest']).read_text())
        assert all(first_index <= entry['system_index'] <= last_index for entry in reuse.values())
    root = cpath.parent
    rank = int(os.environ.get('SLURM_PROCID', '0'))
    world = int(os.environ.get('SLURM_NTASKS', '1'))
    job = os.environ.get('SLURM_JOB_ID', 'manual')
    def stopping():
        return STOP or (root/'STOP').exists() or (root/'stop-jobs'/job).exists()
    logdir = root/'workers'/f'{job}-{rank:04d}'
    logdir.mkdir(parents=True, exist_ok=True)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, signal_stop)
    # Coordinator hashes the 1 GB registry once per submission. Each worker
    # verifies the pinned stat/recipe and opens it read-only; no network scan.
    registry = Path(c['registry'])
    assert registry.stat().st_size == c['registry_bytes']
    db = sqlite3.connect('file:'+str(registry)+'?mode=ro&immutable=1', uri=True)
    assert db.execute("SELECT value FROM metadata WHERE key='recipe_sha256'").fetchone()[0] == c['recipe_sha256']
    assert db.execute('SELECT max(system_index) FROM systems').fetchone()[0] == c['systems']
    contract = FeatureContract(**c['feature_contract'])
    assert sha(Path(c['model'])/'weights/kfold.pth') == contract.checkpoint_sha256
    assert sha(Path(c['model'])/'config.yaml') == contract.config_sha256
    assert sha(c['ccd']) == c['ccd_sha256']
    torch.set_num_threads(4)
    model = KFold.from_pretrained(c['model'], device='cuda:0', cache_dir=c['cache'],
        use_rna_encoder=False, kernel_backend='triton')
    runner = KFoldRunner(model, ccd=CCD.load(c['ccd']), cache_dir=c['cache'], verbose=False)
    assert float(model.lm_dropout) == 0.25
    # Native libraries may install signal handlers during model initialization.
    # A job-scoped file also permits a reliable input-boundary stop without
    # depending on Slurm's signal propagation to each Python process.
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, signal_stop)
    save(logdir/'hardware.json', dict(gpu=torch.cuda.get_device_name(),
        vram_bytes=torch.cuda.get_device_properties(0).total_memory, torch=str(torch.__version__),
        campaign_sha256=sha(cpath), feature_contract=asdict(contract), rank=rank, world=world, job=job,
        driver_sha256=sha(__file__), pid=os.getpid(), hostname=os.uname().nodename,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    checked_apos = {}
    chunk_size = c['chunk_size']
    count = (last_index - first_index + 1 + chunk_size - 1) // chunk_size
    # First partition work cheaply; fallback allows resumption with any GPU count.
    order = list(range(rank, count, world)) + [i for i in range(count) if i % world != rank]
    chunks_done = 0
    total_done = 0
    start_run = time.time()
    for number in order:
        if stopping(): break
        if a.max_records and total_done >= a.max_records: break
        chunk = root/'chunks'/f'{number:06d}'
        if (chunk/'done.json').exists(): continue
        chunk.mkdir(parents=True, exist_ok=True)
        with (chunk/'writer.lock').open('a+') as lock:
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: continue
            if (chunk/'done.json').exists(): continue
            save(chunk/'owner.json', dict(job=job, rank=rank, started=time.time()))
            store = ShardStore(chunk/'shards', c['storage_contract_id'])
            failures_path = chunk/'exclusions.json'
            exclusions = json.loads(failures_path.read_text()) if failures_path.exists() else {}
            chunk_start, chunk_length = chunk_interval(c, number)
            tasks = tasks_for_chunk(db, chunk_start, chunk_length)
            assert len(tasks) == chunk_length, 'Missing registry indices in campaign interval'
            try:
                with (chunk/'metrics.jsonl').open('a') as metrics_log:
                    for task in tasks:
                        if stopping(): break
                        if a.max_records and total_done >= a.max_records: break
                        key = task['input_id']
                        if key in store.entries:
                            # Successful partial writes are never trusted by count alone.
                            old = store.get(key)
                            validate_cached_record(old, task, asdict(contract))
                            del old
                            continue
                        if key in exclusions: continue
                        if key in reuse:
                            assert reuse[key]['system_index'] == task['system_index']
                            old = read_pilot_record(reuse[key], task, asdict(contract))
                            sizes = store.add(key, old)
                            restored = store.get(key)
                            validate_cached_record(restored, task, asdict(contract))
                            assert restored['cache_identity_sha256'] == old['cache_identity_sha256']
                            metrics_log.write(json.dumps(dict(input_id=key,
                                system_index=task['system_index'],status='reused_verified_pilot',
                                reused_from=reuse[key],**sizes),allow_nan=False)+'\n')
                            metrics_log.flush()
                            del old, restored
                            continue
                        started = time.perf_counter()
                        row = dict(input_id=key, system_index=task['system_index'], total_tokens=task['total_tokens'])
                        result = None
                        for attempt in range(2):
                            try:
                                result, measured = extract_one(task, runner, model, contract, checked_apos)
                                row.update(measured)
                                break
                            except torch.cuda.OutOfMemoryError as exc:
                                row['oom_last_error'] = str(exc)
                                gc.collect(); torch.cuda.empty_cache()
                        if result is None:
                            row.update(status='cuda_oom_twice', attempts=2)
                            exclusions[key] = row
                            save(failures_path, exclusions)
                        else:
                            t = time.perf_counter()
                            row.update(store.add(key, result), write_seconds=time.perf_counter()-t)
                            restored = store.get(key)  # Checks both compressed and decoded SHA.
                            assert restored['cache_identity_sha256'] == result['cache_identity_sha256']
                            del restored, result
                            row['status'] = 'passed'
                        row['elapsed_seconds'] = time.perf_counter()-started
                        metrics_log.write(json.dumps(row, allow_nan=False)+'\n'); metrics_log.flush()
                        total_done += 1
                        save(logdir/'progress.json', dict(time=time.time(), elapsed=time.time()-start_run,
                            systems_processed=total_done, current_chunk=number, chunks_done=chunks_done, last=row))
                        gc.collect(); torch.cuda.empty_cache()
            except Exception as exc:
                error = dict(job=job, rank=rank, chunk=number, error=repr(exc), time=time.time())
                save(logdir/'error.json', error)
                # O_EXCL avoids racing writers. STOP requires reviewed explicit removal.
                try:
                    with (root/'STOP').open('x') as stop: json.dump(error, stop)
                except FileExistsError: pass
                raise
            finally:
                store.seal()
            keys = {t['input_id'] for t in tasks}
            assert set(store.entries).isdisjoint(exclusions)
            if keys == set(store.entries) | set(exclusions):
                save(chunk/'done.json', dict(chunk=number, systems=len(tasks),
                    passed=len(store.entries), excluded=len(exclusions),
                    storage_contract_id=c['storage_contract_id'], time=time.time()))
                chunks_done += 1
            if a.max_chunks and chunks_done >= a.max_chunks: break
    save(logdir/'exit.json', dict(time=time.time(), stopped=stopping(),
        chunks_done=chunks_done, systems_processed=total_done))


if __name__ == '__main__': main()
