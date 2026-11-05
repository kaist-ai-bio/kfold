import pytest

from kfold.affinity.cache_locations import CacheLocations, RESERVE_BYTES


def catalog(tmp_path):
    return CacheLocations(tmp_path/'locations.sqlite',
        {'/appsdata':'/appsdata/test/cache','/scratch':'/scratch/test/cache'},purpose='integration_pilot')


def quota(app=RESERVE_BYTES+100, scratch=RESERVE_BYTES+1000):
    return lambda: {'/appsdata':dict(free_bytes=app,free_files=5000),
                    '/scratch':dict(free_bytes=scratch,free_files=5000)}


def reserve(c, source='a', plan='b', **kwargs):
    args=dict(byte_bound=80,file_bound=10,quota_provider=quota(),
        future_writes={'/appsdata':0,'/scratch':0})
    args.update(kwargs)
    return c.reserve(source*64,plan*64,**args)


def test_appsdata_first_then_account_for_concurrent_reservation(tmp_path):
    c=catalog(tmp_path)
    first=reserve(c)
    assert first['mount']=='/appsdata'
    second=reserve(c,source='c')
    assert second['mount']=='/scratch'
    assert reserve(c)==first


def test_concurrent_initialization_and_reservation_preserves_capacity(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    gate = Barrier(8)
    def worker(index):
        gate.wait()
        c = catalog(tmp_path)
        return reserve(c, source=format(index, 'x'))
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(worker, range(8)))
    assert len({row['source_sha'] for row in rows}) == 8
    assert sum(row['mount'] == '/appsdata' for row in rows) == 1
    assert sum(row['mount'] == '/scratch' for row in rows) == 7


def test_catalog_failure_rolls_back_releases_lock_and_closes_connection(tmp_path):
    import sqlite3
    c = catalog(tmp_path)
    with pytest.raises(RuntimeError, match='injected'):
        with c._db() as db:
            db.execute("INSERT INTO metadata VALUES ('injected','value')")
            raise RuntimeError('injected')
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        db.execute('SELECT 1')
    with c._db() as fresh:
        assert fresh.execute("SELECT value FROM metadata WHERE key='injected'").fetchone() is None
    assert reserve(c)['state'] == 'reserved'


def test_future_raw_write_forecast_prevents_unsafe_overflow(tmp_path):
    c=catalog(tmp_path)
    with pytest.raises(RuntimeError,match='Projected quota'):
        reserve(c,quota_provider=quota(app=RESERVE_BYTES+20),
            future_writes={'/appsdata':0,'/scratch':950})
    with c._db() as db:
        assert db.execute('SELECT count(*) FROM locations').fetchone()[0]==0


def test_resume_never_moves_partial_cache_to_another_filesystem(tmp_path):
    c=catalog(tmp_path);first=reserve(c)
    with pytest.raises(RuntimeError):reserve(c,quota_provider=quota(app=RESERVE_BYTES+30))
    with c._db() as db:
        assert db.execute('SELECT destination FROM locations').fetchone()[0]==first['destination']
    with pytest.raises(ValueError,match='changed'):reserve(c,plan='c')


@pytest.mark.parametrize('bad', [{}, {'/appsdata':{'free_bytes':None,'free_files':5000}}])
def test_unknown_quota_fails_closed(tmp_path,bad):
    with pytest.raises(ValueError,match='quota unavailable'):
        reserve(catalog(tmp_path),quota_provider=lambda:bad)


def test_missing_completion_or_forecast_cannot_be_promoted(tmp_path):
    c=catalog(tmp_path)
    with pytest.raises(ValueError,match='forecast'):reserve(c,future_writes={})
    with pytest.raises(ValueError,match='reservation'):c.finish('a'*64,'b'*64)
    with pytest.raises(ValueError,match='SHA256'):c.finish('../x','b'*64)


def test_cannot_ignore_other_filesystem_projected_shortage(tmp_path):
    c=catalog(tmp_path)
    with pytest.raises(RuntimeError,match='Projected quota'):
        reserve(c,future_writes={'/appsdata':0,'/scratch':2000})


def test_completion_links_real_verified_converter_and_detects_tampering(tmp_path):
    from kfold.affinity.conversion import convert_sealed_shard
    from kfold.affinity.preparation import digest
    from test_conversion import prepared
    from pathlib import Path
    c=catalog(tmp_path)
    # Simulate both mounted filesystems inside pytest's isolated directory.
    c.roots={mount:str(tmp_path/mount.strip('/')) for mount in c.roots}
    source,plan=prepared(tmp_path)
    place=c.reserve(plan['source_sha256'],digest(plan),byte_bound=1_000_000,file_bound=20,
        quota_provider=quota(app=RESERVE_BYTES+2_000_000),future_writes={'/appsdata':0,'/scratch':0})
    convert_sealed_shard(source,place['destination'],plan,check_quota=lambda:None,head_validator=lambda s,c:None)
    completed=c.finish(plan['source_sha256'],digest(plan))
    assert completed['state']=='complete'
    c.purpose='training_cache'
    with pytest.raises(ValueError,match='remain separate'):
        c.finish(plan['source_sha256'],digest(plan))
    c.purpose='integration_pilot'
    manifest=Path(completed['manifest_path'])
    manifest.write_text(manifest.read_text()+' ')
    with pytest.raises(ValueError,match='manifest changed'):
        c.reserve(plan['source_sha256'],digest(plan),byte_bound=1_000_000,file_bound=20,
            quota_provider=quota(),future_writes={'/appsdata':0,'/scratch':0})
