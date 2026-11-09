import sqlite3
import pytest
from kfold.affinity.catalog_migration import relocate_catalog
from kfold.affinity.bundles import file_sha

def fixture(path):
    c=sqlite3.connect(path)
    c.executescript('CREATE TABLE records(system_id PRIMARY KEY,path,identity); CREATE TABLE bundles(bundle_id PRIMARY KEY,state);')
    c.execute('INSERT INTO records VALUES (?,?,?)',('a','/appsdata/immutable.sqlite','original-identity'))
    c.execute('INSERT INTO bundles VALUES (?,?)',('b','complete'));c.commit();c.close()

def test_relocation_preserves_old_and_exact_references(tmp_path):
    old=tmp_path/'old.sqlite';new=tmp_path/'scratch/catalog.sqlite';fixture(old);before=file_sha(old)
    result=relocate_catalog(old,new,expected_records=1)
    assert result['logical_content_equal'] and file_sha(old)==before
    c=sqlite3.connect(new);assert c.execute('SELECT * FROM records').fetchone()==('a','/appsdata/immutable.sqlite','original-identity')
    c.execute("INSERT INTO records VALUES ('next','/scratch/new.sqlite','new-id')");c.commit();c.close()
    assert file_sha(old)==before
    with pytest.raises(ValueError,match='Fresh'):relocate_catalog(old,new,expected_records=1)

@pytest.mark.parametrize('defect',['pending','count','partial'])
def test_uncertain_migrations_rejected(tmp_path,defect):
    old=tmp_path/'old.sqlite';new=tmp_path/'new.sqlite';fixture(old)
    if defect=='pending':
        c=sqlite3.connect(old);c.execute("UPDATE bundles SET state='pending'");c.commit();c.close()
    if defect=='partial':new.with_suffix('.migrating').write_text('prior evidence')
    with pytest.raises(ValueError):relocate_catalog(old,new,expected_records=2 if defect=='count' else 1)
    assert not new.exists()

def test_scratch_only_budget_never_reserves_readonly_appsdata():
    from kfold.affinity.bundles import BatchBudget
    snapshot={'/scratch':dict(free_bytes=10_000_000_000_000,free_files=100000),
              '/appsdata':dict(free_bytes=1,free_files=1)}
    budget=BatchBudget({'/scratch':snapshot['/scratch']},{'/scratch':2_000_000_000_000})
    assert budget.reserve('one',1024**3)['mount']=='/scratch'
    with pytest.raises(RuntimeError):budget.reserve('forbidden',1024**3,mount='/appsdata')
    with pytest.raises(RuntimeError):BatchBudget({'/scratch':dict(free_bytes=5_000_000_000_000,free_files=100000)},{'/scratch':2_000_000_000_000})
