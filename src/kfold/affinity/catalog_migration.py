"""Quiescent catalog relocation; payload paths and completed files stay intact."""
import hashlib
import json
import os
import sqlite3
from pathlib import Path

from .bundles import file_sha


def logical_inventory(db):
    result = {}
    for name, schema in db.execute("SELECT name,sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall():
        quoted = '"' + name.replace('"', '""') + '"'
        columns = db.execute('PRAGMA table_info('+quoted+')').fetchall()
        order = ','.join(str(i+1) for i in range(len(columns)))
        h=hashlib.sha256(); count=0
        for row in db.execute('SELECT * FROM '+quoted+' ORDER BY '+order):
            h.update((json.dumps(row,separators=(',',':'),ensure_ascii=True)+'\n').encode());count+=1
        result[name]=dict(rows=count,sha256=h.hexdigest(),schema=schema)
    return result


def relocate_catalog(source, destination, *, expected_records):
    """Caller MUST hold the shared campaign lock and verify all writers stopped."""
    source=Path(source).resolve(); destination=Path(destination).resolve()
    temporary=destination.with_suffix('.migrating')
    if source==destination or destination.exists() or temporary.exists():
        raise ValueError('Fresh catalog destination required; uncertain relocation cannot repeat')
    before=file_sha(source)
    old=sqlite3.connect(source.as_uri()+'?mode=ro',uri=True)
    new=None
    try:
        old.execute('PRAGMA temp_store=MEMORY')
        if old.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('Old catalog integrity failed')
        if old.execute("SELECT count(*) FROM bundles WHERE state!='complete'").fetchone()[0]:
            raise ValueError('Pending bundle requires reconciliation before relocation')
        if old.execute('SELECT count(*) FROM records').fetchone()[0]!=expected_records:
            raise ValueError('Catalog baseline changed')
        expected=logical_inventory(old)
        destination.parent.mkdir(parents=True,exist_ok=True)
        with temporary.open('xb'):pass
        new=sqlite3.connect(temporary);old.backup(new);new.execute('PRAGMA temp_store=MEMORY')
        if new.execute('PRAGMA quick_check').fetchone()[0]!='ok' or logical_inventory(new)!=expected:
            raise ValueError('Catalog relocation changed logical content')
        new.close();new=None
        if file_sha(source)!=before:raise ValueError('Old catalog changed during relocation')
        with temporary.open('rb') as f:os.fsync(f.fileno())
        temporary.replace(destination)
        fd=os.open(destination.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
        return dict(old_catalog=str(source),old_sha256=before,new_catalog=str(destination),
            new_sha256=file_sha(destination),logical_inventory=expected,records=expected_records,
            old_catalog_preserved=True,payload_moved=False,payload_copied=False,logical_content_equal=True)
    finally:
        if new is not None:new.close()
        old.close()
