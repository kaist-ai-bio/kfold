"""Transactional, bounded SQLite shards with lossless per-record compression.

One writer per directory. A committed partial shard is recoverable; completed
shards are immutable. SHA256 validates compressed and decoded payload bytes.
"""
import hashlib, io, json, sqlite3, zlib
from pathlib import Path
import torch

def sha_bytes(b):return hashlib.sha256(b).hexdigest()

def encode(payload):
    f=io.BytesIO();torch.save(payload,f);raw=f.getvalue()
    blob=zlib.compress(raw,1)
    return blob,sha_bytes(blob),sha_bytes(raw),len(raw)

def decode(blob,blob_sha,raw_sha):
    if sha_bytes(blob)!=blob_sha:raise ValueError('Compressed payload checksum mismatch')
    raw=zlib.decompress(blob)
    if sha_bytes(raw)!=raw_sha:raise ValueError('Decoded payload checksum mismatch')
    return torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)

class ShardStore:
    def __init__(self,root,contract_id,max_bytes=1024**3):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.contract_id=contract_id;self.max_bytes=max_bytes;self.entries={};self.db=None
        files=sorted(self.root.glob('shard-*.sqlite*'))
        files=[p for p in files if p.name.endswith(('.sqlite','.partial'))]
        partials=[p for p in files if p.name.endswith('.partial')]
        if len(partials)>1:raise ValueError('More than one partial shard')
        for p in files:
            c=sqlite3.connect(p)
            assert c.execute('PRAGMA quick_check').fetchone()[0]=='ok'
            assert c.execute("SELECT value FROM metadata WHERE key='contract_id'").fetchone()[0]==contract_id,'Contract changed on resume'
            for key,identity in c.execute('SELECT system_id,identity FROM records'):
                if key in self.entries:raise ValueError('Duplicate system across shards')
                self.entries[key]=(p,identity)
            c.close()
        self.next_index=max([int(p.name.split('-')[1].split('.')[0]) for p in files],default=-1)+1
        if partials:self._open(partials[0])

    def _open(self,path=None):
        self.path=path or self.root/f'shard-{self.next_index:06d}.sqlite.partial'
        if path is None:self.next_index+=1
        self.db=sqlite3.connect(self.path)
        self.db.executescript('PRAGMA synchronous=FULL; PRAGMA journal_mode=DELETE; CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE IF NOT EXISTS records(system_id TEXT PRIMARY KEY,identity TEXT,blob_sha TEXT,raw_sha TEXT,raw_bytes INTEGER,payload BLOB);')
        self.db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',('contract_id',self.contract_id));self.db.commit()
        self.bytes=self.db.execute('SELECT coalesce(sum(length(payload)),0) FROM records').fetchone()[0]

    def get(self,key):
        p,_=self.entries[key];c=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True)
        row=c.execute('SELECT payload,blob_sha,raw_sha FROM records WHERE system_id=?',(key,)).fetchone();c.close()
        return decode(*row)

    def add(self,key,payload):
        identity=payload['cache_identity_sha256']
        if key in self.entries:
            if self.entries[key][1]!=identity:raise ValueError('Input/cache identity changed')
            return {'reused':True}
        blob,bsha,rsha,raw_size=encode(payload)
        if self.db is not None and self.bytes and self.bytes+len(blob)>self.max_bytes:self.seal()
        if self.db is None:self._open()
        self.db.execute('INSERT INTO records VALUES (?,?,?,?,?,?)',(key,identity,bsha,rsha,raw_size,blob));self.db.commit()
        self.entries[key]=(self.path,identity);self.bytes+=len(blob)
        return {'reused':False,'compressed_bytes':len(blob),'serialized_bytes':raw_size}

    def seal(self):
        if self.db is None:return
        assert self.db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        count=self.db.execute('SELECT count(*) FROM records').fetchone()[0];self.db.close();self.db=None
        old=self.path;dest=old.with_suffix('');old.rename(dest)
        h=hashlib.sha256()
        with dest.open('rb') as f:
            for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
        dest.with_suffix('.receipt.json').write_text(json.dumps(dict(shard=dest.name,records=count,bytes=dest.stat().st_size,sha256=h.hexdigest(),contract_id=self.contract_id),indent=2)+'\n')
        self.entries={k:(dest if p==old else p,i) for k,(p,i) in self.entries.items()}
