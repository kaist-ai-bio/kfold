import sqlite3
import pytest
import torch
from kfold.affinity.shards import ShardStore

def payload(i):return {'cache_identity_sha256':str(i),'values':torch.arange(24).to(torch.bfloat16)}

def test_partial_recovery_rotation_and_identity(tmp_path):
    s=ShardStore(tmp_path,'a',max_bytes=1);s.add('a',payload(1));s.db.close()
    s=ShardStore(tmp_path,'a',max_bytes=1)
    assert torch.equal(s.get('a')['values'],payload(1)['values'])
    with pytest.raises(ValueError):s.add('a',payload(2))
    s.add('b',payload(2));s.seal()
    assert len(list(tmp_path.glob('*.sqlite')))==2
    assert ShardStore(tmp_path,'a').get('b')['cache_identity_sha256']=='2'
    with pytest.raises(AssertionError):ShardStore(tmp_path,'other')

def test_corruption_is_not_reused(tmp_path):
    s=ShardStore(tmp_path,'a');s.add('a',payload(1));s.seal()
    p=next(tmp_path.glob('*.sqlite'));db=sqlite3.connect(p)
    db.execute('UPDATE records SET payload=?',(b'corrupt',));db.commit();db.close()
    with pytest.raises(ValueError):ShardStore(tmp_path,'a').get('a')
