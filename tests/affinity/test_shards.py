import sqlite3
import pytest
import torch
from kfold.affinity.shards import ShardStore
from kfold.affinity.shards import (SHUFFLE_CODEC, compress_bytes, decompress_bytes,
    encode, decode, sha_bytes)

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

def test_crash_between_seal_and_receipt_is_recoverable(tmp_path):
    import json, hashlib
    s=ShardStore(tmp_path,'a');s.add('a',payload(1));s.seal()
    receipt=next(tmp_path.glob('*.receipt.json'));receipt.unlink()
    recovered=ShardStore(tmp_path,'a')
    r=json.loads(receipt.read_text())
    assert r['sha256']==hashlib.sha256(next(tmp_path.glob('*.sqlite')).read_bytes()).hexdigest()
    assert r['records']==1 and recovered.get('a')['cache_identity_sha256']=='1'


@pytest.mark.parametrize('size', [0, 1, 2, 3, 255, 256, 65537])
def test_lossless_byte_order_for_odd_even_lengths(size):
    raw = bytes(i % 251 for i in range(size))
    for codec in ('zlib1', SHUFFLE_CODEC):
        assert decompress_bytes(compress_bytes(raw, codec)) == raw


def test_payload_legacy_compatibility_and_bit_exact_bfloat16():
    import io, zlib
    p = payload(5)
    p['values'] = torch.tensor([0, -32768, 32640, -128, 32705], dtype=torch.int16).view(torch.bfloat16)
    f = io.BytesIO(); torch.save(p, f); raw = f.getvalue()
    old = zlib.compress(raw, 1)
    restored = decode(old, sha_bytes(old), sha_bytes(raw))
    assert torch.equal(restored['values'].view(torch.int16), p['values'].view(torch.int16))
    blob, bsha, rsha, _ = encode(p, SHUFFLE_CODEC)
    restored = decode(blob, bsha, rsha)
    assert torch.equal(restored['values'].view(torch.int16), p['values'].view(torch.int16))


@pytest.mark.parametrize('defect', ['header', 'version', 'length', 'truncated', 'trailing', 'checksum'])
def test_codec_rejects_bad_frames(defect):
    blob = compress_bytes(b'abcdefg', SHUFFLE_CODEC)
    if defect == 'header': blob = blob[:5]
    if defect == 'version': blob = blob[:4] + b'\x02' + blob[5:]
    if defect == 'length': blob = blob[:5] + (6).to_bytes(8, 'big') + blob[13:]
    if defect == 'truncated': blob = blob[:-1]
    if defect == 'trailing': blob += b'extra'
    if defect == 'checksum': blob = blob[:-1] + bytes([blob[-1] ^ 1])
    with pytest.raises(ValueError): decompress_bytes(blob)


def test_codec_resume_partial_sealed_and_old_metadata(tmp_path):
    s = ShardStore(tmp_path/'new', 'a', codec=SHUFFLE_CODEC)
    s.add('a', payload(1)); s.db.close()
    with pytest.raises(ValueError, match='codec changed'): ShardStore(tmp_path/'new', 'a')
    s = ShardStore(tmp_path/'new', 'a', codec=SHUFFLE_CODEC)
    assert s.add('a', payload(1))['reused']
    s.add('b', payload(2)); s.seal()
    assert ShardStore(tmp_path/'new', 'a', codec=SHUFFLE_CODEC).get('b')['cache_identity_sha256'] == '2'
    old = ShardStore(tmp_path/'old', 'a'); old.add('a', payload(1)); old.seal()
    with sqlite3.connect(next((tmp_path/'old').glob('*.sqlite'))) as db:
        db.execute("DELETE FROM metadata WHERE key='codec'")
    assert ShardStore(tmp_path/'old', 'a').get('a')['cache_identity_sha256'] == '1'
    with pytest.raises(ValueError, match='codec changed'):
        ShardStore(tmp_path/'old', 'a', codec=SHUFFLE_CODEC)
