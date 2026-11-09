import importlib.util
from pathlib import Path
import sys,types
import pytest

base=Path(__file__).parents[2]/'src/kfold/affinity'
pkg=types.ModuleType('_quota_test_pkg');pkg.__path__=[str(base)];sys.modules[pkg.__name__]=pkg
spec=importlib.util.spec_from_file_location('_quota_test_pkg.session_quota',base/'session_quota.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Fake:
    def __init__(self,fs):self.fs=fs;self.calls=0;self.closed=False;self.error=False
    def read_fresh(self):
        self.calls+=1
        if self.error:raise TimeoutError('query')
        return f'{self.fs} {self.calls} 10000000000 10000000000 - 1 5000000 5000000 -\n'
    def close(self):self.closed=True

def test_every_call_both_mounts_fresh_without_process_recreation():
    q=m.SessionQuota(Fake);a=q();b=q()
    assert all(s.calls==2 for s in q.sessions.values())
    assert b['/scratch']['free_bytes']==a['/scratch']['free_bytes']-1024
    q.close();assert all(s.closed for s in q.sessions.values())

def test_error_latches_and_closes_both_without_retry():
    q=m.SessionQuota(Fake);q.sessions['/appsdata'].error=True
    with pytest.raises(TimeoutError):q()
    calls=[s.calls for s in q.sessions.values()]
    assert all(s.closed for s in q.sessions.values())
    with pytest.raises(RuntimeError):q()
    assert calls==[s.calls for s in q.sessions.values()]

def test_process_owner_mismatch_blocks_use():
    q=m.SessionQuota(Fake);q.owner=-1
    with pytest.raises(RuntimeError):q()
    assert all(s.calls==0 for s in q.sessions.values());q.close()


def test_partial_initialization_closes_first_session():
    made=[]
    def factory(fs):
        if made:raise OSError('launch failed')
        s=Fake(fs);made.append(s);return s
    with pytest.raises(OSError):m.SessionQuota(factory)
    assert made[0].closed
