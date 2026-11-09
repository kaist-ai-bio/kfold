import importlib.util
from pathlib import Path
import sys
import pytest

spec=importlib.util.spec_from_file_location('quota_session',Path(__file__).parents[2]/'src/kfold/affinity/quota_session.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def child(mode):
    code='''import sys,time
footer=%r
mode=%r
n=0
for line in sys.stdin:
 if line.startswith('quota'):
  n+=1
  if mode=='timeout':time.sleep(2)
  if mode=='exit':sys.exit(1)
  if mode=='error':print('quota: failed',flush=True)
  if mode=='wrong':print('/other 1 2 3 - 4 5 6 -',flush=True)
  else:print('/scratch %%d 20 20 - 4 5 5 -'%%n,flush=True)
 elif line.strip()=='help':
  if mode=='truncated':print('usage: incomplete',flush=True)
  else:sys.stdout.write(footer);sys.stdout.flush()
''' % (m.FOOTER.decode(),mode)
    return [sys.executable,'-u','-c',code]

def test_every_request_runs_command_again_same_process():
    s=m.QuotaSession('/scratch',command=child('ok'),timeout=1)
    try:
        pid=s.process.pid
        assert s.read_fresh().split()[1]=='1'
        assert s.read_fresh().split()[1]=='2'
        assert s.process.pid==pid
    finally:s.close()
    assert s.process.poll() is not None

@pytest.mark.parametrize('mode',['timeout','exit','error','wrong','truncated'])
def test_any_failure_poisons_and_stops_process(mode):
    s=m.QuotaSession('/scratch',command=child(mode),timeout=.15)
    with pytest.raises((RuntimeError,TimeoutError)):s.read_fresh()
    assert s.failed and s.process.poll() is not None
    with pytest.raises(RuntimeError):s.read_fresh()
    s.close()

def test_unapproved_mount_rejected():
    with pytest.raises(ValueError):m.QuotaSession('/home')
