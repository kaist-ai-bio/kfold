import sqlite3
import sys
from types import SimpleNamespace
import pytest
from kfold.affinity.training_monitor import validation_positions,scalar_report,start_tracking


def test_validation_panel_preserves_strata_filters_unavailable_and_is_fixed(tmp_path):
    path=tmp_path/'labels.sqlite';d=sqlite3.connect(path)
    d.executescript('CREATE TABLE examples(row_id INTEGER,cohort TEXT,endpoint TEXT);CREATE TABLE task_index(task TEXT,split TEXT,position INTEGER,row_id INTEGER);')
    for position in range(16):
        d.execute('INSERT INTO examples VALUES(?,?,?)',(position,'measured' if position<12 else 'synthetic','Ki' if position<6 else 'IC50'))
        d.execute('INSERT INTO task_index VALUES(?,?,?,?)',('classification','validation',position,position))
    d.execute('INSERT INTO task_index VALUES(?,?,?,?)',('classification','external_novel',1000,0));d.commit();d.close()
    class Data:
        task='classification';split='validation'
        def connect(self):return sqlite3.connect(path)
        def logical_position(self,p):return None if p==0 else p-1
    a=validation_positions(Data(),per_stratum=2,seed=51)
    assert len(a)==6 and a==sorted(set(a)) and all(0<=i<15 for i in a)
    assert a==validation_positions(Data(),per_stratum=2,seed=51)
    assert a!=validation_positions(Data(),per_stratum=2,seed=91)
    assert len(validation_positions(Data(),per_stratum=100,seed=51))==15
    bad=Data();bad.split='external_novel'
    with pytest.raises(ValueError):validation_positions(bad,per_stratum=2,seed=1)


def test_tracking_never_uploads_sources_and_requires_existing_run_on_resume(tmp_path,monkeypatch):
    calls=[];metrics=[]
    class Settings:
        def __init__(self,**kw):self.values=kw
    def init(**kw):
        calls.append(kw);return SimpleNamespace(define_metric=lambda *a,**k:metrics.append((a,k)))
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(init=init,Settings=Settings))
    settings=dict(enabled=True,entity='K-Fold-Affinity',project='initial training',run_ids={'regression':'fixed'},group='baseline')
    assert start_tracking({},tmp_path,task='regression',variant='colleague_cross',contract='a',config={'seed':1}) is None
    for resume in (False,True):
        start_tracking(settings,tmp_path,task='regression',variant='colleague_cross',contract='a',
                       config={'seed':1,'variant':'colleague_cross'},resume=resume)
    assert [x['resume'] for x in calls]==['never','must']
    assert calls[0]['settings'].values['save_code'] is False
    assert calls[0]['settings'].values['disable_git'] is True
    assert calls[0]['config']['training_contract_sha256']=='a'
    assert scalar_report({'measured':{'MAE':1.,'Pearson':None,'n':5}},'validation_panel')=={'validation_panel/measured/MAE':1.,'validation_panel/measured/n':5}
