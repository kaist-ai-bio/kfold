"""Reuse sealed crop metadata to exclude unavailable views without dropping labels."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3

VIEWS=('query_200','query_350')
TASKS=('regression','classification')
SPLITS=('train','validation','external_novel','external_overlap_aux','kinase_aux')


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro&immutable=1',uri=True)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def query_flags(row):
    plans=list(row.get('historical_plans',{}).values()) or [row]
    flags=[]
    for p in plans:
        if p['status']!='final':raise ValueError('Unfinished crop plan')
        flags.append(tuple(view in p['views'] for view in VIEWS))
    if len(set(flags))!=1:raise ValueError('Historical aliases disagree on shared query availability')
    return dict(zip(VIEWS,flags[0]))


def freeze_availability(manifest,catalog,parent_audit,output,*,expected_parent=2795993,expected_new=21898):
    output=Path(output)
    if output.exists():raise ValueError('Completed view eligibility must not repeat')
    gate=json.loads(Path(manifest).with_name('completion.json').read_text())
    if not gate['passed'] or sha(manifest)!=gate['manifest_sha256']:
        raise ValueError('Unsealed or changed label manifest')
    parent=Path(parent_audit)
    proof=json.loads((parent/'completion.json').read_text())
    coverage=json.loads((parent/'coverage.json').read_text())
    if not proof['passed'] or not proof['catalog_unchanged'] or proof['records']!=expected_parent:
        raise ValueError('Parent view audit is incomplete')
    audit=readonly(parent/'catalog-audit.sqlite')
    if audit.execute('SELECT count(*) FROM audit_views').fetchone()[0]!=expected_parent:
        raise ValueError('Parent view audit coverage changed')
    unavailable={v:{x[0] for x in audit.execute('SELECT system_id FROM audit_views WHERE '+column+'=0')}
        for v,column in [('query_200','q200'),('query_350','q350')]}
    audit.close()
    index=readonly(catalog);new_ids=set();plan_pins={}
    bundles=list(index.execute("SELECT path,sha FROM bundles WHERE bundle_id LIKE 'biolip3-%' AND state='complete'"))
    if not bundles:raise ValueError('No sealed new-source bundles')
    for path,bundle_sha in bundles:
        d=readonly(path)
        for source,plan_sha,encoded,state in d.execute('SELECT source_sha,plan_sha,plan,state FROM sources'):
            plan=json.loads(encoded)
            canonical=json.dumps(plan,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            if state!='complete' or hashlib.sha256(canonical).hexdigest()!=plan_sha:
                raise ValueError('Source plan changed')
            plan_pins[source]=plan_sha
            for identity,row in plan['records'].items():
                if identity in new_ids:raise ValueError('Duplicated new-source plan input')
                new_ids.add(identity)
                for view,ok in query_flags(row).items():
                    if not ok:unavailable[view].add(identity)
        d.close()
    if len(new_ids)!=expected_new:raise ValueError('New-source view inventory incomplete')
    if index.execute('SELECT count(*) FROM records').fetchone()[0]!=expected_parent+expected_new:
        raise ValueError('Logical cache changed')
    index.close()
    d=readonly(manifest)
    excluded={v:{task:{split:[] for split in SPLITS} for task in TASKS} for v in VIEWS}
    union=set().union(*unavailable.values());affected=defaultdict(set)
    if union:
        sql='SELECT row_id,input_id FROM examples WHERE input_id IN ('+','.join('?'*len(union))+')'
        for row_id,identity in d.execute(sql,sorted(union)):
            affected[row_id]={v for v in VIEWS if identity in unavailable[v]}
        for task,split,position,row_id in d.execute('SELECT task,split,position,row_id FROM task_index'):
            for view in affected.get(row_id,()):excluded[view][task][split].append(position)
    counts={}
    for task,split,n in d.execute('SELECT task,split,count(*) FROM task_index GROUP BY task,split'):
        counts.setdefault(task,{})[split]=dict(source_eligible=n,
            **{v:n-len(excluded[v][task][split]) for v in VIEWS})
    for v in VIEWS:
        for task in TASKS:
            for split in SPLITS:excluded[v][task][split].sort()
    # Fail closed before the fixed pilot loads a known-unavailable view.
    batches=json.loads(Path(manifest).with_name('pilot-batches.json').read_text())
    for task,values in batches.items():
        ids=[i for batch in values['train'] for i in batch]
        ids.extend(i for rows in values['evaluation'].values() for i in rows)
        ids.extend(i for rows in values.get('compatibility',{}).values() for i in rows)
        for identity, in d.execute('SELECT input_id FROM examples WHERE row_id IN ('+','.join('?'*len(ids))+')',ids):
            if identity in unavailable['query_200']:raise ValueError('Pilot batch needs reviewed view replacement')
    d.close()
    result=dict(passed=True,manifest_sha256=gate['manifest_sha256'],catalog=str(catalog),
        parent_audit=str(parent),parent_catalog_sha256=coverage['catalog_sha256'],
        reused_parent_view_records=expected_parent,new_view_records=expected_new,
        new_plan_pins=plan_pins,unavailable_input_ids={v:sorted(ids) for v,ids in unavailable.items()},
        excluded_positions=excluded,counts=counts,labels_dropped=False,raw_payloads_read=False,
        physical_cache_modified=False,source_hash_audit_repeated=False)
    temporary=output.with_suffix('.partial');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(output)
    return result
