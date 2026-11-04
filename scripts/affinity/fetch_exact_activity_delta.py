"""Fetch a provisional activity-ID delta without assigning temporal eligibility."""
import concurrent.futures
import gzip
import hashlib
import json
import sys
import urllib.parse
from pathlib import Path
from fetch_chembl_validation import BASE, request

plan_path=Path(sys.argv[1]);root=Path(sys.argv[2]);root.mkdir(parents=True,exist_ok=True)
plan=json.loads(plan_path.read_text());status=request(BASE+'status.json')
assert status['chembl_db_version']=='ChEMBL_37'
identity=dict(plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),status=status)
q=root/'query.json'
if q.exists():assert json.loads(q.read_text())==identity
else:q.write_text(json.dumps(identity,indent=2)+'\n')

def collect(endpoint,key,id_key,ids):
    folder=root/endpoint;folder.mkdir(exist_ok=True)
    names=sorted(set(ids));assert len(names)==len(ids)
    def fetch(start):
        batch=names[start:start+50]
        url=BASE+endpoint+'.json?'+urllib.parse.urlencode({id_key+'__in':','.join(map(str,batch)),'limit':1000})
        p=folder/f'{start:08d}.json.gz'
        if p.exists():
            with gzip.open(p,'rt') as f:d=json.load(f)
        else:
            d=request(url)
            with gzip.open(p.with_suffix('.part'),'wt') as f:json.dump(d,f,separators=(',',':'))
            p.with_suffix('.part').replace(p)
        got=[r[id_key] for r in d[key]]
        assert set(got)==set(batch) and len(got)==len(batch)==d['page_meta']['total_count']
        print(endpoint,start,len(got),flush=True)
        return dict(file=str(p),url=url,rows=len(got),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pages=list(pool.map(fetch,range(0,len(names),50)))
    assert sum(p['rows'] for p in pages)==len(names)
    return dict(total=len(names),pages=pages)

activities=collect('activity','activities','activity_id',plan['activity_ids'])
documents=collect('document','documents','document_chembl_id',plan['document_ids'])
after=request(BASE+'status.json');assert after==status
(root/'receipt.json').write_text(json.dumps(dict(status_before=status,status_after=after,
    activities=activities,documents=documents,scope=plan['source_scope'],
    chembl35_official_verified=False,eligibility_assigned=False),indent=2)+'\n')
print('EXACT_ACTIVITY_DOWNLOAD_VERIFIED',activities['total'],documents['total'],flush=True)
