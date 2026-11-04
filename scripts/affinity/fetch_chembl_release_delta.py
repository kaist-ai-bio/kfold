#!/usr/bin/env python3
"""Fetch newly indexed documents after SAIR's ChEMBL35 source snapshot.

This is a conservative new-document subset, NOT a complete activity-level release
diff: later additions to already indexed documents require comparing full dumps.
"""
import concurrent.futures
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import os
import datetime
import urllib.parse
from fetch_chembl_validation import BASE, request


def main():
    root, reuse = map(Path, sys.argv[1:3])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    root.mkdir(parents=True, exist_ok=True)
    (root/'fetch-process.json').write_text(json.dumps({'pid':os.getpid(),'workers':workers,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})+'\n')
    prior = json.loads((reuse/"receipt.json").read_text())
    status = request(BASE+"status.json")
    assert status == prior["status_after"]
    identity = {"chembl_release__chembl_release__in":"CHEMBL_36,CHEMBL_37", "types":["Ki","Kd","IC50","EC50"],"publication_year_filter":None}
    identity_path=root/'query.json'
    if identity_path.exists():
        assert json.loads(identity_path.read_text())==identity
    else:
        identity_path.write_text(json.dumps(identity,indent=2)+'\n')
    def fetch(folder,name,endpoint,params,key):
        path=root/folder/name
        path.parent.mkdir(exist_ok=True)
        url=BASE+endpoint+'.json?'+urllib.parse.urlencode(dict(params,limit=1000))
        if path.exists():
            with gzip.open(path,'rt') as stream:data=json.load(stream)
        else:
            try:
                data=request(url)
            except Exception as exc:
                raise RuntimeError(f"Failed page {folder}/{name}: {url}") from exc
            with gzip.open(path.with_suffix('.part'),'wt') as stream:json.dump(data,stream,separators=(',',':'))
            path.with_suffix('.part').replace(path)
        receipt={'url':url,'file':str(path),'rows':len(data[key]),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        return data,receipt
    doc_filter={"chembl_release__chembl_release__in":"CHEMBL_36,CHEMBL_37","order_by":"document_chembl_id"}
    def docs_page(offset):
        data,receipt=fetch('document',f'{offset:08d}.json.gz','document',dict(doc_filter,offset=offset),'documents')
        assert data['page_meta']['offset']==offset
        assert all((d.get('chembl_release') or {}).get('chembl_release') in ['CHEMBL_36','CHEMBL_37'] for d in data['documents'])
        print('DOCUMENTS',offset,len(data['documents']),flush=True)
        return data,receipt
    first,receipt=docs_page(0)
    total=first['page_meta']['total_count']
    docs=first['documents'];doc_receipts=[receipt]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for data,receipt in pool.map(docs_page,range(1000,total,1000)):
            assert data['page_meta']['total_count']==total
            docs.extend(data['documents']);doc_receipts.append(receipt)
    doc_ids={d['document_chembl_id'] for d in docs}
    assert len(docs)==len(doc_ids)==total
    old_docs=set()
    for page in prior['documents']['pages']:
        with gzip.open(page['file'],'rt') as stream:
            old_docs.update(d['document_chembl_id'] for d in json.load(stream)['documents'])
    assert old_docs <= doc_ids
    activity_receipts=[];activity_ids=[]
    (root/'activity').mkdir(exist_ok=True)
    for index,page in enumerate(prior['activities']['pages']):
        src=Path(page['file']);dst=root/'activity'/f'reused-{index:08d}.json.gz'
        if not dst.exists():shutil.copyfile(src,dst)
        assert hashlib.sha256(dst.read_bytes()).hexdigest()==page['sha256']
        with gzip.open(dst,'rt') as stream:
            acts=json.load(stream)['activities']
        assert all(a['document_chembl_id'] in doc_ids for a in acts)
        activity_ids.extend(a['activity_id'] for a in acts)
        activity_receipts.append(dict(page,file=str(dst),reused_from=str(src)))
    needed=sorted(doc_ids-old_docs)
    def activity_batch(start):
        batch=needed[start:start+50]
        params={'document_chembl_id__in':','.join(batch),'standard_type__in':'Ki,Kd,IC50,EC50','order_by':'activity_id'}
        offset=0;receipts=[];ids=[];total=None
        while total is None or offset<total:
            data,receipt=fetch('activity',f'new-{start:08d}-{offset:08d}.json.gz','activity',dict(params,offset=offset),'activities')
            if total is None:total=data['page_meta']['total_count']
            assert data['page_meta']['total_count']==total
            assert data['page_meta']['offset']==offset
            for a in data['activities']:
                assert a['document_chembl_id'] in batch
                assert a['standard_type'] in identity['types']
                ids.append(a['activity_id'])
            receipts.append(receipt);offset+=1000
        assert len(ids)==len(set(ids))==total
        print('ACTIVITY_BATCH',start,total,flush=True)
        return receipts,ids
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for receipts,ids in pool.map(activity_batch,range(0,len(needed),50)):
            activity_receipts.extend(receipts);activity_ids.extend(ids)
    assert len(activity_ids)==len(set(activity_ids))
    assert request(BASE+'status.json')==status
    (root/'receipt.json').write_text(json.dumps({'status_before':status,'status_after':status,'documents':{'total':len(docs),'pages':doc_receipts},'activities':{'total':len(activity_ids),'pages':activity_receipts},'scope':'Newly indexed documents in releases36/37, no publication-year cutoff. Not a full activity-level release diff.'},indent=2)+'\n')
    print('RELEASE_DELTA_VERIFIED',len(docs),len(activity_ids),flush=True)


if __name__=='__main__':
    main()
