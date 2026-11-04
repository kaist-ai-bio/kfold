#!/usr/bin/env python3
"""Collect publication dates without treating indexing/acceptance as publication."""
import calendar
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET


def interval(node):
    year = node.findtext("Year")
    if not year or not year.isdigit():
        text = node.findtext("MedlineDate") or ""
        years = re.findall(r"\b(?:19|20)\d{2}\b", text)
        if not years:
            return None
        return {"lower": min(years)+"-01-01", "upper": max(years)+"-12-31", "precision": "year_range"}
    year = int(year)
    month_text = node.findtext("Month") or ""
    try:
        month = int(month_text)
    except ValueError:
        month = next((i for i in range(1,13) if calendar.month_abbr[i].lower() == month_text[:3].lower()), None)
    if not month or not 1 <= month <= 12:
        return {"lower": f"{year}-01-01", "upper": f"{year}-12-31", "precision": "year"}
    day_text = node.findtext("Day") or ""
    if day_text.isdigit():
        date = dt.date(year, month, int(day_text)).isoformat()
        return {"lower": date, "upper": date, "precision": "day"}
    return {"lower": f"{year}-{month:02d}-01", "upper": f"{year}-{month:02d}-{calendar.monthrange(year,month)[1]:02d}", "precision": "month"}


def main():
    root = Path(sys.argv[1])
    ids = set()
    for page in sorted((root/"document").glob("*.json.gz")):
        with gzip.open(page,"rt") as stream:
            ids.update(str(d["pubmed_id"]) for d in json.load(stream)["documents"] if d.get("pubmed_id"))
    ids = sorted(ids)
    folder = root/"pubmed"
    folder.mkdir(exist_ok=True)
    records, receipts = {}, []
    for offset in range(0,len(ids),200):
        batch = ids[offset:offset+200]
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id="+",".join(batch)
        path=folder/f"{offset:08d}.xml.gz"
        if path.exists():
            with gzip.open(path,"rb") as stream:
                payload=stream.read()
        else:
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(url,timeout=120) as response:
                        payload=response.read()
                    break
                except Exception:
                    if attempt==3:
                        raise
                    time.sleep(2**(attempt+1))
            with gzip.open(path.with_suffix('.part'),"wb") as stream:
                stream.write(payload)
            path.with_suffix('.part').replace(path)
            time.sleep(.4)
        xml=ET.fromstring(payload)
        got=[]
        for article in xml.findall("PubmedArticle"):
            pmid=article.findtext("MedlineCitation/PMID")
            assert pmid in batch
            got.append(pmid)
            events=[]
            for node in article.findall("MedlineCitation/Article/ArticleDate"):
                date=interval(node)
                if date:
                    events.append(dict(date, source="ArticleDate:"+str(node.get("DateType"))))
            for node in article.findall("MedlineCitation/Article/Journal/JournalIssue/PubDate"):
                date=interval(node)
                if date:
                    events.append(dict(date, source="JournalIssue:PubDate"))
            for node in article.findall("PubmedData/History/PubMedPubDate"):
                if node.get("PubStatus") in {"epublish","ppublish"}:
                    date=interval(node)
                    if date:
                        events.append(dict(date, source="History:"+node.get("PubStatus")))
            records[pmid]={"pmid":pmid,"publication_events":events,"first_known_lower":min((e["lower"] for e in events),default=None),"first_known_upper":min((e["upper"] for e in events),default=None)}
        assert len(got)==len(set(got))
        receipts.append({"file":str(path),"url":url,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"requested":len(batch),"returned":len(got),"missing":sorted(set(batch)-set(got))})
        print("PUBMED",offset,len(got),flush=True)
    (root/"publication-dates.json").write_text(json.dumps(records,indent=2)+"\n")
    (root/"pubmed-receipt.json").write_text(json.dumps({"requested":len(ids),"returned":len(records),"pages":receipts,"interpretation":"Earliest recorded publication event; no accepted/received or PubMed indexing dates; does not establish absence of an earlier preprint or other dataset source."},indent=2)+"\n")
    print("PUBLICATION_DATES_VERIFIED",len(records),flush=True)


if __name__ == "__main__":
    main()
