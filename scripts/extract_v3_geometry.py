"""Extract source geometry for the already frozen visual subset, before queries."""
import gc
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import pdfplumber
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chipclaw.corpus_v3 import V3_RAW
from chipclaw.config import ROOT
from chipclaw.geometry_worker import extract_page

def main():
    subset_path=ROOT/'data/corpus-v3/visual_subset.json'
    subset=json.loads(subset_path.read_text(encoding='utf-8'))
    folder=ROOT/'data/corpus-v3/geometry';folder.mkdir(exist_ok=True)
    docs=sorted({p['doc_id'] for p in subset['pages']});records=[]
    for doc in docs:
        destination=folder/(doc+'.json')
        if destination.is_file():
            record=json.loads(destination.read_text(encoding='utf-8'))
            assert record['subset_sha256']==hashlib.sha256(subset_path.read_bytes()).hexdigest()
        else:
            wanted=[r for r in subset['pages'] if r['doc_id']==doc]
            with pdfplumber.open(V3_RAW/(doc+'.pdf')) as pdf:
                pages={}
                for row in wanted:
                    pages[str(row['page'])]=extract_page(pdf.pages[row['page']-1])
                    pdf.pages[row['page']-1].close()
            record={'doc_id':doc,'pdf_sha256':wanted[0]['source_sha256'],
                    'subset_sha256':hashlib.sha256(subset_path.read_bytes()).hexdigest(),
                    'pdfplumber':pdfplumber.__version__,'pdfminer':metadata.version('pdfminer.six'),
                    'pages':pages}
            destination.write_text(json.dumps(record,ensure_ascii=False),encoding='utf-8')
        print('geometry',doc,len(record['pages']),sum(len(p['records']) for p in record['pages'].values()),flush=True)
        records.append({'doc_id':doc,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'pages':len(record['pages'])})
        del record;gc.collect()
    (folder/'manifest.json').write_text(json.dumps({'files':records,'query_access':False},indent=2),encoding='utf-8')

if __name__=='__main__':main()
