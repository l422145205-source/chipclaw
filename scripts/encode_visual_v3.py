"""Render/encode exactly the frozen pages; queries cannot affect page admission."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pypdfium2 as pdfium
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chipclaw.config import ROOT
from chipclaw.corpus_v3 import V3_RAW
from chipclaw.visual_v3 import ColSmolEncoder

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    import torch
    torch.set_num_threads(4);torch.manual_seed(42)
    folder=ROOT/'data/visual-v3';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'index_manifest.json').exists():raise ValueError('Frozen visual index already encoded')
    images=ROOT/'.cache/v3/page_images';vectors=ROOT/'.cache/v3/visual_vectors'
    images.mkdir(parents=True,exist_ok=True);vectors.mkdir(parents=True,exist_ok=True)
    subset_path=ROOT/'data/corpus-v3/visual_subset.json';subset=json.loads(subset_path.read_text(encoding='utf-8'))
    encoder=ColSmolEncoder();records=[];started=time.perf_counter();documents={}
    for row in subset['pages']:
        doc,page=row['doc_id'],row['page'];stem=f'{doc}_p{page}'
        png=images/(stem+'.png');npz=vectors/(stem+'.npz')
        if doc not in documents:documents[doc]=pdfium.PdfDocument(V3_RAW/(doc+'.pdf'))
        image=documents[doc][page-1].render(scale=1.5).to_pil().convert('RGB')
        image.save(png)
        encoded=encoder.images([image])[0]
        if not np.isfinite(encoded).all() or encoded.shape[1]!=128:raise ValueError('Invalid visual embeddings')
        np.savez_compressed(npz,vectors=encoded)
        records.append({**row,'image_path':str(png.relative_to(ROOT)).replace('\\','/'),'image_sha256':sha(png),
                        'image_pixels':list(image.size),'render_scale':1.5,
                        'vectors_path':str(npz.relative_to(ROOT)).replace('\\','/'),'vectors_sha256':sha(npz),
                        'tokens':len(encoded),'dimensions':encoded.shape[1],'encoding_ms':encoder.last_metrics['latency_ms']})
        image.close()
        if len(records)%12==0:print('encoded',len(records),'pages',round(time.perf_counter()-started,1),'seconds',flush=True)
    for document in documents.values():document.close()
    manifest={'version':'visual-v3','subset_sha256':sha(subset_path),'pages':records,'page_count':len(records),
              'frozen_before_queries':subset['frozen_before_query_generation'],'encoder':encoder.identity(),
              'elapsed_seconds':time.perf_counter()-started,'vectors_total_bytes':sum((ROOT/r['vectors_path']).stat().st_size for r in records),
              'peak_vram_bytes':torch.cuda.max_memory_allocated(),'trained_this_run':False,
              'scope':'240 selected pages, not all 1,822 pages. Official pretrained page-image multi-vector retrieval only.'}
    (folder/'index_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['page_count','elapsed_seconds','vectors_total_bytes','peak_vram_bytes']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
