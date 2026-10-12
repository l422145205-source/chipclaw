"""Actual image MaxSim/text/fusion on the query-independent frozen page subset."""
import collections,gc,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chipclaw.config import ROOT
from chipclaw.visual_v3 import ColSmolEncoder
from chipclaw.retrieval import Retriever,rrf
from chipclaw.query_rewrite import terminology_query

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    torch.set_num_threads(4);folder=ROOT/'reports/visual-v3';folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists():raise ValueError('Frozen visual experiment exists')
    manifest=json.loads((ROOT/'data/visual-v3/index_manifest.json').read_text());pages=manifest['pages']
    source_pages={(p['doc_id'],p['page']) for p in pages}
    eligible=[q for q in rows(ROOT/'data/ranking-v3-frozen/queries.jsonl') if (q['doc_id'],q['source_page']) in source_pages]
    groups=collections.defaultdict(list)
    for q in eligible:groups[q['doc_id']].append(q)
    selected=[]
    for doc,group in sorted(groups.items()):
        # Fixed hash-stratified source selection, never selected from retrieval scores.
        selected+=sorted(group,key=lambda q:hashlib.sha256(q['id'].encode()).hexdigest())[:10]
    dump(folder/'queries_frozen.json',selected)
    run={'subset_sha256':manifest['subset_sha256'],'index_manifest_sha256':sha(ROOT/'data/visual-v3/index_manifest.json'),
         'query_bank_sha256':sha(ROOT/'data/ranking-v3-frozen/queries.jsonl'),'query_selection_sha256':sha(folder/'queries_frozen.json'),
         'pages':len(pages),'semantic_queries':len(selected),'languages':['en','zh'],'source_labels':'Template source anchors, not blind human QA.',
         'page_selection_preceded_query_bank':True,'selection':'Up to 10 hash-selected eligible intents per device; no retrieval results used.',
         'comparison':'All modes search same frozen 240 pages across 20 devices; source-page relevance only.',
         'latency_note':'Teacher inference may run concurrently during this diagnostic; ranking scores unaffected, latency is not an isolated service benchmark.'}
    vectors=[]
    for p in pages:
        path=ROOT/p['vectors_path'];assert sha(path)==p['vectors_sha256'];vectors.append(np.load(path)['vectors'])
    width=max(len(v) for v in vectors);padded=np.zeros((len(vectors),width,128),dtype=np.float32);mask=np.zeros((len(vectors),width),dtype=bool)
    for i,v in enumerate(vectors):padded[i,:len(v)]=v;mask[i,:len(v)]=True
    docs=torch.tensor(padded,device='cuda');valid=torch.tensor(mask,device='cuda')
    texts={(p['doc_id'],p['page']):p['text'] for p in rows(ROOT/'data/corpus-v3/pages.jsonl')}
    index=folder/'text_index';index.mkdir(exist_ok=True)
    (index/'chunks.jsonl').write_text(''.join(json.dumps({'id':f'{p["doc_id"]}:{p["page"]}','doc_id':p['doc_id'],'page':p['page'],'text':texts[p['doc_id'],p['page']]},ensure_ascii=False)+'\n' for p in pages),encoding='utf-8')
    text=Retriever(dense=False,index_dir=index);encoder=ColSmolEncoder();run['encoder']=encoder.identity();dump(folder/'run.json',run)
    encoder.queries([selected[0]['query_en']]);results=[];started=time.perf_counter();query_folder=ROOT/'.cache/v3/visual_query_vectors';query_folder.mkdir(exist_ok=True)
    with (folder/'predictions.jsonl').open('w',encoding='utf-8') as out:
        for n,q in enumerate(selected):
            for lang in ['en','zh']:
                query=q['query_'+lang];begin=time.perf_counter();qv=encoder.queries([query])[0];torch.cuda.synchronize();encoding_ms=(time.perf_counter()-begin)*1000
                target=query_folder/(q['id']+'_'+lang+'.npz');np.savez_compressed(target,vectors=qv)
                begin=time.perf_counter()
                scores=torch.einsum('qd,pkd->pqk',torch.tensor(qv.astype(np.float32),device='cuda'),docs).masked_fill(~valid[:,None,:],float('-inf')).max(dim=-1).values.sum(dim=-1).cpu().numpy()
                torch.cuda.synchronize();search_ms=(time.perf_counter()-begin)*1000
                visual=sorted(range(len(pages)),key=lambda i:(-float(scores[i]),text.chunks[i]['id']))
                begin=time.perf_counter();expanded,_=terminology_query(query);views=list(dict.fromkeys(v for v in [query,expanded] if v))
                lexical,_=rrf([text._rank_query(v,list(range(len(pages))),'bm25')[0] for v in views]);text_ms=(time.perf_counter()-begin)*1000
                # BM25 can have no matching terms; absent pages stay absent, rank >240.
                fusion,_=rrf([lexical,visual]);gold=next(i for i,p in enumerate(pages) if (p['doc_id'],p['page'])==(q['doc_id'],q['source_page']))
                ranks={name:(ranking.index(gold)+1 if gold in ranking else 241) for name,ranking in [('text',lexical),('visual',visual),('fusion',fusion)]}
                record={'id':q['id'],'language':lang,'doc_id':q['doc_id'],'split':q['split'],'anchor_kind':q['anchor_kind'],'gold_page':q['source_page'],'ranks':ranks,
                        'latency_ms':{'query_image_model':encoding_ms,'maxsim':search_ms,'text':text_ms},
                        'query_vector_path':str(target.relative_to(ROOT)).replace('\\','/'),'query_vector_sha256':sha(target),
                        'visual_top5':[{'doc_id':pages[i]['doc_id'],'page':pages[i]['page'],'score':float(scores[i])} for i in visual[:5]]}
                out.write(json.dumps(record,ensure_ascii=False)+'\n');out.flush();results.append(record)
            if (n+1)%20==0:print('visual queries',n+1,'/',len(selected),flush=True)
    def metrics(part):
        return {mode:{'n':len(part),'mrr10':sum(1/r['ranks'][mode] if r['ranks'][mode]<=10 else 0 for r in part)/max(1,len(part)),
                      'recall5':sum(r['ranks'][mode]<=5 for r in part)/max(1,len(part)),'top1':sum(r['ranks'][mode]==1 for r in part)} for mode in ['text','visual','fusion']}
    summary={'all':metrics(results),'languages':{l:metrics([r for r in results if r['language']==l]) for l in ['en','zh']},
             'splits':{s:metrics([r for r in results if r['split']==s]) for s in ['train','dev','new_holdout']},
             'elapsed_seconds':time.perf_counter()-started,'predictions_sha256':sha(folder/'predictions.jsonl'),
             'mean_latency_ms':{k:float(np.mean([r['latency_ms'][k] for r in results])) for k in ['query_image_model','maxsim','text']},
             'scope':'240 frozen pages only, no image reasoning/visual model training/human QA accuracy claim.'}
    dump(folder/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
