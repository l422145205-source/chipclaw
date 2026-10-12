"""Calibrate on actual BM25 chunk pools; never inject a labelled source page."""
import argparse,collections,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chipclaw.config import ROOT
from chipclaw.retrieval_v3 import V3Retriever
from chipclaw.retrieval_v4 import candidate_text
from chipclaw.cascade_v3 import ordered,cascade_order
from chipclaw.cascade_v4 import MultilingualStudent,AdaptiveRanker,FEATURES,features,fit_gain,expected_gain
from chipclaw.teacher_v3 import QwenTeacher,REVISION
def rows(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
def dump(path,v):path.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def page_rr(rank,pages,gold):
    unique=list(dict.fromkeys(pages[i] for i in rank))
    return 1/(unique.index(gold)+1) if gold in unique[:10] else 0.
def heldout_subset(queries):
    grouped=collections.defaultdict(list)
    for q in queries:
        if q['split']=='new_holdout':grouped[q['doc_id']].append(q)
    return [q for doc,group in sorted(grouped.items()) for q in sorted(group,key=lambda q:hashlib.sha256(q['id'].encode()).hexdigest())[:20]]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',default='adaptive-v4');parser.add_argument('--domain',action='store_true');args=parser.parse_args()
    if Path(args.report).name!=args.report:raise ValueError('Simple report name required')
    torch.set_num_threads(4);folder=ROOT/'reports'/args.report;folder.mkdir(exist_ok=True)
    if (folder/'protocol.json').exists():raise ValueError('Frozen calibration run exists')
    queries=rows(ROOT/'data/ranking-v3-frozen/queries.jsonl');dev=[q for q in queries if q['split']=='dev'];holdout=heldout_subset(queries)
    protocol={'selection_split':'dev','dev_intents':len(dev),'reused_holdout_intents':len(holdout),'candidate_count':16,'teacher_top_n':6,
        'candidate_policy':'Original and protected terminology BM25 views with RRF on full 10668 chunks, document filtered. No positive/page injection.',
        'labels':'Weak source-page reciprocal rank; alternate pages are unjudged. Not row correctness or answer accuracy.',
        'gate_fit':'Per-language ridge expected source-page RR gain; semantic-ID hash80% dev fit,20% dev threshold calibration.',
        'gate_budget':.35,'acceptance':'Each language calibration RR must improve and empirical harmful fraction of escalations <=0.15. Otherwise disable.',
        'holdout_note':'Same previously used v3 holdout subset; regression, not a new blind benchmark.',
        'corpus_sha256':sha(ROOT/'data/corpus-v3/index/chunks.jsonl'),'data_sha256':sha(ROOT/'data/ranking-v3-frozen/queries.jsonl'),
        'teacher_kind':'domain_lora_v4' if args.domain else 'public_qwen3',
        'source_sha256':{p:sha(ROOT/p) for p in ['scripts/calibrate_adaptive_v4.py','chipclaw/cascade_v4.py','chipclaw/retrieval_v4.py']}}
    dump(folder/'protocol.json',protocol);dump(folder/'benchmark_queries.json',holdout)
    retriever=V3Retriever();student=MultilingualStudent()
    if args.domain:
        from chipclaw.domain_teacher_v4 import DomainTeacher
        teacher=DomainTeacher()
    else:teacher=QwenTeacher()
    dev_records=[];started=time.perf_counter()
    with (folder/'dev_candidates.jsonl').open('w',encoding='utf-8') as log:
        for n,q in enumerate(dev):
            for lang in ['en','zh']:
                question=q['query_'+lang];chunks=[retriever.chunks[i] for i in retriever.lexical_candidates(question,q['doc_id'])[:16]]
                ids=[c['id'] for c in chunks];pages=[c['page'] for c in chunks];texts=[candidate_text(c) for c in chunks]
                ss=student.predict([(question,p) for p in texts]);ts=teacher.predict([(question,p) for p in texts]);rank=ordered(ss,ids);head=rank[:6]
                cascade=cascade_order(ss,ids,[ts[i] for i in head],6)
                record={'id':q['id'],'language':lang,'doc_id':q['doc_id'],'source_page':q['source_page'],'candidate_ids':ids,'pages':pages,
                    'student_scores':ss,'teacher_scores':ts,'source_page_present':q['source_page'] in pages,
                    'student_rr':page_rr(rank,pages,q['source_page']),'teacher_rr':page_rr(ordered(ts,ids),pages,q['source_page']),
                    'cascade_rr':page_rr(cascade,pages,q['source_page']),'features':features(question,ss,pages).tolist(),
                    'fold':'calibration' if int(hashlib.sha256(q['id'].encode()).hexdigest()[:8],16)%5==0 else 'fit'}
                dev_records.append(record);log.write(json.dumps(record)+'\n');log.flush()
            if (n+1)%40==0:print('actual dev retrieval pools',n+1,'/',len(dev),flush=True)
    language_models={};calibration={}
    for lang in ['en','zh']:
        fit=[r for r in dev_records if r['language']==lang and r['fold']=='fit'];cal=[r for r in dev_records if r['language']==lang and r['fold']=='calibration']
        model=fit_gain([r['features'] for r in fit],[r['cascade_rr']-r['student_rr'] for r in fit]);prediction=[expected_gain(model,r['features']) for r in cal]
        thresholds=[1e6]+sorted(set(float(x) for x in np.quantile(prediction,np.linspace(0,1,21))));trials=[]
        base=float(np.mean([r['student_rr'] for r in cal]))
        for threshold in thresholds:
            flags=[p>=threshold for p in prediction];calls=sum(flags);harm=sum(f and r['cascade_rr']<r['student_rr'] for f,r in zip(flags,cal))
            rr=float(np.mean([r['cascade_rr'] if f else r['student_rr'] for f,r in zip(flags,cal)]))
            trials.append({'threshold':threshold,'calls':calls,'call_rate':calls/len(cal),'harmful_escalations':harm,'harm_rate':harm/max(1,calls),'page_mrr10':rr})
        feasible=[t for t in trials if t['call_rate']<=.35 and t['harm_rate']<=.15]
        choice=max(feasible,key=lambda t:(t['page_mrr10'],-t['calls'],t['threshold']))
        enabled=choice['page_mrr10']>base+1e-9
        model.update(threshold=choice['threshold'] if enabled else 1e6,enabled=enabled,fit_rows=len(fit),calibration_rows=len(cal))
        language_models[lang]=model;calibration[lang]={'base_page_mrr10':base,'chosen':choice,'enabled':enabled,'trials':trials}
    gate={'selection_split':'dev','features':FEATURES,'student_weight_sha256':student.weight_sha,'teacher_revision':REVISION,
        'teacher_identity':teacher.identity(),'teacher_top_n':6,'corpus_sha256':protocol['corpus_sha256'],'languages':language_models,
        'calibration':calibration,'dev_candidates_sha256':sha(folder/'dev_candidates.jsonl'),'scope':'Actual full-chunk candidates; weak source-page expected gain; empirical calibration only, no formal risk guarantee.'}
    dump(folder/'gate.json',gate);print('Frozen language gates',json.dumps({l:m['enabled'] for l,m in language_models.items()}),flush=True)
    ranker=AdaptiveRanker(student,teacher,folder/'gate.json');records=[]
    warm=dev_records[0];texts=[candidate_text(retriever.by_id[c]) for c in warm['candidate_ids']]
    student.predict([('warmup',p) for p in texts]);teacher.predict([('warmup',p) for p in texts])
    with (folder/'timings.jsonl').open('w',encoding='utf-8') as log:
        for n,q in enumerate(holdout):
            for lang in ['en','zh']:
                question=q['query_'+lang];chunks=[retriever.chunks[i] for i in retriever.lexical_candidates(question,q['doc_id'])[:16]]
                ids=[c['id'] for c in chunks];pages=[c['page'] for c in chunks];texts=[candidate_text(c) for c in chunks]
                modes=['multilingual_v4','teacher_all','adaptive_v4'];offset=(2*n+int(lang=='zh'))%3;modes=modes[offset:]+modes[:offset]
                for mode in modes:
                    start=time.perf_counter()
                    if mode=='teacher_all':
                        ts=teacher.predict([(question,p) for p in texts]);order=ordered(ts,ids)
                        info={'teacher_pairs':len(ids),'teacher_forward_batches':teacher.last_metrics['forward_batches'],'student_forward_batches':0,'escalated':True}
                    else:order=ranker.rank(question,ids,texts,pages,mode);info=ranker.last_metrics
                    record={'id':q['id'],'language':lang,'doc_id':q['doc_id'],'mode':mode,'page_rr10':page_rr(order,pages,q['source_page']),
                        'source_page_present':q['source_page'] in pages,**info,'latency_ms':(time.perf_counter()-start)*1000}
                    records.append(record);log.write(json.dumps(record)+'\n');log.flush()
            if (n+1)%20==0:print('actual heldout inference',n+1,'/',len(holdout),flush=True)
    def stats(part):return {'lists':len(part),'page_mrr10':float(np.mean([r['page_rr10'] for r in part])),
        'source_page_candidate_recall':float(np.mean([r['source_page_present'] for r in part])),
        'latency_mean_ms':float(np.mean([r['latency_ms'] for r in part])),'latency_p95_ms':float(np.quantile([r['latency_ms'] for r in part],.95)),
        'teacher_query_calls':sum(r['teacher_pairs']>0 for r in part),'teacher_pairs':sum(r['teacher_pairs'] for r in part)}
    summary={'protocol':protocol,'gate_calibration':calibration,'modes':{m:stats([r for r in records if r['mode']==m]) for m in modes},
        'languages':{l:{m:stats([r for r in records if r['mode']==m and r['language']==l]) for m in modes} for l in ['en','zh']},
        'elapsed_seconds':time.perf_counter()-started,'timings_sha256':sha(folder/'timings.jsonl')}
    dump(folder/'summary.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
