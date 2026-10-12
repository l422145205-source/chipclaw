"""Actual HTTP evidence pipeline: real weights, retrieval and original PDF cells."""
import hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from chipclaw import api
from chipclaw.config import ROOT
def main():
    folder=ROOT/'reports/runtime-v4';folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists():raise ValueError('Frozen real HTTP probes exist')
    # Use the actual documented startup configuration, not a replaced lifespan.
    # Every handler, model forward, PDF extraction and numeric check is real.
    original=os.environ.get('CHIPCLAW_API_WARMUP');os.environ['CHIPCLAW_API_WARMUP']='none'
    questions={'en':'What is the typical supply current of MAX3232 with no load and VCC 3.3 V to 5 V?',
        'zh':'MAX3232 在空载、VCC 3.3 V 至 5 V 时，典型供电电流是多少？'}
    claims=['No load, V CC = 3.3 V to 5 V, the typical supply current is 0.3 mA.',
        'No load, V CC = 3.3 V to 5 V, the typical supply current is 1 mA.',
        'The typical supply current at no load is 0.3 mA.']
    records=[]
    corpus_records=[]
    try:
        with TestClient(api.app) as client:
            for method in ['multilingual_v4','domain_teacher_v4','adaptive_v4','visual_fusion_v4']:
                for language,question in questions.items():
                    request={'question':question,'doc_id':'max3232','retrieval_method':method,'claims':claims,'budget_chars':6000}
                    response=client.post('/evidence',json=request);assert response.status_code==200,response.text[:500]
                    result=response.json();assert result['metrics']['generation_calls']==0 and result['budget']['used_chars']<=6000
                    assert result['retrieval_trace']['method']==method
                    for span in result['evidence']:
                        assert span['doc_id']=='max3232' and span['source_sha256'] and span['start']<span['end']
                    verdicts=[r['check']['verdict'] for r in result['claims']]
                    assert verdicts[1]!='supported' and verdicts[2]!='supported'
                    name=f'{method}-{language}.json';path=folder/name
                    path.write_text(json.dumps({'request':request,'http_status':200,'result':result},ensure_ascii=False,indent=2),encoding='utf-8')
                    records.append({'file':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'method':method,'language':language,
                        'verdicts':verdicts,'retrieved_pages':result['retrieved_pages'],'used_chars':result['budget']['used_chars']})
                    print('real evidence HTTP',method,language,verdicts,flush=True)
            # One deterministic existing frozen intent per document, without
            # putting its known source page or answer into the request.
            queries=[json.loads(line) for line in (ROOT/'data/ranking-v3-frozen/queries.jsonl').read_text(encoding='utf-8').splitlines()]
            from chipclaw.corpus_v3 import SOURCES_V3
            for number,doc_id in enumerate(sorted(SOURCES_V3)):
                q=min((q for q in queries if q['doc_id']==doc_id),key=lambda q:hashlib.sha256(q['id'].encode()).hexdigest())
                language='en' if number%2==0 else 'zh'
                request={'question':q['query_'+language],'doc_id':doc_id,'retrieval_method':'multilingual_v4','budget_chars':6000}
                response=client.post('/evidence',json=request);assert response.status_code==200,response.text[:500]
                result=response.json();assert result['metrics']['generation_calls']==0 and result['budget']['used_chars']<=6000
                assert all(s['doc_id']==doc_id and s['source_sha256'] and s['text'] for s in result['evidence'])
                name='corpus-'+doc_id+'.json';path=folder/name
                path.write_text(json.dumps({'intent_id':q['id'],'request':request,'http_status':200,'result':result},ensure_ascii=False,indent=2),encoding='utf-8')
                corpus_records.append({'file':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'doc_id':doc_id,
                    'language':language,'retrieved_pages':result['retrieved_pages'],'evidence_spans':len(result['evidence']),
                    'used_chars':result['budget']['used_chars']})
                print('real corpus HTTP',doc_id,language,len(result['evidence']),flush=True)
    finally:
        if original is None:os.environ.pop('CHIPCLAW_API_WARMUP',None)
        else:os.environ['CHIPCLAW_API_WARMUP']=original
    summary={'probes':records,'http_200':len(records),'fully_bound_numeric_probes':sum(r['verdicts']==['supported','contradicted','unknown'] for r in records),
        'corpus_probes':corpus_records,'corpus_smoke_http_200':len(corpus_records),'corpus_documents':len(corpus_records),
        'corpus_nonempty_evidence':sum(r['evidence_spans']>0 for r in corpus_records),
        'source_page_injections':0,'generation_calls':0,'ollama_calls':0,'mocked_retrieval_or_verification':False,
        'actual_application_lifespan':True,'startup_configuration':'CHIPCLAW_API_WARMUP=none',
        'scope':'One MAX3232 question in EN/ZH across four methods plus one existing frozen intent per each of 20 documents; actual application startup, HTTP, models and PDFs. Corpus probes check execution and provenance, not semantic answer accuracy; no generated-answer or H accuracy claim.'}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
