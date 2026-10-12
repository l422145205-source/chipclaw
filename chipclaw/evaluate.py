"""Reproducible evaluation. Gold data is consumed only by this module."""
import hashlib
import json
import platform
import re
import time
from datetime import datetime, timezone
import numpy as np
import requests
from .config import DATA, ROOT, INDEX, MODEL, OLLAMA, EMBEDDING, EMBEDDING_REVISION, RERANKER, RERANKER_REVISION
from .retrieval import Retriever
from .generation import generate, verify, validate_citations


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def rubric_pass(question, answer, refused):
    if not question['answerable']:
        return refused
    if refused:
        return False
    positions=[answer.lower().find(p.lower()) for p in question.get('ordered_patterns',[])]
    if positions and (min(positions)<0 or positions!=sorted(positions)):
        return False
    return (all(any(re.search(p, answer, re.I) for p in group) for group in question['requirements'])
            and not any(re.search(p, answer, re.I) for p in question.get('forbidden', [])))


def retrieval_metrics(hits, gold):
    relevant = [i for i, c in enumerate(hits, 1) if c['page'] in gold]
    return {'hit_at_5': bool(relevant), 'mrr_at_5': 1/min(relevant) if relevant else 0,
            'annotated_page_recall_at_5': len(set(c['page'] for c in hits) & set(gold))/len(gold) if gold else None}


def mean(values):
    return float(np.mean(values)) if values else None


def wilson(success, n):
    if not n:
        return None
    z=1.96; p=success/n; divisor=1+z*z/n
    center=(p+z*z/(2*n))/divisor
    radius=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/divisor
    return [center-radius, center+radius]


def run(split='test', output_name=None, modes=('A','B','C','D'), retrieval='rewrite_hybrid'):
    questions = [r for r in read_jsonl(DATA/'benchmark.jsonl') if r['split']==split]
    directory = ROOT/'reports'/(output_name or split)
    directory.mkdir(parents=True, exist_ok=True)
    benchmark_hash = hashlib.sha256((DATA/'benchmark.jsonl').read_bytes()).hexdigest()
    code_hash = hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'chipclaw').glob('*.py')))).hexdigest()
    frozen = {'benchmark_sha256':benchmark_hash, 'code_sha256':code_hash, 'model':MODEL,
              'embedding':EMBEDDING, 'embedding_revision':EMBEDDING_REVISION, 'split':split, 'retrieval':retrieval, 'top_k':5, 'modes':list(modes),
              'temperature':0, 'seed':42, 'num_ctx':12288,
              'reranker':RERANKER,'reranker_revision':RERANKER_REVISION,'rerank_candidates':30,
              'corpus_sha256':hashlib.sha256((INDEX/'chunks.jsonl').read_bytes()).hexdigest()}
    meta_path = directory/'run.json'
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding='utf-8'))
        if existing['frozen'] != frozen:
            raise RuntimeError('Run configuration changed; choose a new --name, never mix runs')
    else:
        meta_path.write_text(json.dumps({'frozen':frozen, 'started_at':datetime.now(timezone.utc).isoformat(),
            'platform':platform.platform(), 'python':platform.python_version(),
            'ollama_models':requests.get(OLLAMA+'/api/tags',timeout=10).json(),
            'corpus_manifest':json.loads((INDEX/'manifest.json').read_text(encoding='utf-8'))},ensure_ascii=False,indent=2),encoding='utf-8')
    retriever=Retriever()
    rewrites_path=directory/'query_rewrites.json'
    if rewrites_path.exists():
        saved=json.loads(rewrites_path.read_text(encoding='utf-8'))
        retriever.rewrites=saved['queries'];retriever.rewrite_metrics=saved['metrics']
    for q in questions:
        retriever.translate(q['question'])
        rewrites_path.write_text(json.dumps({'queries':retriever.rewrites,'metrics':retriever.rewrite_metrics},ensure_ascii=False,indent=2),encoding='utf-8')
    ablations=[]
    for q in questions:
        if not q['answerable']:
            continue
        for method in ['bm25','dense','hybrid','rewrite_hybrid','rerank']:
            start=time.perf_counter()
            hits=retriever.retrieve(q['question'],q['doc_id'],method=method)
            ablations.append({'id':q['id'],'method':method,**retrieval_metrics(hits,q['gold_pages']),
                              'retrieval_ms':(time.perf_counter()-start)*1000,
                              'retrieved_ids':[h['id'] for h in hits]})
    (directory/'retrieval.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ablations),encoding='utf-8')
    records_path=directory/'predictions.jsonl'
    records=read_jsonl(records_path) if records_path.exists() else []
    done={(r['id'],r['mode']):r for r in records}
    for q in questions:
        start=time.perf_counter()
        hits=retriever.context_windows(retriever.retrieve(q['question'],q['doc_id'],method=retrieval))
        retrieval_ms=(time.perf_counter()-start)*1000 + retriever.rewrite_metrics[q['question']]['latency_ms']
        for mode in modes:
            if (q['id'],mode) in done:
                continue
            print(q['id'],mode,flush=True)
            if mode=='D':
                base=done.get((q['id'],'C'))
                result=verify(base['result'] if base else generate(q['question'],hits,'C'),hits)
            else:
                result=generate(q['question'],hits if mode!='A' else [],mode)
            score=rubric_pass(q,result['answer'],result['refused'])
            record={'id':q['id'],'mode':mode,'question':q['question'],'type':q['type'],
                    'answerable':q['answerable'],'reference':q['reference'],
                    'result':result,'rubric_pass':score, 'retrieval_ms':retrieval_ms if mode!='A' else 0,
                    'citation_valid':not validate_citations(result['output'],hits) if result['output'] and not result['refused'] else None,
                    'evidence':hits if mode!='A' else []}
            with records_path.open('a',encoding='utf-8') as f:
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
            records.append(record);done[(q['id'],mode)]=record
    summary={'split':split,'n_questions':len(questions),'rubric_warning':'Rule coverage proxy; NOT independent human accuracy or hallucination rate.',
             'retrieval':{},'generation':{}}
    for method in ['bm25','dense','hybrid','rewrite_hybrid','rerank']:
        rs=[r for r in ablations if r['method']==method]
        summary['retrieval'][method]={'n_answerable':len(rs),'hit_at_5':mean([r['hit_at_5'] for r in rs]),
                                     'mrr_at_5':mean([r['mrr_at_5'] for r in rs])}
    for mode in modes:
        rs=[r for r in records if r['mode']==mode]
        answered=[r for r in rs if not r['result']['refused']]
        answerable=[r for r in rs if r['answerable']]
        unanswerable=[r for r in rs if not r['answerable']]
        latencies=[r['result']['metrics']['latency_ms']+r['retrieval_ms'] for r in rs]
        valid=[r for r in answered if r['citation_valid'] is not None]
        passed=sum(r['rubric_pass'] for r in answerable)
        summary['generation'][mode]={'n':len(rs),'n_answerable':len(answerable),'n_unanswerable':len(unanswerable),
            'answerable_rubric_pass':mean([r['rubric_pass'] for r in answerable]),
            'answerable_rubric_wilson95':wilson(passed,len(answerable)),
            'unanswerable_refusal':mean([r['result']['refused'] for r in unanswerable]),
            'answerable_false_refusal':mean([r['result']['refused'] for r in answerable]),
            'answer_coverage':len(answered)/len(rs),
            'selective_rubric_pass':mean([r['rubric_pass'] for r in answered]),
            'provenance_valid_rate_among_answered':mean([r['citation_valid'] for r in valid]),
            'latency_p50_ms':float(np.percentile(latencies,50)), 'latency_p95_ms':float(np.percentile(latencies,95))}
    # Paired bootstrap measures sampling uncertainty only; not annotation bias.
    if 'A' in modes and 'C' in modes:
        delta=np.array([int(done[(q['id'],'C')]['rubric_pass'])-int(done[(q['id'],'A')]['rubric_pass']) for q in questions if q['answerable']])
        rng=np.random.default_rng(42)
        estimates=rng.choice(delta,(5000,len(delta)),replace=True).mean(axis=1)
        summary['paired_C_minus_A']={'rubric_pass_delta':float(delta.mean()),'bootstrap95':np.percentile(estimates,[2.5,97.5]).tolist()}
    (directory/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary
