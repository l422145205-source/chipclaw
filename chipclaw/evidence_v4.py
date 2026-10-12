"""Real retrieval/source-budget/numeric verification without generation dependency."""
import time
from .packets_v4 import build_dependency_spans
from .fact_verifier import verify_claim
def collect_evidence(question,doc_id,retriever,loader,method='adaptive_v4',budget_chars=6000,max_pages=5,claims=()):
    if not isinstance(question,str) or not question.strip():raise ValueError('Nonempty question required')
    if type(max_pages) is not int or not 1<=max_pages<=5 or type(budget_chars) is not int or not 256<=budget_chars<=24000:raise ValueError('Invalid bounded evidence request')
    if len(claims)>8 or any(not isinstance(c,str) or not c.strip() or len(c)>400 for c in claims):raise ValueError('Bounded claim list required')
    started=time.perf_counter();hits=retriever.retrieve(question,doc_id,method,top_k=8);retrieval_ms=(time.perf_counter()-started)*1000
    pages=[];seen={}
    for hit in hits:
        if hit['doc_id']!=doc_id:raise ValueError('Cross-document evidence hit')
        key=(doc_id,hit['page'])
        if key in seen:
            if seen[key]!=hit['source_sha256']:raise ValueError('Conflicting source fingerprints')
            continue
        if len(pages)>=max_pages:break
        pages.append(loader.read_page(doc_id,hit['page'],hit['source_sha256']));seen[key]=hit['source_sha256']
    spans,budget=build_dependency_spans(pages,question,budget_chars,max_pages,getattr(retriever,'idf',{}))
    checks=[{'claim':c,'check':verify_claim(c,spans)} for c in claims]
    return {'query':question,'doc_id':doc_id,'retrieval_method':method,'evidence':spans,'budget':budget,'claims':checks,
        'retrieved_pages':list(dict.fromkeys(h['page'] for h in hits)),'retrieval_trace':retriever.v3_events[-1] if retriever.v3_events else None,
        'metrics':{'retrieval_ms':retrieval_ms,'total_ms':(time.perf_counter()-started)*1000,'generation_calls':0},
        'scope':'Original-PDF retrieval, closed source budget and deterministic numeric atoms; no generated answer or whole-answer correctness claim.'}
