import threading
import logging
import json
import time
import os
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator
from .config import ROOT, SOURCES, HOLDOUT_SOURCES, HOLDOUT_INDEX, EXPANSION_SOURCES, EXPANDED_INDEX, DOCUMENT_SOURCES, MODEL
from .retrieval import Retriever
from .generation import generate, verify
from .repair import verify_with_repair
from .planning import EvidencePlanner, answer_requirements
from .reliable import ReliableAnswerer
from .orchestration import StructuredEvidenceAnswerer
from .corpus_v3 import SOURCES_V3

retriever=None
holdout_retriever=None
expanded_retriever=None
expanded_reliable=None
expanded_orchestration=None
v3_answerers={}
v3_retrievers={}
planners={}
lock=threading.Lock()
logger=logging.getLogger(__name__)

def initialize_legacy_retriever():
    global retriever
    retriever=Retriever()
    planners['main']=EvidencePlanner(retriever)

@asynccontextmanager
async def lifespan(app):
    global retriever
    warmup=os.environ.get('CHIPCLAW_API_WARMUP','legacy')
    if warmup not in {'legacy','none'}:raise ValueError('CHIPCLAW_API_WARMUP must be legacy or none')
    if warmup=='legacy':initialize_legacy_retriever()
    yield

app=FastAPI(title='EvidencePlan · ChipClaw',version='0.9.0',lifespan=lifespan)

from .workbench import install_workbench
install_workbench(app, live=True)

class Question(BaseModel):
    question:str=Field(min_length=2,max_length=1800)
    doc_id:str
    corpus:Literal['v2','v3']='v2'
    mode:Literal['B','C','D','E','F','G','H']='G'
    retrieval_method:Literal['rewrite_hybrid','domain_rerank','protected_hybrid','student_v3','teacher_v3','cascade_v3','visual_fusion_v3','multilingual_v4','adaptive_v4','domain_teacher_v4','visual_fusion_v4']='rewrite_hybrid'

    @model_validator(mode='after')
    def check_domain_path(self):
        from .corpus_v3 import SOURCES_V3
        from .retrieval_v3 import METHODS_V3
        from .retrieval_v4 import METHODS_V4
        catalog=SOURCES_V3 if self.corpus=='v3' else DOCUMENT_SOURCES
        if self.doc_id not in catalog:raise ValueError('Unknown document for selected corpus')
        if self.corpus=='v3' and self.mode not in {'G','H'}:raise ValueError('v3 corpus is implemented for G/H only')
        if self.retrieval_method in METHODS_V3|METHODS_V4 and (self.corpus!='v3' or self.mode not in {'G','H'}):
            raise ValueError('v3 retrieval requires corpus=v3 and G/H')
        if self.retrieval_method=='domain_rerank' and self.mode not in {'G','H'}:
            raise ValueError('Domain reranking is implemented for G/H only')
        if self.retrieval_method=='protected_hybrid' and self.mode not in {'G','H'}:
            raise ValueError('Protected hybrid retrieval is implemented for G/H only')
        return self

class EvidenceQuestion(Question):
    corpus:Literal['v3']='v3'
    mode:Literal['G']='G'
    retrieval_method:Literal['multilingual_v4','adaptive_v4','domain_teacher_v4','visual_fusion_v4']='adaptive_v4'
    budget_chars:int=Field(default=6000,ge=256,le=24000)
    max_pages:int=Field(default=5,ge=1,le=5)
    claims:list[str]=Field(default_factory=list,max_length=8)
    @model_validator(mode='after')
    def bounded_claims(self):
        if any(not c.strip() or len(c)>400 for c in self.claims):raise ValueError('Claims must be nonempty and at most 400 characters')
        return self

@app.post('/evidence')
def evidence(q:EvidenceQuestion):
    if not lock.acquire(blocking=False):raise HTTPException(429,'Local model busy')
    try:
        from .retrieval_v4 import V4Retriever
        from .evidence_v4 import collect_evidence
        from .reliable import SourceLoader
        from .corpus_v3 import V3_RAW
        from .geometry_v3 import V3GeometryLoader
        if 'v4' not in v3_retrievers:v3_retrievers['v4']=V4Retriever()
        return collect_evidence(q.question,q.doc_id,v3_retrievers['v4'],SourceLoader(V3_RAW,sources=SOURCES_V3,geometry_loader=V3GeometryLoader()),
            q.retrieval_method,q.budget_chars,q.max_pages,q.claims)
    finally:lock.release()

@app.get('/')
def home():return FileResponse(ROOT/'web'/'index.html')

@app.get('/health')
def health():return {'ready':retriever is not None or bool(v3_retrievers),'documents':list(DOCUMENT_SOURCES),'model':MODEL,
                     'v4_ready':'v4' in v3_retrievers,'startup_warmup':os.environ.get('CHIPCLAW_API_WARMUP','legacy'),
                     'v3_documents':list(SOURCES_V3),
                     'version':app.version,'source_role_guard':True,'evidence_planning':True,
                     'research_selector':'protected','default_mode':'G','default_condition_guard':True,
                     'reliable_answering':True,'layout_preservation':True,'structured_orchestration':True,
                     'independent_numeric_verification':True,'protected_query_rewriting':True}

@app.get('/regression')
def regression():
    for name in ['evidence-challenge-8b','evidence-challenge-v2']:
        path=ROOT/'reports'/name/'predictions.jsonl'
        if path.exists():
            rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
            return {'run':name,'rows':[{k:r[k] for k in ['category','output','expected_supported','provenance_accept','typed_guard_accept','full_verifier_accept']} for r in rows]}
    return {'run':None,'rows':[]}


@app.get('/demonstration/{case_id}')
def demonstration(case_id:Literal['current_modes','spi_timing','missing_measurement']):
    path=ROOT/'reports'/'structured-orchestration-v2'/(case_id+'.json')
    if not path.exists():
        raise HTTPException(404,'当前源码包中没有该演示记录。')
    record=json.loads(path.read_text(encoding='utf-8'))
    return {'run':'structured-orchestration-v2','recorded':True,'input':record['input'],
            'result':record['result'],'note':'已保存的真实运行，当前请求不调用模型。'}

@app.post('/ask')
def ask(q:Question):
    global holdout_retriever,expanded_retriever,expanded_reliable,expanded_orchestration
    if not lock.acquire(blocking=False):
        raise HTTPException(429,'本地模型正忙，请稍后重试。')
    try:
        start=time.perf_counter()
        if q.corpus=='v2' and retriever is None:
            # Evidence-only servers do not need legacy dense model startup.
            # A later v2 request still initializes its genuine backend.
            initialize_legacy_retriever()
        current=retriever
        corpus='main'
        if q.mode not in {'G','H'} and q.doc_id in HOLDOUT_SOURCES:
            if holdout_retriever is None:
                holdout_retriever=Retriever(index_dir=HOLDOUT_INDEX)
                planners['holdout']=EvidencePlanner(holdout_retriever)
            current,corpus=holdout_retriever,'holdout'
        elif q.mode not in {'G','H'} and q.doc_id in EXPANSION_SOURCES:
            if expanded_retriever is None:
                expanded_retriever=Retriever(index_dir=EXPANDED_INDEX)
                planners['expanded']=EvidencePlanner(expanded_retriever)
            current,corpus=expanded_retriever,'expanded'
        if q.corpus=='v3':
            from .corpus_v3 import V3_RAW,SOURCES_V3
            from .retrieval_v3 import V3Retriever
            from .reliable import SourceLoader
            from .geometry_v3 import V3GeometryLoader
            from .retrieval_v4 import V4Retriever,METHODS_V4
            init_start=time.perf_counter()
            v4=q.retrieval_method in METHODS_V4
            key=(q.mode,'v4') if v4 else q.mode
            if key not in v3_answerers:
                kwargs={}
                backend='v4' if v4 else 'v3'
                if backend not in v3_retrievers:
                    v3_retrievers[backend]=(V4Retriever if v4 else V3Retriever)()
                if v4:
                    from .packets_v4 import build_dependency_spans
                    kwargs['packet_builder']=build_dependency_spans
                v3_answerers[key]=(StructuredEvidenceAnswerer if q.mode=='H' else ReliableAnswerer)(v3_retrievers[backend],source_loader=SourceLoader(V3_RAW,sources=SOURCES_V3,geometry_loader=V3GeometryLoader()),**kwargs)
            init_ms=(time.perf_counter()-init_start)*1000
            result=v3_answerers[key].answer(q.question,q.doc_id,retrieval_method=q.retrieval_method)
            result['metrics']['retriever_init_ms']=init_ms
            result['corpus']='v3';hits=result['evidence']
        elif q.mode=='H':
            init_start=time.perf_counter()
            if expanded_orchestration is None:
                expanded_orchestration=StructuredEvidenceAnswerer(Retriever(index_dir=EXPANDED_INDEX))
            init_ms=(time.perf_counter()-init_start)*1000
            result=expanded_orchestration.answer(q.question,q.doc_id,**({'retrieval_method':q.retrieval_method} if q.retrieval_method!='rewrite_hybrid' else {}))
            result['metrics']['retriever_init_ms']=init_ms
            hits=result['evidence']
        elif q.mode=='G':
            init_start=time.perf_counter()
            if expanded_reliable is None:
                expanded_reliable=ReliableAnswerer(Retriever(index_dir=EXPANDED_INDEX))
            init_ms=(time.perf_counter()-init_start)*1000
            result=expanded_reliable.answer(q.question,q.doc_id,**({'retrieval_method':q.retrieval_method} if q.retrieval_method!='rewrite_hybrid' else {}))
            result['metrics']['retriever_init_ms']=init_ms
            hits=result['evidence']
        elif q.mode=='F':
            hits,trace=planners[corpus].retrieve(q.question,q.doc_id,method='protected')
            result=answer_requirements(q.question,hits,trace)
        else:
            hits=current.context_windows(current.retrieve(q.question,q.doc_id))
            result=generate(q.question,hits,'C' if q.mode in {'D','E'} else q.mode)
            if q.mode=='D':result=verify(result,hits)
            if q.mode=='E':result=verify_with_repair(result,hits)
        if not result['refused'] and not result['answer'].strip():
            raise RuntimeError('Invalid empty model answer')
        result['evidence']=hits
        result['metrics']['request_ms']=(time.perf_counter()-start)*1000
        return result
    except Exception as e:
        logger.exception('Local inference failed in mode %s for document %s', q.mode, q.doc_id)
        raise HTTPException(503,'推理未完成，请检查本地模型服务和日志。') from e
    finally:
        lock.release()
