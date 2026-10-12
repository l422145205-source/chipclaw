"""Opt-in v3 corpus, local cascade and source-page visual/text fusion."""
from .retrieval import Retriever,rrf
from .corpus_v3 import V3_INDEX,SOURCES_V3
from .query_rewrite import terminology_query

METHODS_V3={'student_v3','teacher_v3','cascade_v3','visual_fusion_v3'}

class V3Retriever(Retriever):
    def __init__(self,ranker=None,visual=None):
        super().__init__(dense=False,index_dir=V3_INDEX)
        self.ranker=ranker;self.visual=visual;self.v3_events=[]

    def lexical_candidates(self,question,doc_id):
        allowed=[i for i,c in enumerate(self.chunks) if c['doc_id']==doc_id]
        expanded,audit=terminology_query(question)
        views=list(dict.fromkeys(v for v in [question,expanded] if v))
        rankings=[self._rank_query(v,allowed,'bm25')[0] for v in views]
        rank,_=rrf(rankings)
        self.rewrite_audits[question]=audit
        return rank

    def retrieve(self,question,doc_id,method='cascade_v3',top_k=5,diversify=True):
        if doc_id not in SOURCES_V3:raise ValueError('Unknown v3 document')
        if method not in METHODS_V3:return super().retrieve(question,doc_id,method,top_k,diversify)
        lexical=self.lexical_candidates(question,doc_id)
        event={'method':method,'corpus':'v3','documents':20,'full_text_chunks':len(self.chunks),
               'query':question,'scope_note':'Runtime full-chunk BM25 candidates differ from source-row dev calibration; gate remains experimental.'}
        if method=='visual_fusion_v3':
            if self.visual is None:
                from .visual_v3 import VisualPageRetriever
                self.visual=VisualPageRetriever()
            images=self.visual.retrieve(question,doc_id,top_k=12)
            text_pages=list(dict.fromkeys((self.chunks[i]['doc_id'],self.chunks[i]['page']) for i in lexical))
            image_pages=[(p['doc_id'],p['page']) for p in images]
            pages,page_scores=rrf([text_pages,image_pages])
            rank=[]
            for key in pages:
                candidates=[i for i in lexical if (self.chunks[i]['doc_id'],self.chunks[i]['page'])==key]
                if not candidates:candidates=[i for i,c in enumerate(self.chunks) if (c['doc_id'],c['page'])==key]
                if candidates:rank.append(candidates[0])
            for p in images:
                if any(c['doc_id']==p['doc_id'] and c['page']==p['page'] and c['source_sha256']!=p['source_sha256'] for c in self.chunks):
                    raise ValueError('Visual/text source fingerprint mismatch')
            event.update(visual_scope=self.visual.last_metrics,visual_pages=image_pages,fusion='page RRF k=60; source text recovered for physical verification')
        else:
            if self.ranker is None:
                from .cascade_v3 import CascadeRanker
                self.ranker=CascadeRanker()
            candidates=lexical[:16]
            if not candidates:return []
            ids=[self.chunks[i]['id'] for i in candidates]
            texts=[f'{doc_id.upper()}; PDF page {self.chunks[i]["page"]}: '+self.chunks[i]['text'] for i in candidates]
            order=self.ranker.rank(question,ids,texts,method);rank=[candidates[i] for i in order]
            event.update(self.ranker.last_metrics)
        self.v3_events.append(event);self.retrieval_events.append(event)
        selected=[];seen=set()
        for i in rank:
            key=(self.chunks[i]['doc_id'],self.chunks[i]['page'])
            if diversify and key in seen:continue
            selected.append({**self.chunks[i],'score':1/(len(selected)+1),'v3_trace':event});seen.add(key)
            if len(selected)>=top_k:break
        return selected
