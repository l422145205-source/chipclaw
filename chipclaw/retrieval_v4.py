"""Opt-in bilingual/cost-aware ranking; unchanged twenty-device corpus."""
from .retrieval_v3 import V3Retriever
from .corpus_v3 import SOURCES_V3
METHODS_V4={'multilingual_v4','adaptive_v4','domain_teacher_v4','visual_fusion_v4'}
def candidate_text(chunk):return f'{chunk["doc_id"].upper()}; PDF page {chunk["page"]}: '+chunk['text']
class V4Retriever(V3Retriever):
    def __init__(self,ranker=None,visual=None):
        super().__init__(ranker=None,visual=visual)
        self.v4_ranker=ranker
    def retrieve(self,question,doc_id,method='adaptive_v4',top_k=5,diversify=True):
        if type(top_k) is not int or top_k<1 or top_k>32:raise ValueError('Bounded top_k required')
        if method=='visual_fusion_v4':
            result=super().retrieve(question,doc_id,'visual_fusion_v3',top_k,diversify)
            if self.v3_events:self.v3_events[-1].update(method=method,ranking_backend='visual_fusion_v3')
            return result
        if method not in METHODS_V4:return super().retrieve(question,doc_id,method,top_k,diversify)
        if doc_id not in SOURCES_V3:raise ValueError('Unknown v4 document')
        lexical=self.lexical_candidates(question,doc_id);candidates=lexical[:16]
        if not candidates:return []
        if self.v4_ranker is None:
            from .cascade_v4 import AdaptiveRanker
            self.v4_ranker=AdaptiveRanker()
        chunks=[self.chunks[i] for i in candidates];order=self.v4_ranker.rank(question,[c['id'] for c in chunks],[candidate_text(c) for c in chunks],[c['page'] for c in chunks],method)
        event={**self.v4_ranker.last_metrics,'method':method,'query':question,'corpus':'v3','documents':20,'full_text_chunks':len(self.chunks)}
        self.v3_events.append(event);self.retrieval_events.append(event)
        result=[];seen=set()
        for position in order:
            chunk=chunks[position];key=(doc_id,chunk['page'])
            if diversify and key in seen:continue
            result.append({**chunk,'score':1/(len(result)+1),'v3_trace':event});seen.add(key)
            if len(result)>=top_k:break
        return result
