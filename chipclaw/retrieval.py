import hashlib
import json
import math
import re
from pathlib import Path
from collections import Counter
import numpy as np
from .config import INDEX, EMBEDDING, EMBEDDING_REVISION, RERANKER, RERANKER_REVISION
from .query_rewrite import (has_chinese, protect_query, restore_translation,
                            terminology_query, validate_cached_translation)

# Domain glossary, frozen before test runs; no reference answers or question IDs.
ALIASES = {'主频':'clock frequency', '频率':'frequency', '内核':'core processor',
           '容量':'memory size', '闪存':'flash', '温度':'temperature', '电压':'voltage',
           '通道':'channel', '分辨率':'resolution', '定时器':'timer', '寄存器':'register',
           '串口':'UART', '中断':'interrupt', '复位':'reset', '功耗':'power consumption',
           '休眠':'sleep', '蓝牙':'Bluetooth', '启动':'boot', '缓存':'cache',
           '看门狗':'watchdog', '总线':'bus', '双核':'dual core', '时钟':'clock'}


def tokens(text):
    value = text.lower()
    for zh, en in ALIASES.items():
        if zh in value:
            value += ' ' + en.lower()
    return re.findall(r'[a-z][a-z0-9_]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]', value)


def rrf(rankings, k=60):
    result = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, 1):
            result[idx] = result.get(idx, 0.0) + 1 / (k + rank)
    return sorted(result, key=lambda i: (-result[i], i)), result


def protect_view_candidates(rank, rankings, budget=30):
    """Reserve equal per-view candidate shares before a bounded English reranker.

    A good English expansion must not disappear merely because a weak Chinese
    branch disagrees. Selection uses ranks only, never reference evidence labels.
    RRF order is preserved within the admitted pool; this is candidate admission,
    not a learned relevance score.
    """
    if budget <= 0 or not rankings:
        raise ValueError('A positive candidate budget and query views are required')
    quota = budget // len(rankings)
    protected = {i for view in rankings for i in view[:quota]}
    admitted = set(protected)
    for i in rank:
        if len(admitted) >= budget:
            break
        admitted.add(i)
    ordered = [i for i in rank if i in admitted]
    return ordered + [i for i in rank if i not in admitted], {
        'type': 'equal_query_view_candidate_admission', 'budget': budget,
        'per_view_quota': quota, 'views': len(rankings), 'reserved_candidates': len(protected),
        'note': 'Each view top quota survives candidate admission; no source labels used.'}


class Retriever:
    def __init__(self, dense=True, index_dir=None):
        self.index_dir = Path(index_dir or INDEX)
        self.chunks = [json.loads(line) for line in (self.index_dir / 'chunks.jsonl').read_text(encoding='utf-8').splitlines()]
        self.by_id = {c['id']: c for c in self.chunks}
        self.counts = [Counter(tokens(c['text'])) for c in self.chunks]
        self.lengths = np.array([sum(c.values()) for c in self.counts])
        self.avg_len = max(float(self.lengths.mean()), 1)
        df = Counter(t for c in self.counts for t in c)
        n = len(self.chunks)
        self.idf = {t: math.log(1 + (n - f + .5) / (f + .5)) for t, f in df.items()}
        self.model = None
        self.vectors = None
        self.query_cache = {}
        self.rewrites = {}
        self.rewrite_metrics = {}
        self.rewrite_audits = {}
        self.retrieval_events = []
        self.reranker = None
        self.domain_reranker = None
        self.domain_rerank_events = []
        if dense:
            self.load_dense()

    def load_dense(self):
        if self.model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(EMBEDDING, revision=EMBEDDING_REVISION, device='cpu')
        self.model.max_seq_length = 512
        corpus_hash = hashlib.sha256((self.index_dir / 'chunks.jsonl').read_bytes()).hexdigest()
        signature = {'corpus_sha256': corpus_hash, 'embedding': EMBEDDING,
                     'max_seq_length': 512, 'normalize': True, 'prefix': 'passage: '}
        meta, vector_path = self.index_dir / 'vectors.meta.json', self.index_dir / 'vectors.npy'
        if meta.exists() and vector_path.exists() and json.loads(meta.read_text()) == signature:
            self.vectors = np.load(vector_path, allow_pickle=False)
        else:
            self.vectors = self.model.encode(['passage: ' + c['text'] for c in self.chunks],
                                            batch_size=32, normalize_embeddings=True, show_progress_bar=True)
            np.save(vector_path, self.vectors)
            meta.write_text(json.dumps(signature, indent=2), encoding='utf-8')

    def translate(self, question):
        # Existing unmasked caches are checked rather than silently trusted.
        if question in self.rewrites and question not in self.rewrite_audits:
            if not has_chinese(question) or validate_cached_translation(question, self.rewrites[question]):
                self.rewrite_audits[question] = {'method': 'validated_legacy_cache',
                    'semantic_equivalence_proven': False}
            else:
                self.rewrites.pop(question)
        if question not in self.rewrites and not has_chinese(question):
            self.rewrites[question] = question
            self.rewrite_metrics[question] = {'latency_ms': 0, 'prompt_tokens': 0, 'output_tokens': 0,
                                            'model': None, 'done_reason': 'english_query_passthrough'}
            self.rewrite_audits[question] = {'method': 'english_query_passthrough'}
        if question not in self.rewrites:
            from .generation import call_model
            protection = protect_query(question)
            translated, metrics = call_model([
                {'role':'system','content':('Translate the Chinese query into English retrieval keywords only. '
                    'Never answer it or add facts. Preserve every CCP0000X-style placeholder exactly once '
                    'and in its original order; do not interpret placeholders. '
                    'Do not introduce numbers. Return JSON with a query field.')},
                {'role':'user','content':protection['masked']}],
                schema={'type':'object','properties':{'query':{'type':'string','pattern':'^[ -~]+$'}},
                        'required':['query'],'additionalProperties':False}, max_tokens=220)
            try:
                english = restore_translation(protection, json.loads(translated)['query'])
                audit = {'method': 'protected_llm_translation', 'version': protection['version'],
                         'protected_atoms': protection['atoms'], 'glossary_sha256': protection['glossary_sha256'],
                         'protection_passed': True, 'semantic_equivalence_proven': False}
            except (ValueError, KeyError, TypeError) as error:
                english, audit = terminology_query(question)
                audit.update({'protection_passed': False, 'rejected_translation': translated,
                              'rejection_reason': str(error)})
            if not english:
                raise RuntimeError('No English-compatible query anchors; retain Chinese retrieval instead')
            self.rewrites[question] = english
            self.rewrite_audits[question] = audit
            self.rewrite_metrics[question] = metrics
        return self.rewrites[question]

    def context_windows(self, hits):
        """Retrieve on small chunks; give adjacent evidence to the generator.

        Source IDs identify the original pieces; :window is a separate context ID.
        The original passages are joined without paraphrasing.
        """
        result=[]
        seen=set()
        for hit in hits:
            if (hit['doc_id'], hit['page']) in seen:
                continue
            seen.add((hit['doc_id'], hit['page']))
            siblings=[c for c in self.chunks if c['doc_id']==hit['doc_id'] and c['page']==hit['page']]
            pos=next(i for i,c in enumerate(siblings) if c['id']==hit['id'])
            window=siblings[max(0,pos-1):pos+2]
            result.append({**hit,'id':hit['id']+':window','retrieval_id':hit['id'],
                           'source_ids':[c['id'] for c in window],
                           'text':'\n\n'.join(c['text'] for c in window),
                           'start':window[0]['start'],'end':window[-1]['end']})
        return result

    def _rank_query(self, question, allowed, method):
        """One lexical/dense query view; English historical ranking stays unchanged."""
        lexical = np.zeros(len(self.chunks))
        for t in set(tokens(question)):
            idf = self.idf.get(t, 0)
            if not idf:
                continue
            for i in allowed:
                tf = self.counts[i].get(t, 0)
                lexical[i] += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * self.lengths[i] / self.avg_len))
        sparse_rank = sorted(allowed, key=lambda i: (-lexical[i], i))[:50]
        sparse_rank = [i for i in sparse_rank if lexical[i] > 0]
        dense_scores = None
        if method in {'dense', 'hybrid'}:
            self.load_dense()
            if question not in self.query_cache:
                self.query_cache[question] = self.model.encode(['query: ' + question], normalize_embeddings=True)[0]
            dense_scores = self.vectors @ self.query_cache[question]
            dense_rank = sorted(allowed, key=lambda i: (-float(dense_scores[i]), i))[:50]
        if method == 'bm25':
            rank, scores = sparse_rank, dict(enumerate(lexical))
        elif method == 'dense':
            rank, scores = dense_rank, dict(enumerate(dense_scores))
        else:
            rank, scores = rrf([sparse_rank, dense_rank])
        return rank, scores

    def retrieve(self, question, doc_id, method='rewrite_hybrid', top_k=5, diversify=True):
        if method not in {'bm25', 'dense', 'hybrid', 'rewrite_hybrid', 'protected_hybrid', 'rerank', 'domain_rerank'}:
            raise ValueError('Unknown retrieval method')
        allowed = [i for i, c in enumerate(self.chunks) if c['doc_id'] == doc_id]
        if not allowed:
            return []
        original_question, original_method = question, method
        domain = method == 'domain_rerank'
        rerank = method == 'rerank' or domain
        views = [question]
        if method == 'protected_hybrid':
            # Offline route for diagnostics / deterministic recall. No model translation.
            english, audit = terminology_query(question)
            views = list(dict.fromkeys([question, english])) if english else [question]
            self.rewrite_audits[question] = audit
            if english:
                self.rewrites[question] = english
            self.rewrite_metrics[question] = {'latency_ms': 0, 'prompt_tokens': 0, 'output_tokens': 0,
                'model': None, 'done_reason': 'hand_authored_terminology_expansion', 'llm_calls': 0}
            method = 'hybrid'
        elif method in {'rewrite_hybrid', 'rerank', 'domain_rerank'}:
            if not domain or has_chinese(question):
                question = self.translate(question)
            if has_chinese(original_question):
                terms, _ = terminology_query(original_question)
                views = list(dict.fromkeys(v for v in [original_question, question, terms] if v))
            else:
                views = [question]
            method = 'hybrid'
        ranked_views = [self._rank_query(view, allowed, method) for view in views]
        if len(ranked_views) == 1:
            rank, scores = ranked_views[0]
            candidate_policy = {'type': 'single_query_view', 'budget': 30}
        else:
            rank, scores = rrf([item[0] for item in ranked_views])
            rank, candidate_policy = protect_view_candidates(rank, [item[0] for item in ranked_views])
        event = {'original_query': original_question, 'method': original_method, 'query_views': views,
                 'rewrite_audit': self.rewrite_audits.get(original_question),
                 'candidate_policy': candidate_policy,
                 'candidate_ids': [self.chunks[i]['id'] for i in rank[:30]]}
        self.retrieval_events.append(event)
        if rerank:
            if domain:
                if self.domain_reranker is None:
                    from .domain_reranker import DomainReranker
                    self.domain_reranker=DomainReranker()
                scorer=self.domain_reranker
            else:
                if self.reranker is None:
                    from sentence_transformers import CrossEncoder
                    self.reranker=CrossEncoder(RERANKER,revision=RERANKER_REVISION,device='cpu',max_length=512)
                scorer=self.reranker
            candidates=rank[:30]
            logits=scorer.predict([(question,self.chunks[i]['text']) for i in candidates],batch_size=8 if domain else 16,show_progress_bar=False)
            if domain:
                self.domain_rerank_events.append(dict(scorer.last_metrics, query=question,
                    original_query=original_question, query_views=views, rewrite_audit=event['rewrite_audit'],
                    candidate_policy=candidate_policy))
            scores={i:float(s) for i,s in zip(candidates,logits)}
            rank=sorted(candidates,key=lambda i:(-scores[i],i))
        selected, page_counts = [], Counter()
        for i in rank:
            c = self.chunks[i]
            if diversify and page_counts[c['page']] >= 2:
                continue
            selected.append({**c, 'score': float(scores[i])})
            page_counts[c['page']] += 1
            if len(selected) == top_k:
                break
        return selected
