"""Document evidence graph from PDF bookmarks and literal cross references.

This is a graph-guided retrieval experiment, not Microsoft's GraphRAG package.
Edges describe document structure, never unverified semantic dependencies.
"""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from pypdf import PdfReader
from .config import DATA, INDEX
from .retrieval import tokens, rrf

GRAPH_PATH=INDEX/'evidence_graph.json'


def anchor_key(title):
    match=re.match(r'(?i)^(table|figure)\s+(\d+)\b',title)
    if match:return match[1].lower()+':'+match[2]
    match=re.match(r'^(\d+(?:\.\d+)*)\.?\s+',title)
    if match:return 'section:'+match[1]
    match=re.match(r'(?i)^Chapter\s+(\d+)\b',title)
    if match:return 'section:'+match[1]
    return None


def literal_references(text):
    return [(m[1].lower().rstrip('s')+':'+m[2],m.start(),m.end())
            for m in re.finditer(r'\b(Sections?|Tables?|Figures?)\s+(\d+(?:\.\d+)*)\b',text,re.I)]


def build_graph(chunks,raw_dir=None,index_dir=None):
    raw_dir=Path(raw_dir or DATA/'raw')
    target_index=Path(index_dir or INDEX)
    anchors={};edges=[]
    for doc_id in sorted({c['doc_id'] for c in chunks}):
        reader=PdfReader(raw_dir/(doc_id+'.pdf'))
        outlines=[]
        def walk(entries):
            for entry in entries:
                if isinstance(entry,list):walk(entry)
                else:
                    key=anchor_key(entry.title)
                    if key:
                        page=reader.get_destination_page_number(entry)+1
                        outlines.append((key,entry.title,page))
        walk(reader.outline)
        sections=[(key,title,page) for key,title,page in outlines if key.startswith('section:')]
        for key,title,page in outlines:
            node_id=doc_id+':'+key
            if node_id in anchors:
                anchors[node_id]['pages']=sorted(set(anchors[node_id]['pages']+[page]))
                continue
            end=page
            if key.startswith('section:'):
                level=key.count('.')
                position=next(i for i,x in enumerate(sections) if x==(key,title,page))
                later=next((p for k,t,p in sections[position+1:] if k.count('.')<=level and p>page),len(reader.pages)+1)
                end=later-1
            anchors[node_id]={'id':node_id,'doc_id':doc_id,'key':key,'title':title,'pages':[page],
                              'start_page':page,'end_page':end,'provenance':'native_pdf_bookmark'}
    for chunk in chunks:
        for key,start,end in literal_references(chunk['text']):
            target=chunk['doc_id']+':'+key
            if target in anchors:
                edges.append({'source':chunk['id'],'target':target,'relation':'explicit_reference',
                              'quote':chunk['text'][start:end],'start':start,'end':end,
                              'page':chunk['page'],'doc_id':chunk['doc_id']})
    manifest={'version':1,'corpus_sha256':hashlib.sha256((target_index/'chunks.jsonl').read_bytes()).hexdigest(),
              'construction':'PDF bookmarks + literal Section/Table/Figure references; no LLM edges',
              'n_anchors':len(anchors),'n_reference_edges':len(edges),
              'raw_pdf_sha256':{doc:hashlib.sha256((raw_dir/(doc+'.pdf')).read_bytes()).hexdigest() for doc in sorted({c['doc_id'] for c in chunks})}}
    graph={'manifest':manifest,'anchors':anchors,'reference_edges':edges}
    target_path=target_index/'evidence_graph.json' if index_dir is not None else GRAPH_PATH
    target_path.write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding='utf-8')
    return graph


class GraphRetriever:
    def __init__(self,retriever,graph=None):
        self.retriever=retriever
        self.graph=graph or json.loads(GRAPH_PATH.read_text(encoding='utf-8'))
        digest=hashlib.sha256((INDEX/'chunks.jsonl').read_bytes()).hexdigest()
        if self.graph['manifest']['corpus_sha256']!=digest:
            raise ValueError('Evidence graph belongs to a different corpus; rebuild it')
        self.anchors=self.graph['anchors']
        self.references={}
        for edge in self.graph['reference_edges']:
            self.references.setdefault(edge['source'],[]).append(edge)

    def retrieve(self,question,doc_id,method='graph',top_k=5):
        if method not in {'baseline','structure','graph','protected_graph'}:raise ValueError('Unknown graph ablation')
        r=self.retriever
        rewritten=r.translate(question)
        baseline=r.retrieve(question,doc_id,method='rewrite_hybrid',top_k=30,diversify=False)
        if method=='baseline':return self._select(baseline,top_k),{'routes':[],'reference_edges':[]}
        query=set(tokens(rewritten))
        # Routing depends only on query and native document headings.
        scored=[]
        for node in self.anchors.values():
            if node['doc_id']!=doc_id:continue
            words=set(tokens(node['title']))
            score=sum(r.idf.get(t,0) for t in query&words)/math.sqrt(max(len(words),1))
            if score>0:scored.append((score,node['id']))
        route_ids=[node for _,node in sorted(scored,key=lambda x:(-x[0],x[1]))[:3]]
        traversed=[]
        if method in {'graph','protected_graph'}:
            for hit in baseline[:5]:
                for edge in self.references.get(hit['id'],[]):
                    if edge['target'] not in route_ids:route_ids.append(edge['target'])
                    traversed.append(edge)
        route_ids=route_ids[:8]
        candidates={}
        for route_id in route_ids:
            node=self.anchors[route_id]
            # Section containment is page based; same-page sibling ambiguity is
            # retained and documented, not presented as a semantic relation.
            for i,c in enumerate(r.chunks):
                if c['doc_id']!=doc_id:continue
                included=(node['start_page']<=c['page']<=node['end_page']) if node['key'].startswith('section:') else c['page'] in node['pages']
                if included:candidates.setdefault(i,[]).append(route_id)
        # BM25 scoring inside graph-reached evidence, with global IDF.
        graph_scores={}
        for i in candidates:
            value=0
            for term in query:
                tf=r.counts[i].get(term,0)
                if tf:value+=r.idf.get(term,0)*tf*2.5/(tf+1.5*(.25+.75*r.lengths[i]/r.avg_len))
            if value>0:graph_scores[i]=value
        graph_rank=sorted(graph_scores,key=lambda i:(-graph_scores[i],i))[:30]
        index_by_id={c['id']:i for i,c in enumerate(r.chunks)}
        baseline_rank=[index_by_id[h['id']] for h in baseline]
        rank,scores=rrf([baseline_rank,graph_rank])
        hits=[{**r.chunks[i],'score':scores[i],'graph_routes':candidates.get(i,[])} for i in rank]
        if method=='protected_graph':
            # Retain five baseline pages, then spend the remaining evidence
            # budget on graph-reached candidates. Compare against baseline
            # at the SAME total top_k; this is a post-hoc diagnostic variant.
            protected=self._select(baseline,min(5,top_k))
            protected_ids={h['id'] for h in protected}
            extra=[h for h in hits if h.get('graph_routes') and h['id'] not in protected_ids]
            hits=protected+extra+baseline
        return self._select(hits,top_k),{'routes':[self.anchors[k] for k in route_ids],
                                      'reference_edges':traversed,'graph_candidate_count':len(candidates)}

    @staticmethod
    def _select(hits,top_k):
        result=[];seen=set()
        for hit in hits:
            key=(hit['doc_id'],hit['page'])
            if key in seen:continue
            seen.add(key);result.append(hit)
            if len(result)==top_k:break
        return result
