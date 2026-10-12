"""Physical-table closure before budget selection. Original offsets remain exact."""
import math,re
from .layout_codec import content_line,source_rows
from .reliable import build_packet_spans
from .retrieval import tokens
from .query_rewrite import terminology_query

# Explicit source-condition aliases, independent of answer values or claims.
# Keep this local to v4 selection so frozen retrieval candidates do not change.
NO_LOAD_ALIASES=('空载','无负载','无负荷')
def selection_terms(question):
    expanded,_=terminology_query(question)
    aliases=[{'source':alias,'canonical':'no load'} for alias in NO_LOAD_ALIASES if alias in question]
    return set(tokens(question+' '+expanded+' '+' '.join(a['canonical'] for a in aliases))),aliases

def closed_table_spans(page):
    """Keep every physical row of one table plus local caption/condition/notes.

    This is local page closure, not a proof of cross-page footnote resolution.
    A contiguous range makes source offsets/citations independently checkable.
    """
    rows=page['text'].splitlines(keepends=True);offsets=[0]
    for row in rows:offsets.append(offsets[-1]+len(row))
    records=page.get('geometry_records') or [];groups={}
    for record in records:
        if record.get('kind')=='table' and record.get('table_index') is not None:
            groups.setdefault(record['table_index'],[]).append(record['r'])
    result=[];ordered=[]
    # Side-by-side/nested physical tables can interleave encoded row indices.
    # Merge intersecting row intervals before computing any contiguous closure.
    for table,ids in sorted(groups.items(),key=lambda item:min(item[1])):
        if ordered and min(ids)<=max(ordered[-1][1]):
            ordered[-1][0].append(table);ordered[-1][1].extend(ids)
        else:ordered.append(([table],list(ids)))
    for index,(tables,ids) in enumerate(ordered):
        table=tables[0]
        low,high=min(ids),max(ids)+1
        previous_end=max(ordered[index-1][1])+1 if index else 0
        next_begin=min(ordered[index+1][1]) if index+1<len(ordered) else len(rows)
        start=max(previous_end,low-4)
        # Footnotes and local conditions remain available. Stop before the next
        # section/figure heading; never stop within the physical table itself.
        end=high
        while end<next_begin:
            visible=content_line(rows[end].rstrip('\n'))
            if re.match(r'^\s*(?:\d+(?:\.\d+)*\s+[A-Z]|Figure\s+\d|Table\s+\d)',visible):break
            end+=1
        if start<0 or end>len(rows):raise ValueError('Physical table row outside cited page')
        a,b=offsets[start],offsets[end];rid={r['r'] for r in source_rows(page['text'][a:b])}
        if not set(ids)<=rid:raise AssertionError('Physical table row omitted from closure')
        span={k:page[k] for k in ['doc_id','page','source_sha256','url','title','extraction']}
        sid=f'{page["doc_id"]}:p{page["page"]}:span{a}-{b}:{page["source_sha256"][:12]}'
        span.update(id=sid,span_id=sid,start=a,end=b,text=page['text'][a:b],source_rows=source_rows(page['text'][a:b]),
            geometry_records=[r for r in records if r['r'] in rid],packet_kind='table',table_id=table,
            closure={'table_index':table,'table_indices':tables,'physical_rows':sorted(set(ids)),'first_context_row':start,'last_context_row':end-1,
                     'scope':'All local physical table rows and adjacent original context; cross-page notes unresolved.'})
        result.append(span)
    return result

def build_dependency_spans(pages,question,budget_chars,max_pages,idf):
    if type(budget_chars) is not int or budget_chars<256 or not 1<=max_pages<=12:raise ValueError('Invalid source budget')
    terms,aliases=selection_terms(question)
    stop={'what','is','the','of','how','does','with','for','under','at','to','and','in','a','an','typical','maximum','minimum','min','max','typ','device'}
    terms-=stop;numeric={t for t in terms if re.fullmatch(r'\d+(?:\.\d+)?',t)}
    candidates=[];rejected=[]
    for ordinal,page in enumerate(pages):
        local=closed_table_spans(page)
        legacy,trace=build_packet_spans([page],question,budget_chars,1,idf);rejected+=trace['rejected_packets']
        # Keep ordinary prose and the original complete-page fallback. The
        # physical alternative competes at its actual character cost.
        for span in local+legacy:
            if len(span['text'])>budget_chars:
                rejected.append({'span_id':span['id'],'reason':'closed_table_exceeds_budget','encoded_chars':len(span['text'])});continue
            words=set(tokens(' '.join(content_line(x) for x in span['text'].splitlines())))
            coverage=terms&words
            # Match subject/condition anchors rather than numeric answer values.
            utility=sum(idf.get(t,1.)*(.1 if t in numeric else 1.) for t in coverage)
            if span.get('closure'):utility*=1.25
            utility/=1+.12*ordinal
            candidates.append({'span':span,'ordinal':ordinal,'terms':coverage,'value':utility})
    unique={c['span']['id']:c for c in candidates};candidates=list(unique.values())
    def compatible(candidate,chosen):
        s=candidate['span']
        return all((s['doc_id'],s['page'])!=(x['span']['doc_id'],x['span']['page']) or s['end']<=x['span']['start'] or x['span']['end']<=s['start'] for x in chosen)
    def objective(chosen):
        covered=set().union(*(x['terms'] for x in chosen)) if chosen else set()
        return sum(idf.get(t,1.)*(.1 if t in numeric else 1.) for t in covered)+.12*sum(x['value'] for x in chosen)
    best=[];best_value=0;trials=[]
    # Empty start plus every feasible individual start prevents a cheap curve
    # packet monopolising a greedy path. Strict budgets apply to each trial.
    for seed in [None]+sorted(candidates,key=lambda c:(-c['value'],c['span']['id']))[:24]:
        chosen=[] if seed is None else [seed];spent=sum(len(c['span']['text']) for c in chosen)
        while True:
            feasible=[c for c in candidates if c not in chosen and spent+len(c['span']['text'])<=budget_chars and compatible(c,chosen)
                and len({(x['span']['doc_id'],x['span']['page']) for x in chosen+[c]})<=max_pages]
            if not feasible:break
            before=objective(chosen)
            pick=max(feasible,key=lambda c:((objective(chosen+[c])-before)/math.sqrt(max(1,len(c['span']['text'])/1400)),-c['ordinal'],c['span']['id']))
            if objective(chosen+[pick])<=before:break
            chosen.append(pick);spent+=len(pick['span']['text'])
        value=objective(chosen);trials.append({'seed':None if seed is None else seed['span']['id'],'objective':value,'chars':spent})
        if value>best_value or value==best_value and spent<sum(len(x['span']['text']) for x in best):best=chosen;best_value=value
    best.sort(key=lambda c:(c['ordinal'],c['span']['start']));spans=[c['span'] for c in best]
    used=sum(len(s['text']) for s in spans)
    assert used<=budget_chars and len({(s['doc_id'],s['page']) for s in spans})<=max_pages
    return spans,{'budget_chars':budget_chars,'max_pages':max_pages,'used_chars':used,'selected_pages':len({(s['doc_id'],s['page']) for s in spans}),
        'selection':'physical_table_dependency_closure_multistart','condition_aliases':aliases,'objective':best_value,'trials':trials,'rejected_packets':rejected,
        'pages':[{'doc_id':p['doc_id'],'page':p['page'],'layout_chars':len(p['text']),'selected_chars':sum(len(s['text']) for s in spans if s['doc_id']==p['doc_id'] and s['page']==p['page'])} for p in pages],
        'note':'Only original contiguous ranges are sent to generation/verifier. No gold pages, reference answers, or generated table cells enter selection.'}
