"""Reversible encoding of PDF layout whitespace; positions are metadata, not facts."""
import json
import re

VERSION = 'pypdf-layout-cell-offsets-v3'
CELLS = re.compile(r'\S(?:(?! {3,}).)*?(?= {3,}|$)')

def encode_layout(text):
    """Keep every non-padding character and every cell's original column."""
    records=[]
    for row,line in enumerate(text.split('\n')):
        if '\t' in line:
            raise ValueError('Layout codec expects expanded spaces, not tabs')
        cells=[[m.start(),m.group(0)] for m in CELLS.finditer(line)]
        records.append(json.dumps({'r':row,'c':cells},ensure_ascii=False,separators=(',',':')))
    encoded='\n'.join(records)
    if decode_layout(encoded)!=text:
        raise ValueError('Layout encoding did not round-trip exactly')
    return encoded

def decode_layout(encoded):
    lines=[]
    for line in encoded.splitlines():
        record=json.loads(line)
        value=''
        for column,text in record['c']:
            if type(column) is not int or column<len(value) or not isinstance(text,str):
                raise ValueError('Invalid or overlapping cell positions')
            value+=' '*(column-len(value))+text
        lines.append(value)
    return '\n'.join(lines)

def content_line(line):
    try:
        record=json.loads(line)
    except (ValueError,TypeError):
        return line
    if isinstance(record,dict) and 'c' in record:
        return '  '.join(cell[1] for cell in record['c'])
    return line

def content_text(text):
    return '\n'.join(content_line(line) for line in text.splitlines())

def source_rows(text):
    return [json.loads(line) for line in text.splitlines()]

def select_packets(pages, question, budget_chars, max_pages, idf, tokenize, stopwords):
    """Choose whole detected tables; never silently slice a table's value rows."""
    caption=re.compile(r'^\s*Table\s*(\d+(?:[-.]\d+)*)\b',re.I)
    header=re.compile(r'\b(symbol|parameter|conditions?|units?|min(?:imum)?|typ(?:ical)?|max(?:imum)?)\b',re.I)
    requested=set(re.findall(r'\bTable\s*(\d+(?:[-.]\d+)*)\b',question,re.I))
    terms=set(tokenize(question))-stopwords
    role_aliases={'max':'maximum','maximum':'maximum','min':'minimum','minimum':'minimum',
                  'typ':'typical','typical':'typical','default':'default','reset':'reset',
                  'working':'operating','operating':'operating'}
    requested_roles={role_aliases[t] for t in terms if t in role_aliases}
    terms.update(requested_roles)
    header_sets=[set(tokenize('\n'.join(content_text(p['text']).splitlines()[:3]))) for p in pages]
    common=set.intersection(*header_sets) if header_sets else set()
    terms-=common
    candidates=[]
    rejected=[]
    for ordinal,page in enumerate(pages[:max_pages]):
        rows=page['text'].splitlines(keepends=True)
        offsets=[0]
        for row in rows: offsets.append(offsets[-1]+len(row))
        plain=[content_line(row.rstrip('\n')) for row in rows]
        boundaries=[i for i,row in enumerate(plain) if caption.search(row)]
        table_like=any(len(header.findall(row))>=3 for row in plain)
        ranges=[]
        if boundaries:
            if boundaries[0]:
                ranges.append((0,boundaries[0],'prose',None))
            for i,start in enumerate(boundaries):
                end=boundaries[i+1] if i+1<len(boundaries) else len(rows)
                begin=0 if i==0 and offsets[start]<=800 else start
                ranges.append((begin,end,'table',caption.search(plain[start]).group(1)))
        elif table_like:
            ranges.append((0,len(rows),'table',None))
        elif len(page['text'])<=2400:
            ranges.append((0,len(rows),'prose',None))
        else:
            start=0
            while start<len(rows):
                end=start+1
                while end<len(rows) and offsets[end+1]-offsets[start]<=1400: end+=1
                ranges.append((start,end,'prose',None))
                if end==len(rows): break
                start=max(start+1,end-2)
        for start,end,kind,table in ranges:
            a,b=offsets[start],offsets[end]
            if a==b: continue
            if b-a>budget_chars:
                rejected.append({'doc_id':page['doc_id'],'page':page['page'],'table':table,
                                 'reason':'whole_packet_exceeds_source_budget','encoded_chars':b-a})
                continue
            body='\n'.join(plain[start:end])
            words=set(tokenize(body))
            captions=[row for row in plain[start:end] if caption.search(row)]
            title=captions[0] if kind=='table' and captions else ' '.join(plain[start:min(start+3,end)])
            candidates.append({'ordinal':ordinal,'start':a,'end':b,'kind':kind,'table':table,
                'words':words & terms,'title_words':set(tokenize(title)) & terms,
                'caption_roles':{role_aliases[t] for t in tokenize(title) if t in role_aliases},'page':page})
    chosen=[]
    covered=set()
    used=0
    remaining=list(candidates)
    while remaining:
        feasible=[p for p in remaining if used+p['end']-p['start']<=budget_chars and
                  not any(p['page']['doc_id']==q['page']['doc_id'] and p['page']['page']==q['page']['page'] and
                          p['start']<q['end'] and p['end']>q['start'] for q in chosen)]
        if not feasible: break
        def utility(packet):
            weight=lambda term:idf.get(term,1.0)
            marginal=sum(weight(t)*(0.25 if t in covered else 1) for t in packet['words'])
            title=sum(weight(t)*(2.0 if t not in covered else 0.15) for t in packet['title_words'])
            caption_roles=packet['caption_roles']
            if packet['kind']=='table' and requested_roles & caption_roles:
                # A requested extreme/default plus a newly covered caption
                # subject should outrank a cheaper curve sharing only keywords.
                title+=25*sum(weight(t) for t in packet['title_words']-covered)
                title+=20*len(requested_roles & caption_roles)
            explicit=100 if packet['table'] in requested else 0
            denominator=max(1,(packet['end']-packet['start'])/1400)**0.5
            return (marginal+title+explicit)/denominator/(1+0.12*packet['ordinal'])
        pick=max(feasible,key=lambda p:(p['table'] in requested,utility(p),-p['ordinal'],-p['start']))
        if utility(pick)<=0: break
        pick=dict(pick,utility=utility(pick))
        chosen.append(pick)
        used+=pick['end']-pick['start']
        covered|=pick['words']
        remaining.remove(next(p for p in remaining if p['ordinal']==pick['ordinal'] and p['start']==pick['start'] and p['end']==pick['end']))
    chosen.sort(key=lambda p:(p['ordinal'],p['start']))
    return chosen,rejected
