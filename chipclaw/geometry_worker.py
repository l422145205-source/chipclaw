"""PDF geometry worker. No model, question or answer label is consumed."""
import argparse
import hashlib
import json
import re
from pathlib import Path

VERSION='pdfplumber-physical-cells-v1'

def normalize(value):
    return ' '.join((value or '').split())

def words_text(words):
    lines=[]
    for word in sorted(words,key=lambda w:(w['top'],w['x0'])):
        if not lines or word['top']-lines[-1][0]>5:
            lines.append((word['top'],[word]))
        else:lines[-1][1].append(word)
    return normalize(' '.join(' '.join(w['text'] for w in sorted(line,key=lambda w:w['x0'])) for _,line in lines))

def resolve_table(table, words):
    """Resolve vertical merged cells by their rectangles, not forward fill."""
    boxes=list(table.cells)
    xs=sorted({b[i] for b in boxes for i in (0,2)})
    ys=sorted({b[i] for b in boxes for i in (1,3)})
    columns=[(a,b) for a,b in zip(xs,xs[1:]) if b-a>=8]
    rows=[(a,b) for a,b in zip(ys,ys[1:]) if b-a>=2]
    cell_words={}
    for i,box in enumerate(boxes):
        x0,y0,x1,y1=box
        cell_words[i]=[w for w in words if x0-.1<=(w['x0']+w['x1'])/2<=x1+.1 and
                      y0-.1<=(w['top']+w['bottom'])/2<=y1+.1]
    matrix=[]
    provenance=[]
    for y0,y1 in rows:
        values=[]
        refs=[]
        y=(y0+y1)/2
        for x0,x1 in columns:
            x=(x0+x1)/2
            containing=[(i,b) for i,b in enumerate(boxes) if b[0]-.1<=x<=b[2]+.1 and b[1]-.1<=y<=b[3]+.1]
            if not containing:
                values.append('')
                refs.append(None)
                continue
            i,box=min(containing,key=lambda item:(item[1][2]-item[1][0])*(item[1][3]-item[1][1]))
            selected=cell_words[i]
            spanning=[(a,b) for a,b in columns if box[0]-.1<=(a+b)/2<=box[2]+.1]
            numeric=sum(bool(re.match(r'^[+−–-]?\d',w['text'])) for w in selected)
            # Missing vertical borders sometimes combine MIN and MAX. Split
            # distinct numeric words by physical column, rather than duplicating.
            if len(spanning)>1 and numeric>=2:
                selected=[w for w in selected if x0-.1<=(w['x0']+w['x1'])/2<=x1+.1]
            value=words_text(selected)
            values.append(value)
            refs.append({'bbox':list(box),'text':value,'word_boxes':[
                [w['x0'],w['top'],w['x1'],w['bottom'],w['text']] for w in selected]})
        matrix.append(values)
        provenance.append(refs)
    keep=[]
    for col in range(len(columns)):
        vector=[row[col] for row in matrix]
        if not any(vector):continue
        if keep and vector==[row[keep[-1]] for row in matrix]:continue
        keep.append(col)
    return {'bbox':list(table.bbox),'columns':[columns[c] for c in keep],'rows':rows,
            'matrix':[[row[c] for c in keep] for row in matrix],
            'provenance':[[row[c] for c in keep] for row in provenance]}

def extract_page(page):
    words=page.extract_words(x_tolerance=2,y_tolerance=3)
    tables=[]
    for table in page.find_tables():
        result=resolve_table(table,words)
        if len(result['columns'])>=2 and len(result['matrix'])>=2:
            tables.append(result)
    def in_table(word):
        x=(word['x0']+word['x1'])/2
        y=(word['top']+word['bottom'])/2
        return any(t['bbox'][0]<=x<=t['bbox'][2] and t['bbox'][1]<=y<=t['bbox'][3] for t in tables)
    outside=sorted([w for w in words if not in_table(w)],key=lambda w:(w['top'],w['x0']))
    lines=[]
    for w in outside:
        if not lines or abs(w['top']-lines[-1]['top'])>3:
            lines.append({'top':w['top'],'words':[w]})
        else:lines[-1]['words'].append(w)
    events=[]
    for line in lines:
        cells=[]
        for word in sorted(line['words'],key=lambda w:w['x0']):
            if cells and word['x0']-cells[-1]['x1']<=7:
                cells[-1]['text']+=' '+word['text']
                cells[-1]['x1']=word['x1']
                cells[-1]['bottom']=max(cells[-1]['bottom'],word['bottom'])
            else:cells.append({'x0':word['x0'],'top':word['top'],'x1':word['x1'],'bottom':word['bottom'],'text':word['text']})
        events.append({'top':line['top'],'c':[[round(c['x0']),c['text']] for c in cells],
                       'provenance':[{'bbox':[c['x0'],c['top'],c['x1'],c['bottom']],'text':c['text']} for c in cells],
                       'kind':'prose'})
    for index,table in enumerate(tables):
        previous=None
        for row,(y0,y1) in enumerate(table['rows']):
            signature=(table['matrix'][row],[r['bbox'] if r else None for r in table['provenance'][row]])
            if previous is not None and signature==previous:
                continue
            previous=signature
            events.append({'top':y0,'c':[[round(table['columns'][col][0]),value] for col,value in enumerate(table['matrix'][row]) if value],
                           'provenance':table['provenance'][row],'kind':'table','table_index':index})
    events.sort(key=lambda e:e['top'])
    encoded=[]
    for r,event in enumerate(events):
        event['r']=r
        encoded.append(json.dumps({'r':r,'c':event['c']},ensure_ascii=False,separators=(',',':')))
    return {'text':'\n'.join(encoded),'records':events,'table_count':len(tables),
            'page_size':[page.width,page.height],'tables':[{'bbox':t['bbox'],'columns':t['columns']} for t in tables]}

def main():
    import pdfplumber
    import importlib.metadata
    parser=argparse.ArgumentParser()
    parser.add_argument('--pdf',required=True)
    parser.add_argument('--pages',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    pdfpath=Path(args.pdf)
    digest=hashlib.sha256(pdfpath.read_bytes()).hexdigest()
    pages=json.loads(args.pages)
    output={'version':VERSION,'pdf_sha256':digest,'worker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'pdfplumber_version':pdfplumber.__version__,'pdfminer_version':importlib.metadata.version('pdfminer.six'),
            'pages':{}}
    with pdfplumber.open(pdfpath) as pdf:
        for p in pages:
            if type(p) is not int or not 1<=p<=len(pdf.pages):raise ValueError('Invalid page')
            output['pages'][str(p)]=extract_page(pdf.pages[p-1])
    Path(args.out).write_text(json.dumps(output,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':main()
