"""Real frozen PDF closure audit and same-source budget regression grid."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chipclaw.config import ROOT
from chipclaw.corpus_v3 import V3_RAW,SOURCES_V3
from chipclaw.geometry_v3 import V3GeometryLoader
from chipclaw.reliable import SourceLoader,build_packet_spans
from chipclaw.packets_v4 import closed_table_spans,build_dependency_spans
from chipclaw.fact_verifier import verify_claim
def main():
    folder=ROOT/'reports/packets-v4';folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists():raise ValueError('Frozen packet audit exists')
    loader=SourceLoader(V3_RAW,sources=SOURCES_V3,geometry_loader=V3GeometryLoader())
    chunks=[json.loads(x) for x in (ROOT/'data/corpus-v3/index/chunks.jsonl').read_text(encoding='utf-8').splitlines()]
    hashes={c['doc_id']:c['source_sha256'] for c in chunks};audits=[]
    visual=json.loads((ROOT/'data/corpus-v3/visual_subset.json').read_text(encoding='utf-8'))
    # Use actual geometry manifests rather than predicting the subset schema.
    for doc in sorted(SOURCES_V3):
        data=json.loads((ROOT/'data/corpus-v3/geometry'/f'{doc}.json').read_text(encoding='utf-8'))
        for number in sorted(map(int,data['pages'])):
            page=loader.read_page(doc,number,hashes[doc]);spans=closed_table_spans(page)
            for span in spans:
                assert span['text']==page['text'][span['start']:span['end']]
                assert set(span['closure']['physical_rows'])<={r['r'] for r in span['source_rows']}
                assert all(r in page['geometry_records'] for r in span['geometry_records'])
            audits.append({'doc_id':doc,'page':number,'closed_physical_tables':len(spans),'page_chars':len(page['text']),
                'closure_chars':[len(s['text']) for s in spans],'closure_offsets':[[s['start'],s['end']] for s in spans]})
    question='What is the typical supply current of MAX3232 with no load and VCC 3.3 V to 5 V?'
    claims=['No load, V CC = 3.3 V to 5 V, the typical supply current is 0.3 mA.',
        'No load, V CC = 3.3 V to 5 V, the typical supply current is 1 mA.',
        'The typical supply current at no load is 0.3 mA.']
    grid=[]
    for method in ['student_v3','teacher_v3','cascade_v3','visual_fusion_v3']:
        old=json.loads((ROOT/'reports/runtime-v3-final'/f'G-{method}.json').read_text(encoding='utf-8'))
        # Exact old runtime source pool, including any historical injection flag.
        pages=[loader.read_page('max3232',n,hashes['max3232']) for n in old['retrieved_pages']]
        for budget in [1500,3000,6000]:
            for name,builder in [('whole_page_v3',build_packet_spans),('physical_closure_v4',build_dependency_spans)]:
                spans,audit=builder(pages,question,budget,5,{})
                assert sum(len(s['text']) for s in spans)<=budget
                checks=[verify_claim(c,spans) for c in claims];verdicts=[c['verdict'] for c in checks]
                if verdicts[1]=='supported':raise AssertionError('False numeric value accepted')
                if verdicts[2]=='supported':raise AssertionError('Missing condition incorrectly supplied')
                grid.append({'retrieval':method,'selector':name,'budget_chars':budget,'used_chars':audit['used_chars'],
                    'verdicts':verdicts,'spans':[{'id':s['id'],'page':s['page'],'start':s['start'],'end':s['end'],'chars':len(s['text']),'closure':s.get('closure')} for s in spans],
                    'prior_runtime_source_pool':old['retrieved_pages'],'prior_probe_injected':old['probe_source_page_injected']})
    (folder/'physical_page_audit.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    (folder/'budget_grid.json').write_text(json.dumps(grid,indent=2),encoding='utf-8')
    summary={'pages_audited':len(audits),'physical_tables':sum(x['closed_physical_tables'] for x in audits),
        'exact_source_ranges_checked':True,'complete_local_table_rows_checked':True,'comparisons':len(grid),
        'selectors':{m:{'fully_bound_cases':sum(r['verdicts']==['supported','contradicted','unknown'] for r in grid if r['selector']==m),'cases':sum(r['selector']==m for r in grid)} for m in ['whole_page_v3','physical_closure_v4']},
        'false_numeric_acceptances':0,'missing_condition_acceptances':0,
        'scope':'Physical structure audit on 240 frozen pages plus 12 same-source budget cases per selector for one real MAX3232 question; not 24 distinct QA questions.',
        'old_failure_fixed_at_6000':next(r for r in grid if r['retrieval']=='visual_fusion_v3' and r['budget_chars']==6000 and r['selector']=='physical_closure_v4')['verdicts']==['supported','contradicted','unknown']}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
