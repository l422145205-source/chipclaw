"""Actual local teacher inference on fixed bilingual query/candidate inputs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chipclaw.config import ROOT
from chipclaw.teacher_v3 import QwenTeacher

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
def main():
    import torch
    torch.set_num_threads(4);torch.manual_seed(42)
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='ranking-v3-frozen');parser.add_argument('--name',default='teacher-v3-frozen');args=parser.parse_args()
    if Path(args.data).name!=args.data:raise ValueError('Simple dataset name required')
    if Path(args.name).name!=args.name:raise ValueError('Simple run name required')
    data=ROOT/'data'/args.data;folder=ROOT/'reports'/args.name;folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists():raise ValueError('Frozen teacher run exists')
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['dataset_sha256'].items():assert sha(data/name)==digest
    queries=rows(data/'queries.jsonl');passages={p['id']:p for p in rows(data/'passages.jsonl')}
    teacher=QwenTeacher();started=time.perf_counter();teacher.predict([(queries[0]['query_en'],passages[queries[0]['positive_id']]['text'])])
    record_path=folder/'scores.jsonl';seen=set()
    if record_path.exists():
        for record in rows(record_path):
            if record.get('query_sha256')!=hashlib.sha256(next(q for q in queries if q['id']==record['id'])['query_'+record['language']].encode()).hexdigest():raise ValueError('Resume input query differs')
            seen.add((record['id'],record['language']))
    config={'data':args.data,'dataset_sha256':manifest['dataset_sha256'],'teacher':teacher.identity(),
            'scoring':'Final-position yes/no logits normalized over only no/yes, logits_to_keep=1, no generation.',
            'batch_size':8,'languages':['en','zh'],'seed':42,'warmup_pairs':1,'paid_external_api_calls':0,
            'source_sha256':{'chipclaw/teacher_v3.py':sha(ROOT/'chipclaw/teacher_v3.py'),
                             'scripts/score_teacher_v3.py':sha(Path(__file__))}}
    (folder/'run.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    with record_path.open('a',encoding='utf-8') as out:
        for index,q in enumerate(queries):
            for language in ['en','zh']:
                if (q['id'],language) in seen:continue
                query=q['query_'+language]
                pairs=[(query,passages[cid]['text']) for cid in q['candidate_ids']]
                records=teacher.predict(pairs,return_records=True)
                record={'id':q['id'],'split':q['split'],'language':language,'candidate_ids':q['candidate_ids'],
                        'query_sha256':hashlib.sha256(query.encode()).hexdigest(),
                        'passage_sha256':[hashlib.sha256(p.encode()).hexdigest() for _,p in pairs],
                        'scores':records,'metrics':teacher.last_metrics}
                out.write(json.dumps(record,ensure_ascii=False)+'\n');out.flush()
            if (index+1)%25==0:print('teacher queries',index+1,'/',len(queries),'elapsed',round(time.perf_counter()-started,1),flush=True)
    results=rows(record_path)
    assert len(results)==2*len(queries)
    summary={'queries':len(queries),'language_query_lists':len(results),'scored_pairs':sum(len(r['scores']) for r in results),
             'elapsed_this_invocation_seconds':time.perf_counter()-started,'records_sha256':sha(record_path),
             'truncated_inputs':sum(s['truncated'] for r in results for s in r['scores']),
             'peak_vram_bytes':torch.cuda.max_memory_allocated(),'teacher':teacher.identity(),
             'note':'Teacher preference only; source labels remain explicit weak anchors, no human factual gold.'}
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
