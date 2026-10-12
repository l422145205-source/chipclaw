"""Equal-step/seed MiniLM listwise distillation and two necessary controls."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from safetensors import safe_open
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chipclaw.config import ROOT,RERANKER,RERANKER_REVISION
from chipclaw.distillation_v3 import normalize_scores,sample_candidates,ranking_loss

ARMS=[('stratified_kl','stratified',True),('hard_top_kl','hard_top',True),('stratified_no_kd','stratified',False)]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def score(model,tokenizer,queries,passages,batch=32):
    model.eval();output=[]
    with torch.inference_mode():
        for q in queries:
            for lang in ['en','zh']:
                texts=[passages[cid]['text'] for cid in q['candidate_ids']]
                features=tokenizer([q['query_'+lang]]*len(texts),texts,padding=True,truncation='only_second',max_length=512,return_tensors='pt').to('cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):scores=model(**features).logits[:,0].float().cpu().tolist()
                order=sorted(range(len(scores)),key=lambda i:(-scores[i],q['candidate_ids'][i]));ranking=[q['candidate_ids'][i] for i in order]
                rank=ranking.index(q['positive_id'])+1
                output.append({'id':q['id'],'language':lang,'doc_id':q['doc_id'],'split':q['split'],
                               'rank':rank,'scores':scores,'candidate_ids':q['candidate_ids'],
                               'condition_negative_id':q.get('condition_negative_id'),
                               'condition_correct':None if not q.get('condition_negative_id') else scores[q['candidate_ids'].index(q['positive_id'])]>scores[q['candidate_ids'].index(q['condition_negative_id'])]})
    return output

def metrics(results):
    def part(rows):
        return {'n':len(rows),'mrr10':sum(1/r['rank'] if r['rank']<=10 else 0 for r in rows)/len(rows) if rows else 0,
                'top1':sum(r['rank']==1 for r in rows),'condition_pairs':sum(r['condition_correct'] is not None for r in rows),
                'condition_correct':sum(r['condition_correct'] is True for r in rows)}
    return {'all':part(results),'languages':{language:part([r for r in results if r['language']==language]) for language in ['en','zh']},
            'devices':{doc:part([r for r in results if r['doc_id']==doc]) for doc in sorted({r['doc_id'] for r in results})}}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='ranking-v3-frozen');parser.add_argument('--teacher',default='teacher-v3-frozen')
    parser.add_argument('--epochs',type=int,default=3);parser.add_argument('--batch',type=int,default=4);args=parser.parse_args()
    if not 1<=args.epochs<=5 or not 1<=args.batch<=8:raise ValueError('Bounded training required')
    data=ROOT/'data'/args.data;teacher_folder=ROOT/'reports'/args.teacher;folder=ROOT/'reports/minilm-v3'
    if (folder/'protocol.json').exists():raise ValueError('Frozen training protocol exists; choose separate run, do not repeat')
    folder.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'));teacher_run=json.loads((teacher_folder/'run.json').read_text())
    for name,digest in manifest['dataset_sha256'].items():assert sha(data/name)==digest
    assert teacher_run['dataset_sha256']==manifest['dataset_sha256']
    queries=rows(data/'queries.jsonl');passages={p['id']:p for p in rows(data/'passages.jsonl')}
    teacher_records={(r['id'],r['language']):r for r in rows(teacher_folder/'scores.jsonl')}
    assert len(teacher_records)==2*len(queries)
    train=[q for q in queries if q['split']=='train'];dev=[q for q in queries if q['split']=='dev']
    holdout=[q for q in queries if q['split']=='new_holdout']
    assert not {q['series'] for q in train}&{q['series'] for q in dev+holdout}
    snapshot=ROOT/'.cache/huggingface/hub'/('models--'+RERANKER.replace('/','--'))/'snapshots'/RERANKER_REVISION
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    steps_per_epoch=math.ceil(len(train)/args.batch);total_steps=steps_per_epoch*args.epochs
    tokenizer_audit=[]
    for q in queries:
        for lang in ['en','zh']:
            ids=tokenizer(q['query_'+lang],add_special_tokens=False)['input_ids']
            tokenizer_audit.append({'id':q['id'],'language':lang,'tokens':len(ids),'unknown_tokens':ids.count(tokenizer.unk_token_id)})
    protocol={'base_model':RERANKER,'base_revision':RERANKER_REVISION,'base_sha256':sha(snapshot/'model.safetensors'),
              'data':args.data,'dataset_sha256':manifest['dataset_sha256'],'teacher_records_sha256':sha(teacher_folder/'scores.jsonl'),
              'seed':42,'epochs':args.epochs,'batch_query_lists':args.batch,'candidates_per_list':8,
              'steps_per_epoch':steps_per_epoch,'total_steps_each_arm':total_steps,
              'learning_rate':2e-5,'weight_decay':.01,'warmup_fraction':.1,'gradient_clip':1,'max_pair_tokens':512,
              'teacher_minmax':'Full 16-candidate query preference scores, then sample; zero range -> all zero.',
              'sampling':'One sourced positive, a disjoint-cell condition conflict if available, remaining deterministic quantile anchors or hard-top.',
              'objective':'source weak-label CE + source-condition softplus + optional listwise KL at temperature=1',
              'arms':ARMS,'language_schedule':'Same deterministic alternation per query/batch/epoch in all arms, raw en and zh.',
              'auxiliary_heads_enabled':False,'auxiliary_reason':'No sufficiently independently reviewed column-role/condition labels; automatic anchors stay weak and do not justify a new gold auxiliary target.',
              'checkpoint_selection':'Maximum mean bilingual dev MRR@10; ties earlier epoch; no holdout selection.',
              'old_80_used':False,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
              'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [Path(__file__),ROOT/'chipclaw/distillation_v3.py']}}
    dump(folder/'protocol.json',protocol);dump(folder/'tokenizer_audit.json',tokenizer_audit)
    source_snapshot=folder/'source_snapshot';source_snapshot.mkdir()
    for rel in protocol['source_sha256']:shutil.copy2(ROOT/rel,source_snapshot/Path(rel).name)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    selected={};all_started=time.perf_counter()
    for name,policy,distill in ARMS:
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.cuda.reset_peak_memory_stats()
        arm=folder/name;arm.mkdir();target=ROOT/'.cache/v3/student'/name/'best';target.parent.mkdir(parents=True,exist_ok=True)
        model=AutoModelForSequenceClassification.from_pretrained(snapshot,local_files_only=True).to('cuda')
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
        baseline=metrics(score(model,tokenizer,dev,passages));best=-1;steps=0;history=[];started=time.perf_counter();sampling=[]
        with (arm/'steps.jsonl').open('w',encoding='utf-8') as log:
            for epoch in range(args.epochs):
                order=list(range(len(train)));random.Random(42+epoch).shuffle(order);model.train();losses=[]
                for begin in range(0,len(order),args.batch):
                    batch_rows=[train[i] for i in order[begin:begin+args.batch]]
                    texts=[];questions=[];teacher_values=[];condition_indices=[]
                    for offset,q in enumerate(batch_rows):
                        lang='en' if (order[begin+offset]+epoch)%2==0 else 'zh'
                        record=teacher_records[(q['id'],lang)]
                        assert record['candidate_ids']==q['candidate_ids']
                        soft=[s['score'] for s in record['scores']]
                        positive=q['candidate_ids'].index(q['positive_id'])
                        condition=q['candidate_ids'].index(q['condition_negative_id']) if q.get('condition_negative_id') else None
                        picks,audit=sample_candidates(soft,positive,condition,k=8,policy=policy)
                        if epoch==0:sampling.append({'id':q['id'],'language':lang,'candidate_indices':picks,**audit})
                        questions.extend([q['query_'+lang]]*8);texts.extend([passages[q['candidate_ids'][i]]['text'] for i in picks])
                        teacher_values.append(normalize_scores(soft)[picks].tolist())
                        condition_indices.append(picks.index(condition) if condition is not None else -1)
                    features=tokenizer(questions,texts,padding=True,truncation='only_second',max_length=512,return_tensors='pt').to('cuda')
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        scores=model(**features).logits[:,0].reshape(len(batch_rows),8).float()
                        loss,components=ranking_loss(scores,torch.tensor(teacher_values,device='cuda'),torch.zeros(len(batch_rows),dtype=torch.long,device='cuda'),
                                                       torch.tensor(condition_indices,device='cuda'),distill)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                    warmup=max(1,int(total_steps*.1));scale=(steps+1)/warmup if steps<warmup else .5*(1+math.cos(math.pi*(steps-warmup)/max(1,total_steps-warmup)))
                    for group in optimizer.param_groups:group['lr']=2e-5*scale
                    optimizer.step();steps+=1;losses.append(float(loss.detach()))
                    log.write(json.dumps({'step':steps,'epoch':epoch+1,'loss':float(loss.detach()),'learning_rate':2e-5*scale,
                                          **{k:float(v.detach()) for k,v in components.items()}})+'\n')
                dev_rows=score(model,tokenizer,dev,passages);dev_metrics=metrics(dev_rows)
                entry={'epoch':epoch+1,'steps':steps,'mean_loss':float(np.mean(losses)),'dev':dev_metrics,'elapsed_seconds':time.perf_counter()-started}
                history.append(entry);dump(arm/'history.json',history)
                if dev_metrics['all']['mrr10']>best:
                    best=dev_metrics['all']['mrr10'];best_epoch=epoch+1
                    model.save_pretrained(target,safe_serialization=True);tokenizer.save_pretrained(target)
                    (arm/'selected_dev_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in dev_rows),encoding='utf-8')
                print(name,'epoch',epoch+1,'steps',steps,'dev',best,flush=True)
        assert steps==total_steps
        changed=0;sum_squared=0;count=0;max_change=0
        with safe_open(snapshot/'model.safetensors',framework='pt',device='cpu') as base,safe_open(target/'model.safetensors',framework='pt',device='cpu') as tuned:
            for key in base.keys():
                a=base.get_tensor(key)
                if not a.is_floating_point():continue
                diff=tuned.get_tensor(key).double()-a.double();maximum=float(diff.abs().max());changed+=maximum>0
                sum_squared+=float(diff.square().sum());count+=diff.numel();max_change=max(max_change,maximum)
        artifact={'weight_sha256':sha(target/'model.safetensors'),'base_weight_sha256':protocol['base_sha256'],
                  'training_steps':steps,'changed_float_tensors':changed,'parameter_rms_change':math.sqrt(sum_squared/count),
                  'parameter_max_change':max_change,'best_epoch':best_epoch,'trainable_parameters':sum(p.numel() for p in model.parameters()),
                  'max_pair_tokens':512,'data_sha256':manifest['dataset_sha256'],'arm':name,'policy':policy,'distill':distill,
                  'language_support':'Raw bilingual weak-anchor training; original English WordPiece limitations retained.'}
        assert changed>0 and artifact['weight_sha256']!=protocol['base_sha256']
        dump(target/'training_manifest.json',artifact);dump(arm/'sampling.json',sampling)
        result={'artifact':artifact,'baseline_dev':baseline,'selected_dev_mrr10':best,'elapsed_seconds':time.perf_counter()-started,
                'peak_vram_bytes':torch.cuda.max_memory_allocated(),'history':history}
        dump(arm/'summary.json',result);selected[name]=result
        del model,optimizer;gc.collect();torch.cuda.empty_cache()
    # Arm selection also happens strictly before any new-holdout scoring.
    chosen=max((name for name,_,_ in ARMS),key=lambda name:selected[name]['selected_dev_mrr10'])
    dump(folder/'selected_model.json',{'arm':chosen,'selection_split':'dev','dev_mrr10':selected[chosen]['selected_dev_mrr10'],
                                     'weight_sha256':selected[chosen]['artifact']['weight_sha256'],
                                     'tie_break':'Predeclared arm order; no holdout metric used.'})
    model=AutoModelForSequenceClassification.from_pretrained(snapshot,local_files_only=True).to('cuda')
    base_holdout=metrics(score(model,tokenizer,holdout,passages));dump(folder/'base_new_holdout.json',base_holdout)
    del model;gc.collect();torch.cuda.empty_cache()
    # Every arm checkpoint is selected/finalized BEFORE any new-holdout scoring.
    for name,_,_ in ARMS:
        checkpoint=ROOT/'.cache/v3/student'/name/'best'
        model=AutoModelForSequenceClassification.from_pretrained(checkpoint,local_files_only=True).to('cuda')
        holdout_rows=score(model,tokenizer,holdout,passages)
        (folder/name/'new_holdout_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in holdout_rows),encoding='utf-8')
        selected[name]['new_holdout']=metrics(holdout_rows);dump(folder/name/'summary.json',selected[name])
        del model;gc.collect();torch.cuda.empty_cache()
    dump(folder/'summary.json',{'arms':selected,'total_elapsed_seconds':time.perf_counter()-all_started,
                              'comparison':'Equal seeds/updates/language order/base, fixed candidate pool. Synthetic source anchors; not human QA accuracy.'})
    print(json.dumps({'arms':{k:{'dev':v['selected_dev_mrr10'],'holdout':v['new_holdout']['all']} for k,v in selected.items()}}),flush=True)

if __name__=='__main__':main()
