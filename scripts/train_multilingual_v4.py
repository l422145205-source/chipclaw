"""Real bilingual MiniLM full-parameter training; dev selects before reused holdout."""
import gc,hashlib,json,math,random,sys,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from safetensors import safe_open
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chipclaw.config import ROOT
from chipclaw.distillation_v3 import normalize_scores,sample_candidates,ranking_loss
from cache_multilingual_v4 import FOLDER,REPO,REVISION
from train_minilm_v3 import score,metrics,rows,dump,sha
ARMS=[('multilingual_no_kd',False),('multilingual_kl',True)]
def main():
    folder=ROOT/'reports/multilingual-v4';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'protocol.json').exists():raise ValueError('Frozen run exists; do not overwrite training records')
    data=ROOT/'data/ranking-v3-frozen';manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['dataset_sha256'].items():assert sha(data/name)==digest
    queries=rows(data/'queries.jsonl');passages={p['id']:p for p in rows(data/'passages.jsonl')}
    teacher={(r['id'],r['language']):r for r in rows(ROOT/'reports/teacher-v3-frozen/scores.jsonl')}
    groups={s:[q for q in queries if q['split']==s] for s in ['train','dev','new_holdout']}
    assert not {q['series'] for q in groups['train']}&{q['series'] for q in groups['dev']+groups['new_holdout']}
    tokenizer=AutoTokenizer.from_pretrained(FOLDER,local_files_only=True)
    audit={}
    for lang in ['en','zh']:
        ids=[tokenizer(q['query_'+lang],add_special_tokens=False)['input_ids'] for q in queries]
        audit[lang]={'forms':len(ids),'tokens':sum(map(len,ids)),'unknown':sum(x.count(tokenizer.unk_token_id) for x in ids),'forms_with_unknown':sum(tokenizer.unk_token_id in x for x in ids)}
        audit[lang]['unknown_fraction']=audit[lang]['unknown']/audit[lang]['tokens']
    dump(folder/'tokenizer_audit.json',audit)
    steps_per_epoch=math.ceil(len(groups['train'])/4);steps_total=3*steps_per_epoch
    protocol={'base_repo':REPO,'base_revision':REVISION,'base_weight_sha256':sha(FOLDER/'model.safetensors'),
        'dataset_sha256':manifest['dataset_sha256'],'teacher_scores_sha256':sha(ROOT/'reports/teacher-v3-frozen/scores.jsonl'),
        'train_intents':len(groups['train']),'dev_intents':len(groups['dev']),'holdout_intents':len(groups['new_holdout']),
        'seed':42,'epochs':3,'batch_query_lists':4,'candidate_sample_size':8,'steps_each_arm':steps_total,'learning_rate':2e-5,
        'objective':'source-anchor CE + physical-role/condition pairwise softplus + optional teacher listwise KL',
        'sampling':'identical teacher-score quantile sampling and language schedule across arms',
        'selection':'dev only; both arms finalized before any reused v3 holdout scoring',
        'holdout_note':'Previously reported v3 holdout, reused for regression; not a new unseen benchmark or independent human QA.',
        'arms':ARMS,'source_sha256':{p:sha(ROOT/p) for p in ['scripts/train_multilingual_v4.py','chipclaw/distillation_v3.py']}}
    dump(folder/'protocol.json',protocol)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;started=time.perf_counter();results={}
    for name,kd in ARMS:
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.cuda.reset_peak_memory_stats()
        arm=folder/name;arm.mkdir();target=ROOT/'.cache/v4/student'/name/'best';target.parent.mkdir(parents=True,exist_ok=True)
        model=AutoModelForSequenceClassification.from_pretrained(FOLDER,local_files_only=True).to('cuda')
        baseline_rows=score(model,tokenizer,groups['dev'],passages)
        (arm/'base_dev_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in baseline_rows),encoding='utf-8')
        baseline=metrics(baseline_rows);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
        best=-1;step=0;history=[];sampling=[]
        with (arm/'steps.jsonl').open('w',encoding='utf-8') as log:
            for epoch in range(3):
                order=list(range(len(groups['train'])));random.Random(42+epoch).shuffle(order);model.train();losses=[]
                for begin in range(0,len(order),4):
                    batch=[groups['train'][i] for i in order[begin:begin+4]];questions=[];texts=[];soft_rows=[];conflicts=[]
                    for offset,q in enumerate(batch):
                        lang='en' if (order[begin+offset]+epoch)%2==0 else 'zh';rec=teacher[q['id'],lang]
                        assert rec['candidate_ids']==q['candidate_ids']
                        values=[s['score'] for s in rec['scores']];pos=q['candidate_ids'].index(q['positive_id'])
                        neg=q['candidate_ids'].index(q['condition_negative_id']) if q.get('condition_negative_id') else None
                        picks,sample=sample_candidates(values,pos,neg,8,'stratified')
                        if epoch==0:sampling.append({'id':q['id'],'language':lang,'picks':picks,**sample})
                        questions.extend([q['query_'+lang]]*8);texts.extend(passages[q['candidate_ids'][i]]['text'] for i in picks)
                        soft_rows.append(normalize_scores(values)[picks].tolist());conflicts.append(picks.index(neg) if neg is not None else -1)
                    inputs=tokenizer(questions,texts,padding=True,truncation='only_second',max_length=512,return_tensors='pt').to('cuda')
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=model(**inputs).logits[:,0].reshape(len(batch),8).float()
                        loss,parts=ranking_loss(logits,torch.tensor(soft_rows,device='cuda'),torch.zeros(len(batch),dtype=torch.long,device='cuda'),torch.tensor(conflicts,device='cuda'),kd)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                    warmup=max(1,int(.1*steps_total));scale=(step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/(steps_total-warmup)))
                    for g in optimizer.param_groups:g['lr']=2e-5*scale
                    optimizer.step();step+=1;losses.append(float(loss.detach()))
                    log.write(json.dumps({'step':step,'epoch':epoch+1,'loss':float(loss.detach()),'lr':2e-5*scale,**{k:float(v.detach()) for k,v in parts.items()}})+'\n')
                dev_rows=score(model,tokenizer,groups['dev'],passages);quality=metrics(dev_rows)
                history.append({'epoch':epoch+1,'steps':step,'mean_loss':float(np.mean(losses)),'dev':quality})
                if quality['all']['mrr10']>best:
                    best=quality['all']['mrr10'];best_epoch=epoch+1;model.save_pretrained(target,safe_serialization=True);tokenizer.save_pretrained(target)
                    (arm/'selected_dev_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in dev_rows),encoding='utf-8')
                dump(arm/'history.json',history);print(name,'epoch',epoch+1,'updates',step,'dev',best,flush=True)
        assert step==steps_total
        changed=0
        with safe_open(FOLDER/'model.safetensors',framework='pt',device='cpu') as base,safe_open(target/'model.safetensors',framework='pt',device='cpu') as tuned:
            for key in base.keys():
                a=base.get_tensor(key)
                if a.is_floating_point():changed+=not torch.equal(a,tuned.get_tensor(key))
        artifact={'arm':name,'weight_sha256':sha(target/'model.safetensors'),'base_weight_sha256':protocol['base_weight_sha256'],
            'training_steps':step,'changed_float_tensors':changed,'trainable_parameters':sum(p.numel() for p in model.parameters()),
            'best_epoch':best_epoch,'selection_split':'dev','base_repo':REPO,'base_revision':REVISION,'data_sha256':manifest['dataset_sha256']}
        assert changed>0 and artifact['weight_sha256']!=artifact['base_weight_sha256']
        dump(target/'training_manifest.json',artifact);dump(arm/'sampling.json',sampling)
        results[name]={'artifact':artifact,'baseline_dev':baseline,'selected_dev_mrr10':best,'peak_vram_bytes':torch.cuda.max_memory_allocated(),'history':history}
        dump(arm/'summary.json',results[name]);del model,optimizer;gc.collect();torch.cuda.empty_cache()
    chosen=max([name for name,_ in ARMS],key=lambda n:results[n]['selected_dev_mrr10'])
    dump(folder/'selected_model.json',{'arm':chosen,'selection_split':'dev','dev_mrr10':results[chosen]['selected_dev_mrr10'],'weight_sha256':results[chosen]['artifact']['weight_sha256']})
    for name,_ in ARMS:
        model=AutoModelForSequenceClassification.from_pretrained(ROOT/'.cache/v4/student'/name/'best',local_files_only=True).to('cuda')
        prediction=score(model,tokenizer,groups['new_holdout'],passages);results[name]['reused_holdout']=metrics(prediction)
        (folder/name/'reused_holdout_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in prediction),encoding='utf-8')
        dump(folder/name/'summary.json',results[name]);del model;gc.collect();torch.cuda.empty_cache()
    dump(folder/'summary.json',{'arms':results,'selected_arm':chosen,'tokenizer':audit,'elapsed_seconds':time.perf_counter()-started,'protocol':protocol})
    print(json.dumps({'selected':chosen,'holdout':{n:r['reused_holdout'] for n,r in results.items()},'tokenizer':audit}),flush=True)
if __name__=='__main__':main()
