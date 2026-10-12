"""Qwen3 domain ranking LoRA, not answer-generation fine-tuning."""
import gc,hashlib,json,math,random,sys,time
from pathlib import Path
import numpy as np
import torch
from peft import LoraConfig,get_peft_model
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chipclaw.config import ROOT
from chipclaw.teacher_v3 import QwenTeacher
from chipclaw.domain_teacher_v4 import INSTRUCTION
from chipclaw.distillation_v3 import sample_candidates,ranking_loss
from train_minilm_v3 import rows,metrics,dump,sha
def score(ranker,queries,passages):
    ranker.model.eval();records=[]
    for number,q in enumerate(queries):
        for lang in ['en','zh']:
            raw=ranker.predict([(q['query_'+lang],passages[c]['text']) for c in q['candidate_ids']],batch_size=16,return_records=True)
            values=[r['logit_margin'] for r in raw];ids=q['candidate_ids']
            order=sorted(range(len(values)),key=lambda i:(-values[i],ids[i]));ranking=[ids[i] for i in order]
            neg=q.get('condition_negative_id')
            records.append({'id':q['id'],'language':lang,'doc_id':q['doc_id'],'split':q['split'],'rank':ranking.index(q['positive_id'])+1,
                'scores':values,'candidate_ids':ids,'condition_negative_id':neg,
                'condition_correct':None if not neg else values[ids.index(q['positive_id'])]>values[ids.index(neg)]})
        if (number+1)%80==0:print('raw-margin evaluation',q['split'],number+1,'/',len(queries),flush=True)
    return records
def main():
    folder=ROOT/'reports/domain-lora-v4';folder.mkdir(exist_ok=True)
    if (folder/'protocol.json').exists():raise ValueError('Frozen domain training exists')
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);random.seed(42);torch.cuda.reset_peak_memory_stats()
    queries=rows(ROOT/'data/ranking-v3-frozen/queries.jsonl');passages={p['id']:p for p in rows(ROOT/'data/ranking-v3-frozen/passages.jsonl')}
    train=[q for q in queries if q['split']=='train'];dev=[q for q in queries if q['split']=='dev'];holdout=[q for q in queries if q['split']=='new_holdout']
    teacher={(r['id'],r['language']):r for r in rows(ROOT/'reports/teacher-v3-frozen/scores.jsonl')}
    ranker=QwenTeacher(instruction=INSTRUCTION);base_identity=ranker.identity();started=time.perf_counter()
    baseline=score(ranker,dev,passages);dump(folder/'base_domain_prompt_dev.json',metrics(baseline))
    config=LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=['q_proj','v_proj','o_proj'],task_type='CAUSAL_LM',bias='none')
    ranker.model=get_peft_model(ranker.model,config);ranker.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    ranker.model.enable_input_require_grads();ranker.model.config.use_cache=False
    trainable=[p for p in ranker.model.parameters() if p.requires_grad];parameters=sum(p.numel() for p in trainable)
    total_steps=2*math.ceil(len(train)/4)
    protocol={'base':base_identity,'rank':8,'alpha':16,'dropout':.05,'target_modules':['q_proj','v_proj','o_proj'],
        'trainable_parameters':parameters,'full_parameters':sum(p.numel() for p in ranker.model.parameters()),'epochs':2,'updates':total_steps,
        'query_lists_per_batch':4,'candidates_per_list':8,'max_pair_tokens':512,'seed':42,'lr':2e-4,
        'objective':'Source-anchor listwise CE plus physical condition-pair softplus on last-position yes-minus-no logits; no generative token loss or self-KL.',
        'ranking_score':'Raw yes-minus-no logits at training and evaluation; do not sort saturated sigmoid probabilities.',
        'dataset_sha256':{n:sha(ROOT/'data/ranking-v3-frozen'/n) for n in ['queries.jsonl','passages.jsonl']},
        'selection':'Dev ranking only, two checkpoints finalized before reused holdout scoring; previous holdout has already been published.',
        'source_sha256':sha(Path(__file__))}
    dump(folder/'protocol.json',protocol);optimizer=torch.optim.AdamW(trainable,lr=2e-4,weight_decay=.01);step=0;best=-1;history=[]
    target=ROOT/'.cache/v4/domain_lora/best';target.parent.mkdir(parents=True,exist_ok=True)
    with (folder/'steps.jsonl').open('w',encoding='utf-8') as log:
        for epoch in range(2):
            order=list(range(len(train)));random.Random(42+epoch).shuffle(order);ranker.model.train();losses=[]
            for start in range(0,len(order),4):
                batch=[train[i] for i in order[start:start+4]];pairs=[];negatives=[]
                for offset,q in enumerate(batch):
                    lang='en' if (order[start+offset]+epoch)%2==0 else 'zh';soft=[s['score'] for s in teacher[q['id'],lang]['scores']]
                    pos=q['candidate_ids'].index(q['positive_id']);neg=q['candidate_ids'].index(q['condition_negative_id']) if q.get('condition_negative_id') else None
                    selected,_=sample_candidates(soft,pos,neg,8,'stratified');pairs.extend((q['query_'+lang],passages[q['candidate_ids'][i]]['text']) for i in selected)
                    negatives.append(selected.index(neg) if neg is not None else -1)
                inputs,_=ranker.inputs(pairs);optimizer.zero_grad(set_to_none=True)
                # Only the last vocabulary position is materialised, retaining
                # differentiability through the adapter and frozen base layers.
                out=ranker.model(**inputs,logits_to_keep=1,use_cache=False).logits[:,-1,:].float()
                values=(out[:,ranker.yes]-out[:,ranker.no]).reshape(len(batch),8)
                loss,parts=ranking_loss(values,torch.zeros_like(values),torch.zeros(len(batch),device='cuda',dtype=torch.long),torch.tensor(negatives,device='cuda'),False)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite LoRA ranking loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1.)
                warmup=max(1,int(.1*total_steps));scale=(step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/(total_steps-warmup)))
                for group in optimizer.param_groups:group['lr']=2e-4*scale
                optimizer.step();step+=1;losses.append(float(loss.detach()))
                log.write(json.dumps({'step':step,'epoch':epoch+1,'loss':float(loss.detach()),'lr':2e-4*scale,**{k:float(v.detach()) for k,v in parts.items()}})+'\n');log.flush()
                if step%60==0:print('domain LoRA updates',step,'/',total_steps,flush=True)
            ranker.model.gradient_checkpointing_disable();prediction=score(ranker,dev,passages);quality=metrics(prediction)
            history.append({'epoch':epoch+1,'updates':step,'mean_loss':float(np.mean(losses)),'dev':quality})
            if quality['all']['mrr10']>best:
                best=quality['all']['mrr10'];best_epoch=epoch+1;ranker.model.save_pretrained(target,safe_serialization=True)
                (folder/'selected_dev_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in prediction),encoding='utf-8')
            dump(folder/'history.json',history);print('domain LoRA epoch',epoch+1,'selected dev',best,flush=True)
            ranker.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    assert step==total_steps
    from safetensors import safe_open
    with safe_open(target/'adapter_model.safetensors',framework='pt',device='cpu') as weights:
        nonzero=sum(bool(torch.count_nonzero(weights.get_tensor(k))) for k in weights.keys() if 'lora_B' in k)
    assert nonzero>0
    artifact={'adapter_sha256':sha(target/'adapter_model.safetensors'),'base_sha256':base_identity['weight_sha256'],'base_revision':base_identity['revision'],
        'rank':8,'trainable_parameters':parameters,'training_updates':step,'selected_epoch':best_epoch,'selection_split':'dev',
        'nonzero_lora_B_tensors':nonzero,'dataset_sha256':protocol['dataset_sha256'],'instruction':INSTRUCTION}
    dump(target/'training_manifest.json',artifact);dump(folder/'selected_model.json',artifact)
    del ranker,optimizer;gc.collect();torch.cuda.empty_cache()
    from chipclaw.domain_teacher_v4 import DomainTeacher
    tuned=DomainTeacher();prediction=score(tuned,holdout,passages)
    (folder/'reused_holdout_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in prediction),encoding='utf-8')
    summary={'artifact':artifact,'baseline_domain_prompt_dev':metrics(baseline),'selected_dev_mrr10':best,'reused_holdout':metrics(prediction),
        'peak_vram_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.perf_counter()-started,'history':history,'protocol':protocol}
    dump(folder/'summary.json',summary);print(json.dumps({'dev':best,'holdout':summary['reused_holdout']['all'],'adapter':artifact}),flush=True)
if __name__=='__main__':main()
