"""Local pinned Qwen3 yes/no logit preference; never a factual verifier."""
import hashlib
import json
from pathlib import Path
import time
from .config import ROOT

REPO='Qwen/Qwen3-Reranker-0.6B'
REVISION='e61197ed45024b0ed8a2d74b80b4d909f1255473'
PREFIX='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
INSTRUCTION='Given a web search query, retrieve relevant passages that answer the query'

class QwenTeacher:
    def __init__(self,device='cuda:0',max_length=512,instruction=INSTRUCTION):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        self.path=ROOT/'.cache/v3/models'/REPO.replace('/','--')/REVISION
        manifest=json.loads((self.path/'cache_manifest.json').read_text())
        self.weight_sha=next(r['sha256'] for r in manifest['files'] if r['path']=='model.safetensors')
        # Cache was LFS-verified when downloaded; verify again for runtime trust.
        with (self.path/'model.safetensors').open('rb') as f:
            digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=self.weight_sha:raise ValueError('Teacher fingerprint mismatch')
        self.tokenizer=AutoTokenizer.from_pretrained(self.path,local_files_only=True,padding_side='left')
        self.prefix=self.tokenizer.encode(PREFIX,add_special_tokens=False)
        self.suffix=self.tokenizer.encode(SUFFIX,add_special_tokens=False)
        self.yes=self.tokenizer.convert_tokens_to_ids('yes');self.no=self.tokenizer.convert_tokens_to_ids('no')
        if self.yes==self.no or self.yes is None or self.no is None:raise ValueError('Invalid teacher yes/no tokens')
        self.model=AutoModelForCausalLM.from_pretrained(self.path,local_files_only=True,
                    torch_dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32,
                    device_map=device,attn_implementation='sdpa').eval()
        if not isinstance(instruction,str) or not instruction.strip() or len(instruction)>1000:raise ValueError('Invalid bounded ranking instruction')
        self.instruction=instruction
        self.device=device;self.max_length=max_length;self.last_metrics=None

    def inputs(self,pairs):
        bodies=[f'<Instruct>: {self.instruction}\n<Query>: {q}\n<Document>: {p}' for q,p in pairs]
        full=self.tokenizer(bodies,add_special_tokens=False,padding=False,truncation=False)['input_ids']
        limit=self.max_length-len(self.prefix)-len(self.suffix)
        if any(len(self.tokenizer.encode(f'<Instruct>: {self.instruction}\n<Query>: {q}\n<Document>: ',add_special_tokens=False))>=limit for q,_ in pairs):
            raise ValueError('Query consumes teacher passage budget')
        ids=[self.prefix+body[:limit]+self.suffix for body in full]
        audit=[{'input_ids_sha256':hashlib.sha256(json.dumps(row,separators=(',',':')).encode()).hexdigest(),
                'tokens':len(row),'body_tokens_before_truncation':len(body),'truncated':len(body)>limit}
               for row,body in zip(ids,full)]
        return self.tokenizer.pad({'input_ids':ids},padding=True,return_tensors='pt').to(self.model.device),audit

    def predict(self,pairs,batch_size=8,return_records=False):
        import torch
        started=time.perf_counter();records=[]
        with torch.inference_mode():
            for i in range(0,len(pairs),batch_size):
                inputs,audit=self.inputs(pairs[i:i+batch_size])
                logits=self.model(**inputs,logits_to_keep=1,use_cache=False).logits[:,-1,:].float()
                yes,no=logits[:,self.yes],logits[:,self.no]
                probabilities=torch.stack([no,yes],dim=1).log_softmax(1)[:,1].exp().cpu().tolist()
                for meta,y,n,p in zip(audit,yes.cpu().tolist(),no.cpu().tolist(),probabilities):
                    records.append({**meta,'yes_logit':y,'no_logit':n,'logit_margin':y-n,'score':p})
        if self.device.startswith('cuda'):torch.cuda.synchronize()
        self.last_metrics={'pairs':len(pairs),'latency_ms':(time.perf_counter()-started)*1000,
                           'forward_batches':(len(pairs)+batch_size-1)//batch_size,'identity':self.identity()}
        return records if return_records else [r['score'] for r in records]

    def identity(self):
        return {'repo':REPO,'revision':REVISION,'weight_sha256':self.weight_sha,'max_input_tokens':self.max_length,
                'yes_token_id':self.yes,'no_token_id':self.no,'instruction':self.instruction,
                'prompt_sha256':hashlib.sha256((PREFIX+self.instruction+SUFFIX).encode()).hexdigest(),
                'interpretation':'Two-logit normalized teacher preference, not factual truth or an API call.'}
