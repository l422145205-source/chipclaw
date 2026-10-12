"""Pinned ColSmol page-image multi-vector retrieval with original input prompts."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .config import ROOT

QUERY_PREFIX='Query: '
QUERY_SUFFIX='<end_of_utterance>'*10+'\n'
IMAGE_PROMPT='<|im_start|>User: Describe the image.<image><end_of_utterance>'

def maxsim(query,documents):
    """Exact late interaction: sum_i max_j dot(q_i, d_j), no padded tokens."""
    q=np.asarray(query,dtype=np.float32)
    if q.ndim!=2 or len(q)==0 or not np.isfinite(q).all():raise ValueError('Finite nonempty query token matrix required')
    scores=[]
    for document in documents:
        d=np.asarray(document,dtype=np.float32)
        if d.ndim!=2 or d.shape[1]!=q.shape[1] or len(d)==0 or not np.isfinite(d).all():
            raise ValueError('Finite compatible page token matrix required')
        scores.append(float((q@d.T).max(axis=1).sum()))
    return np.array(scores,dtype=np.float32)

class ColSmolEncoder:
    def __init__(self,device='cuda:0'):
        import torch
        from colpali_engine.models import ColIdefics3,ColIdefics3Processor
        from peft import PeftModel
        cache=json.loads((ROOT/'reports/v3-start/model_cache.json').read_text())
        self.base=ROOT/cache['visual_base'];self.adapter=ROOT/cache['visual_adapter'];self.device=device
        for folder in [self.base,self.adapter]:
            manifest=json.loads((folder/'cache_manifest.json').read_text())
            for record in manifest['files']:
                if record['path'].endswith('.safetensors'):
                    with (folder/record['path']).open('rb') as f:
                        if hashlib.file_digest(f,'sha256').hexdigest()!=record['sha256']:raise ValueError('Visual model fingerprint mismatch')
        model,info=ColIdefics3.from_pretrained(self.base,local_files_only=True,
             torch_dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32,
             device_map=device,attn_implementation='sdpa',output_loading_info=True)
        if info['missing_keys'] or info['unexpected_keys'] or info['mismatched_keys']:
            raise ValueError('Visual base architecture mismatch: '+repr(info))
        self.model=PeftModel.from_pretrained(model,self.adapter,local_files_only=True).eval()
        self.processor=ColIdefics3Processor.from_pretrained(self.adapter,local_files_only=True)
        self.processor.tokenizer.padding_side='left'
        self.cache=cache;self.last_metrics=None

    def _encode(self,features):
        import torch
        features=features.to(self.model.device)
        with torch.inference_mode():
            vectors=self.model(**features)
        return [vector[mask.bool()].float().cpu().numpy().astype(np.float16)
                for vector,mask in zip(vectors,features['attention_mask'])]

    def images(self,images):
        start=time.perf_counter()
        # Explicit original b983e40 training prompt/order, avoiding processor
        # drift in engine 0.3.9/0.3.11/0.3.13 noted in the official model card.
        features=self.processor(text=[IMAGE_PROMPT]*len(images),images=[[i.convert('RGB')] for i in images],
                                return_tensors='pt',padding='longest')
        vectors=self._encode(features)
        self.last_metrics={'latency_ms':(time.perf_counter()-start)*1000,'images':len(images),
                           'tokens':[len(v) for v in vectors]}
        return vectors

    def queries(self,queries):
        features=self.processor.tokenizer([QUERY_PREFIX+q+QUERY_SUFFIX for q in queries],padding='longest',return_tensors='pt')
        return self._encode(features)

    def identity(self):
        import importlib.metadata as metadata
        return {**self.cache,'model':'vidore/colSmol-256M','revision':'c79b633e17e060cc109b11aa2bf92a1517d3bd6f',
                'reference_code_commit':'b983e406a9696d39102bbd680772e1764d5ff217',
                'query_prefix':QUERY_PREFIX,'query_suffix':QUERY_SUFFIX,'image_prompt':IMAGE_PROMPT,
                'engine':metadata.version('colpali-engine'),'transformers':metadata.version('transformers'),
                'padding_tokens_removed':True,'document_mask':'all valid tokens, original default ColIdefics3 forward',
                'note':'Pixel-conditioned page retrieval; no visual answer generation or full-library claim.'}

class VisualPageRetriever:
    def __init__(self,device='cuda:0',encoder=None):
        self.manifest=json.loads((ROOT/'data/visual-v3/index_manifest.json').read_text(encoding='utf-8'))
        self.encoder=encoder or ColSmolEncoder(device)
        self.last_metrics=None

    def retrieve(self,question,doc_id=None,top_k=5):
        started=time.perf_counter();query=self.encoder.queries([question])[0]
        pages=[r for r in self.manifest['pages'] if doc_id is None or r['doc_id']==doc_id]
        embeddings=[]
        for page in pages:
            path=ROOT/page['vectors_path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=page['vectors_sha256']:raise ValueError('Page vector fingerprint mismatch')
            embeddings.append(np.load(path,allow_pickle=False)['vectors'])
        scores=maxsim(query,embeddings)
        order=np.argsort(-scores,kind='stable')[:top_k]
        self.last_metrics={'latency_ms':(time.perf_counter()-started)*1000,'pages_searched':len(pages),
                           'scope':'Frozen 240-page source-stratified subset','score':'sum(query token max page token dot product)'}
        return [{**pages[i],'score':float(scores[i])} for i in order]
