"""Same-backbone base comparison and paired bilingual regression descriptions."""
import gc,json,sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chipclaw.config import ROOT
from cache_multilingual_v4 import FOLDER
from train_minilm_v3 import rows,score,metrics,dump
def main():
    folder=ROOT/'reports/multilingual-v4'
    if (folder/'base_holdout_comparison.json').exists():raise ValueError('Frozen base comparison exists')
    torch.set_num_threads(4);queries=rows(ROOT/'data/ranking-v3-frozen/queries.jsonl');holdout=[q for q in queries if q['split']=='new_holdout']
    passages={p['id']:p for p in rows(ROOT/'data/ranking-v3-frozen/passages.jsonl')};tokenizer=AutoTokenizer.from_pretrained(FOLDER,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(FOLDER,local_files_only=True).to('cuda');base=score(model,tokenizer,holdout,passages)
    (folder/'base_reused_holdout_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in base),encoding='utf-8');del model;gc.collect();torch.cuda.empty_cache()
    chosen=json.loads((folder/'selected_model.json').read_text())['arm'];tuned=rows(folder/chosen/'reused_holdout_predictions.jsonl')
    prior=rows(ROOT/'reports/minilm-v3/stratified_no_kd/new_holdout_predictions.jsonl')
    def paired(a,b):
        amap={(r['id'],r['language']):(1/r['rank'] if r['rank']<=10 else 0) for r in a};bmap={(r['id'],r['language']):(1/r['rank'] if r['rank']<=10 else 0) for r in b}
        assert set(amap)==set(bmap);semantic=sorted({k[0] for k in amap})
        delta=np.array([np.mean([bmap[q,l]-amap[q,l] for l in ['en','zh']]) for q in semantic]);rng=np.random.default_rng(42)
        draws=np.mean(rng.choice(delta,size=(3000,len(delta)),replace=True),axis=1)
        return {'semantic_clusters':len(delta),'mean_delta':float(delta.mean()),'descriptive_95_interval':np.quantile(draws,[.025,.975]).tolist()}
    summary={'same_multilingual_base':metrics(base),'selected_multilingual':metrics(tuned),'prior_english_student':metrics(prior),
        'same_backbone_training_delta':paired(base,tuned),'cross_backbone_total_delta':paired(prior,tuned),
        'scope':'Previously used controlled source-anchor holdout; paired semantic clusters, not human QA confidence or architecture-only causal attribution.'}
    dump(folder/'base_holdout_comparison.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
