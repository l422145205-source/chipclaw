"""Rebuild v4's public Qwen/visual dependencies at the recorded revisions."""
import json
from cache_v3_models import ROOT,download_model

def main():
    # Do not resolve the visual base's moving default branch.
    catalog=json.loads((ROOT/'reports/v3-start/model_cache.json').read_text(encoding='utf-8'))
    expected=[('Qwen/Qwen3-Reranker-0.6B','e61197ed45024b0ed8a2d74b80b4d909f1255473','teacher'),
        ('vidore/colSmol-256M','c79b633e17e060cc109b11aa2bf92a1517d3bd6f','visual_adapter'),
        ('vidore/ColSmolVLM-Instruct-256M-base','99ca96f1f6b95b3a69e6abef74a2416cb738fed0','visual_base')]
    for repo,revision,key in expected:
        recorded=(ROOT/catalog[key]).resolve()
        target=(ROOT/'.cache/v3/models'/repo.replace('/','--')/revision).resolve()
        assert recorded==target,'Frozen public base path changed'
        download_model(repo,revision)
    print('Pinned public dependencies verified; original model catalog unchanged',flush=True)
if __name__=='__main__':main()
