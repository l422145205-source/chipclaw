"""Pinned public model downloads inside ChipClaw, with upstream LFS checks."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import requests
import time

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = 'https://hf-mirror.com'
MODELS = [('Qwen/Qwen3-Reranker-0.6B','e61197ed45024b0ed8a2d74b80b4d909f1255473'),
          ('vidore/colSmol-256M','c79b633e17e060cc109b11aa2bf92a1517d3bd6f')]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def download_model(repo, revision):
    folder = ROOT/'.cache/v3/models'/repo.replace('/','--')/revision
    folder.mkdir(parents=True,exist_ok=True)
    response=requests.get(f'{ENDPOINT}/api/models/{repo}/tree/{revision}',params={'recursive':'true'},timeout=45)
    response.raise_for_status(); tree=response.json()
    (folder/'upstream_tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
    selected=[r for r in tree if r.get('type')=='file' and
              (r['path'].endswith(('.json','.safetensors','.txt','.jinja','.md','.yml')))
              and not r['path'].startswith(('1_','2_','3_','additional_chat_templates/'))]
    records=[]
    for row in selected:
        name=row['path']; path=folder/name; path.parent.mkdir(parents=True,exist_ok=True)
        expected=row.get('lfs',{}).get('oid'); size=row['size']
        if not path.is_file() or path.stat().st_size!=size or expected and digest(path)!=expected:
            print('Downloading',repo,name,size,flush=True)
            url=f'{ENDPOINT}/{repo}/resolve/{revision}/{name}'
            last=None
            for attempt in range(3):
                try:
                    with requests.get(url,stream=True,timeout=(30,90)) as r:
                        r.raise_for_status()
                        with path.with_suffix(path.suffix+'.partial').open('wb') as f:
                            for chunk in r.iter_content(4*1024*1024):f.write(chunk)
                    partial=path.with_suffix(path.suffix+'.partial')
                    assert partial.stat().st_size==size,(name,partial.stat().st_size,size)
                    if expected:assert digest(partial)==expected,(repo,name,'LFS mismatch')
                    partial.replace(path); last=None;break
                except Exception as e:
                    last=e; print('retry',repo,name,attempt,str(e)[:200],flush=True)
            if last:raise last
        records.append({'path':name,'bytes':size,'sha256':digest(path),'upstream_lfs_sha256':expected})
    result={'repo':repo,'revision':revision,'download_endpoint':ENDPOINT,
            'canonical_url':f'https://huggingface.co/{repo}/tree/{revision}','files':records}
    (folder/'cache_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Verified model cache',repo,revision,len(records),flush=True)
    return folder

def main():
    teacher=download_model(*MODELS[0]); visual=download_model(*MODELS[1])
    adapter=json.loads((visual/'adapter_config.json').read_text())
    base=adapter['base_model_name_or_path']
    r=requests.get(f'{ENDPOINT}/api/models/{base}',timeout=45);r.raise_for_status()
    info=r.json(); (ROOT/'papers/v3/visual_base_info.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    base_folder=download_model(base,info['sha'])
    result={'teacher':str(teacher.relative_to(ROOT)),'visual_adapter':str(visual.relative_to(ROOT)),
            'visual_base':str(base_folder.relative_to(ROOT)), 'base_revision':info['sha']}
    (ROOT/'reports/v3-start/model_cache.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
