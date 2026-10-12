"""Pinned, project-local multilingual student download; upstream LFS validation."""
import hashlib,json,sys,time
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
REPO='cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
REVISION='1427fd652930e4ba29e8149678df786c240d8825'
FOLDER=ROOT/'.cache/v4/base'/REVISION
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    folder=FOLDER;folder.mkdir(parents=True,exist_ok=True)
    host='https://hf-mirror.com'
    response=requests.get(f'{host}/api/models/{REPO}/tree/{REVISION}',params={'recursive':'true'},timeout=40)
    response.raise_for_status();tree=response.json()
    wanted={'config.json','model.safetensors','sentencepiece.bpe.model','special_tokens_map.json','tokenizer.json','tokenizer_config.json','README.md'}
    entries=[r for r in tree if r.get('path') in wanted]
    if {r['path'] for r in entries}!=wanted:raise ValueError('Incomplete upstream model tree')
    records=[]
    for row in entries:
        target=folder/row['path'];expected=row.get('lfs',{}).get('oid')
        if not target.is_file() or target.stat().st_size!=row['size'] or expected and sha(target)!=expected:
            print('download',row['path'],row['size'],flush=True)
            for attempt in range(3):
                try:
                    with requests.get(f'{host}/{REPO}/resolve/{REVISION}/{row["path"]}',stream=True,timeout=(30,90)) as r:
                        r.raise_for_status()
                        with target.with_suffix(target.suffix+'.partial').open('wb') as f:
                            for part in r.iter_content(4*1024*1024):f.write(part)
                    part=target.with_suffix(target.suffix+'.partial')
                    assert part.stat().st_size==row['size']
                    if expected:assert sha(part)==expected
                    part.replace(target);break
                except Exception:
                    if attempt==2:raise
                    time.sleep(2)
        records.append({'path':row['path'],'bytes':row['size'],'sha256':sha(target),'upstream_lfs_sha256':expected})
    (folder/'cache_manifest.json').write_text(json.dumps({'repo':REPO,'revision':REVISION,'canonical_url':f'https://huggingface.co/{REPO}/tree/{REVISION}','files':records},indent=2),encoding='utf-8')
    print('Verified multilingual model',REVISION,flush=True)
if __name__=='__main__':main()
