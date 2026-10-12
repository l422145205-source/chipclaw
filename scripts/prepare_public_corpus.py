"""Restore frozen corpus bytes from official PDFs; reject version drift."""
import hashlib
import json
import re
import sys
from pathlib import Path
import requests
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chipclaw.config import ROOT
from chipclaw.ingest import split_text


def main():
    folder = ROOT / 'data/corpus-v3'
    manifest = json.loads((folder / 'index/manifest.json').read_text(encoding='utf-8'))
    raw = ROOT / '.cache/v3/manuals'
    raw.mkdir(parents=True, exist_ok=True)
    chunks, pages = [], []
    for source in manifest['sources']:
        doc = source['doc_id']
        path = raw / (doc + '.pdf')
        if path.exists():
            content = path.read_bytes()
        else:
            response = requests.get(source['url'], timeout=(20, 120))
            response.raise_for_status()
            content = response.content
        if not content.startswith(b'%PDF') or hashlib.sha256(content).hexdigest() != source['sha256']:
            raise ValueError(f'{doc}: official PDF differs from frozen source. Obtain the recorded version; do not silently replace it.')
        if not path.exists():
            path.write_bytes(content)
        for number, page in enumerate(PdfReader(path).pages, 1):
            text = (page.extract_text(extraction_mode='layout', layout_mode_space_vertically=False) or '').replace('\x00', '').replace('\u200b', '')
            text = re.sub(r'\n{3,}', '\n\n', text).strip()
            base = dict(doc_id=doc, page=number, source_sha256=source['sha256'], url=source['url'] + '#page=' + str(number))
            pages.append(dict(doc_id=doc, page=number, text=text, source_sha256=source['sha256'], url=base['url']))
            if len(text) < 40:
                continue
            for ordinal, (start, end, value) in enumerate(split_text(text)):
                chunks.append(dict(id=f'{doc}:p{number}:c{ordinal}', doc_id=doc, page=number, text=value, start=start, end=end, source_sha256=source['sha256'], url=base['url']))
        print('Verified', doc, flush=True)
    outputs = [('index/chunks.jsonl', chunks, manifest['corpus_sha256']), ('pages.jsonl', pages, manifest['pages_sha256'])]
    payloads = []
    for name, rows, expected in outputs:
        payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode('utf-8')
        # Original Windows text writer used CRLF. Compare either recorded convention.
        if hashlib.sha256(payload).hexdigest() != expected:
            payload = payload.replace(b'\n', b'\r\n')
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f'{name}: extraction differs; check pinned pypdf version. Existing index was not modified.')
        payloads.append((folder / name, payload))
    for path, payload in payloads:
        path.write_bytes(payload)
    print('Frozen corpus restored and hash verified.')


if __name__ == '__main__':
    main()
