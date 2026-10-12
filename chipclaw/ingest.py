"""Page-preserving extraction; raw text retains table line breaks.

No claim of general table/OCR support: complex tables need manual gold review.
"""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import requests
from pypdf import PdfReader
from .config import DATA, INDEX, SOURCES


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_text(text, size=1200, overlap=200):
    if not 0 <= overlap < size:
        raise ValueError('Require 0 <= overlap < size')
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind('\n', start + size // 2, end)
            if boundary > start:
                end = boundary
        value = text[start:end].strip()
        if value:
            yield start, end, value
        if end == len(text):
            break
        start = max(start + 1, end - overlap)


def build(download=False):
    raw = DATA / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    INDEX.mkdir(parents=True, exist_ok=True)
    chunks, manifest = [], []
    for doc, source in SOURCES.items():
        path = raw / f'{doc}.pdf'
        if not path.exists():
            if not download:
                raise FileNotFoundError(f'{path}: run python -m chipclaw.cli ingest --download')
            response = requests.get(source['url'], timeout=(10, 120))
            response.raise_for_status()
            if not response.content.startswith(b'%PDF'):
                raise ValueError(f'Not a PDF: {source["url"]}')
            path.write_bytes(response.content)
        reader = PdfReader(path)
        digest = sha256(path)
        manifest.append({'doc_id': doc, **source, 'sha256': digest,
                         'pages': len(reader.pages), 'indexed_at': datetime.now(timezone.utc).isoformat()})
        for page_no, page in enumerate(reader.pages, 1):
            text = (page.extract_text(extraction_mode='layout') or '').replace('\x00', '').replace('\u200b', '')
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text).strip()
            # Exclude only near-empty pages; do not silently drop register/version tables.
            if len(text) < 40:
                continue
            for ordinal, (start, end, value) in enumerate(split_text(text)):
                chunks.append({'id': f'{doc}:p{page_no}:c{ordinal}', 'doc_id': doc,
                               'page': page_no, 'text': value, 'start': start, 'end': end,
                               'source_sha256': digest, 'url': source['url'] + f'#page={page_no}'})
    (INDEX / 'chunks.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in chunks), encoding='utf-8')
    (INDEX / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'documents': len(manifest), 'pages': sum(m['pages'] for m in manifest), 'chunks': len(chunks)}
