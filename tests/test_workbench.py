"""Public demo must be self-contained and must never impersonate live inference."""
import json
import hashlib
from fastapi.testclient import TestClient
from chipclaw.workbench import ROOT, app


def test_demo_routes_and_boundary():
    with TestClient(app) as client:
        assert client.get('/').status_code == 200
        assert client.get('/workbench/status').json()['live'] is False
        assert client.post('/evidence', json={'question': 'new question'}).status_code == 404
        assert client.get('/assets/app.js').status_code == 200
        assert client.get('/demo/../../config.py').status_code == 404


def test_all_published_runs_are_actual_valid_records():
    catalog = json.loads((ROOT / 'web/demo/catalog.json').read_text(encoding='utf-8'))
    assert len(catalog['documents']) == 20
    assert len(catalog['runs']) >= 20
    for entry in catalog['runs']:
        assert '/' not in entry['file'] and '..' not in entry['file']
        source = ROOT / 'web/demo' / entry['file']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == entry['sha256']
        record = json.loads(source.read_text(encoding='utf-8'))
        assert record['http_status'] == 200
        result = record['result']
        assert result['metrics']['generation_calls'] == 0
        assert result['budget']['used_chars'] <= result['budget']['budget_chars']
        assert result['evidence']
        assert all(s['doc_id'] == record['request']['doc_id'] for s in result['evidence'])
        for span in result['evidence']:
            assert span['url'].startswith('https://')
            assert len(span['source_sha256']) == 64
        assert all(c['check']['verdict'] in ('supported', 'contradicted', 'unknown') for c in result['claims'])
