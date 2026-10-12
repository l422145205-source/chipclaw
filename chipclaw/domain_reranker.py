"""Local fine-tuned English MiniLM reranking; source text remains unchanged."""
import hashlib
import json
import os
from pathlib import Path
import re
import time
from .config import ROOT
from .ranking_data import normalize

DEFAULT_CHECKPOINT = ROOT/'.cache/domain_reranker/domain-reranker-v1/best'


def inspect_checkpoint(path):
    folder = Path(path)
    manifest = json.loads((folder/'training_manifest.json').read_text(encoding='utf-8'))
    weight = folder/'model.safetensors'
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    if manifest.get('weight_sha256') != digest:
        raise ValueError('Domain reranker checkpoint fingerprint mismatch')
    if manifest.get('training_steps', 0) <= 0 or manifest.get('changed_float_tensors', 0) <= 0:
        raise ValueError('Checkpoint lacks recorded parameter training')
    if manifest.get('max_pair_tokens') != 512:
        raise ValueError('Unsupported checkpoint input length')
    return manifest


class DomainReranker:
    def __init__(self, checkpoint=None, device='cpu'):
        self.path = Path(checkpoint or os.environ.get('CHIPCLAW_DOMAIN_RERANKER', str(DEFAULT_CHECKPOINT)))
        self.manifest = inspect_checkpoint(self.path)
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.path, local_files_only=True).to(device)
        self.device, self.last_metrics = device, None

    def identity(self):
        return {'type': 'domain_finetuned_minilm', 'weight_sha256': self.manifest['weight_sha256'],
                'best_epoch': self.manifest['best_epoch'], 'trainable_parameters': self.manifest['trainable_parameters'],
                'device': self.device, 'max_pair_tokens': 512}

    def predict(self, pairs, batch_size=8, show_progress_bar=False):
        import torch
        if any(re.search(r'[\u4e00-\u9fff]', query) for query, _ in pairs):
            raise ValueError('Domain reranker expects English queries; translate before ranking')
        scores, start = [], time.perf_counter()
        self.model.eval()
        with torch.inference_mode():
            for offset in range(0, len(pairs), batch_size):
                section = pairs[offset:offset+batch_size]
                features = self.tokenizer([p[0] for p in section], [normalize(p[1]) for p in section],
                    padding=True, truncation='only_second', max_length=512, return_tensors='pt').to(self.device)
                scores.extend(self.model(**features).logits[:, 0].float().cpu().tolist())
        self.last_metrics = {'pairs': len(pairs), 'latency_ms': (time.perf_counter()-start)*1000,
                             'checkpoint': self.identity(), 'note': 'Local reranker forward passes, separate from Ollama call ledger.'}
        return scores
