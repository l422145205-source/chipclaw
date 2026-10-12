"""Source-group partitions and weak-label checks for domain reranker training."""
import hashlib
import math
import re

HOLDOUT_DOCS = {'bme280', 'ina226'}


def normalize(text):
    return re.sub(r'\s+', ' ', text).strip()


def group_for(row):
    return f"{row['doc_id']}:block{(row['page']-1)//4}"


def assign_splits(facts, seed=42):
    by_doc = {}
    for fact in facts:
        by_doc.setdefault(fact['doc_id'], set()).add(group_for(fact))
    mapping = {}
    for doc, groups in sorted(by_doc.items()):
        if doc in HOLDOUT_DOCS:
            mapping.update({group: 'document_holdout' for group in groups})
            continue
        ordered = sorted(groups, key=lambda group: hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest())
        n_dev = max(1, math.ceil(len(ordered)*.2))
        mapping.update({group: 'dev' if i < n_dev else 'train' for i, group in enumerate(ordered)})
    return mapping


def validate_triplet(row, chunk_map, split_map):
    positive, negative = chunk_map[row['positive_id']], chunk_map[row['negative_id']]
    if row['split'] != 'train' or split_map.get(group_for(positive)) != 'train' or split_map.get(group_for(negative)) != 'train':
        raise ValueError('Training triplet crosses source-group split')
    if positive['doc_id'] != negative['doc_id'] or positive['doc_id'] in HOLDOUT_DOCS:
        raise ValueError('Training triplet crosses document or enters document holdout')
    if group_for(positive) == group_for(negative):
        raise ValueError('Training negative enters the positive source block')
    quote = normalize(row['quote'])
    if not quote or quote not in normalize(positive['text']):
        raise ValueError('Positive lacks the reviewed source span')
    if quote in normalize(negative['text']):
        raise ValueError('Negative duplicates the reviewed source span')
    if positive['source_sha256'] != row['source_sha256']:
        raise ValueError('Positive source fingerprint mismatch')
