"""Conservative source-scoped condition negatives, separate from whole chunks.

These annotations are authored/source-reviewed, not automatic human labels.
Only exact source spans or an explicit Min/Typ/Max row may become a passage.
The source block remains in its original partition even when two conflicting
conditions share that block. A chunk containing both conditions is never a
negative: only the independently scoped evidence can be contrasted.
"""
import re
from .ranking_data import HOLDOUT_DOCS, group_for, normalize


def source_span(chunk, text):
    passage, text = normalize(chunk['text']), normalize(text)
    if not text or passage.count(text) != 1:
        raise ValueError('Source span must occur exactly once in its chunk')
    start = passage.index(text)
    return {'text': text, 'start': start, 'end': start + len(text)}


def verify_span(chunk, span):
    passage = normalize(chunk['text'])
    start, end = span['start'], span['end']
    if not (0 <= start < end <= len(passage)) or passage[start:end] != span['text']:
        raise ValueError('Condition evidence source span mismatch')
    if normalize(span['text']) != span['text']:
        raise ValueError('Condition source spans use normalized character offsets')


def min_typ_max_cells(header, row):
    """Accept only an explicit simple three-value row; fail on ambiguous layout.

    This is intentionally narrower than general PDF/table extraction. It must
    not guess multi-line inherited headings, mixed units, or missing columns.
    """
    if normalize(header) != 'Symbol Parameter Min Typ Max Unit':
        raise ValueError('Ambiguous Min/Typ/Max table header')
    pattern = r'(?P<parameter>[A-Za-z][A-Za-z0-9_]*(?:\(\d+\))? [A-Za-z][A-Za-z ]*?) (?P<min>[+-]?\d+(?:\.\d+)?) (?P<typ>[+-]?\d+(?:\.\d+)?) (?P<max>[+-]?\d+(?:\.\d+)?) (?P<unit>[A-Za-zµμ°]+)'
    match = re.fullmatch(pattern, normalize(row))
    if not match:
        raise ValueError('Ambiguous Min/Typ/Max numeric row')
    values = match.groupdict()
    if not float(values['min']) <= float(values['typ']) <= float(values['max']):
        raise ValueError('Min/Typ/Max columns are inconsistent')
    return {statistic: {'parameter': values['parameter'], 'statistic': statistic,
                        'value': values[statistic], 'unit': values['unit']}
            for statistic in ['min', 'typ', 'max']}


def scoped_passage(evidence, chunk):
    """Rebuild the training input from checked spans, never trust stored text."""
    for span in evidence['spans'].values():
        verify_span(chunk, span)
    if evidence['representation'] == 'source_spans':
        if set(evidence['spans']) not in ({'statement'}, {'context', 'statement'}):
            raise ValueError('Unexpected condition source span roles')
        statement = evidence['spans']['statement']['text']
        if evidence['condition_marker'].lower() not in statement.lower():
            raise ValueError('Condition marker absent from its source statement')
        if evidence['parameter_anchor'].lower() not in ' '.join(s['text'] for s in evidence['spans'].values()).lower():
            raise ValueError('Parameter anchor absent from condition evidence')
        return '\n'.join(evidence['spans'][name]['text'] for name in ['context', 'statement'] if name in evidence['spans'])
    if evidence['representation'] == 'min_typ_max_cell':
        if set(evidence['spans']) != {'context', 'header', 'row', 'conditions'}:
            raise ValueError('Min/Typ/Max cell lacks its table lineage')
        cells = min_typ_max_cells(evidence['spans']['header']['text'], evidence['spans']['row']['text'])
        cell = cells[evidence['condition_value']]
        if evidence['parameter_anchor'] != cell['parameter']:
            raise ValueError('Min/Typ/Max parameter mismatch')
        # The cell input is a source-derived serialization. Header and whole
        # row stay in the audit lineage, while only the chosen cell is scored.
        return '\n'.join([evidence['spans']['context']['text'],
                          f"{cell['parameter']}; {cell['statistic']}: {cell['value']} {cell['unit']}",
                          evidence['spans']['conditions']['text']])
    raise ValueError('Unknown condition evidence representation')


def validate_condition_triplet(row, corpus, split_map):
    if row.get('negative_kind') != 'condition_conflict_scoped' or row['split'] != 'train':
        raise ValueError('Condition triplets must be explicit training-only conflicts')
    positive, negative = corpus[row['positive_id']], corpus[row['negative_id']]
    if positive['doc_id'] != negative['doc_id'] or positive['doc_id'] in HOLDOUT_DOCS:
        raise ValueError('Condition evidence crosses document or enters document holdout')
    if any(split_map.get(group_for(c)) != 'train' for c in [positive, negative]):
        raise ValueError('Condition evidence crosses source-group split')
    if any(c['source_sha256'] != row['source_sha256'] for c in [positive, negative]):
        raise ValueError('Condition source fingerprint mismatch')
    a, b = row['positive_evidence'], row['negative_evidence']
    if a['parameter_key'] != b['parameter_key'] or a['condition_axis'] != b['condition_axis']:
        raise ValueError('Condition conflict must compare the same parameter and axis')
    if a['condition_value'] == b['condition_value']:
        raise ValueError('Same condition cannot be labeled a conflicting negative')
    if row['query_condition'] != {'axis': a['condition_axis'], 'value': a['condition_value']}:
        raise ValueError('Query condition does not match the positive evidence')
    if row['query_condition_marker'].lower() not in row['query'].lower():
        raise ValueError('Query lacks its explicit condition marker')
    pos_text, neg_text = scoped_passage(a, positive), scoped_passage(b, negative)
    if normalize(pos_text) == normalize(neg_text):
        raise ValueError('Identical passages cannot be a condition contrast')
    if a['representation'] == b['representation'] == 'source_spans':
        pos_statement, neg_statement = a['spans']['statement']['text'], b['spans']['statement']['text']
        if (a['condition_marker'].lower() in neg_statement.lower()
                or b['condition_marker'].lower() in pos_statement.lower()):
            raise ValueError('Evidence includes both conditions; whole-chunk false negative rejected')
    return pos_text, neg_text


def training_passages(row, corpus, split_map):
    if row.get('negative_kind') == 'condition_conflict_scoped':
        return validate_condition_triplet(row, corpus, split_map)
    from .ranking_data import validate_triplet
    validate_triplet(row, corpus, split_map)
    return normalize(corpus[row['positive_id']]['text']), normalize(corpus[row['negative_id']]['text'])


def visible_span_end(tokenizer, query, passage, max_length=512):
    features = tokenizer(query, passage, truncation='only_second', max_length=max_length,
                         return_offsets_mapping=True)
    return max((offset[1] for offset, sequence in zip(features['offset_mapping'], features.sequence_ids())
                if sequence == 1), default=0)
