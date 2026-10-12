"""Offline, inspectable requirement rubric. Never imported by inference.

This is a deterministic diagnostic rubric, not independent human accuracy.
Only answer claims are graded; source quotes cannot fill missing answers.
"""
import re
import unicodedata
from .generation import normalize_quote
from .guards import UNIT

_CASE_UNITS = set(UNIT) | {'ns', 'KiB', 'MiB', 'GiB', 'kB/s', 'GB/s', 'B/s', 'b/s'}
_CASE_UNITS.update(prefix+base for prefix in ('', 'p', 'n', 'u', 'm', 'k', 'M', 'G', 'T')
                   for base in ('Hz', 's', 'A', 'V', 'W', 'Ω', 'F', 'H', 'Pa', 'S'))
_SPACED_UNIT = r'(?:[pnumkMGT]|\[[u]+\])(?:\\s(?:[*+?]|\{\d+(?:,\d*)?\})?| +)(?:Hz|Pa|[sAVWΩFH])'
_UNIT_FRAGMENT = re.compile(r'(?<![A-Za-z0-9_\\])(?<!\(\?)(?:'+_SPACED_UNIT+'|'
    +'|'.join(re.escape(unit) for unit in sorted(_CASE_UNITS, key=lambda unit: (-len(unit), unit)))
    +r')(?![A-Za-z0-9_])')
_REGEX_CONTROL = re.compile(r'\(\?P<[^>]+>|\(\?P=[^)]+\)|\(\?\([^)]+\)|'
                            r'\(\?[aiLmsux-]+(?::|\))|\[\^?\]?(?:\\.|[^\\\]])*\]')


def normalized_answer(text):
    return unicodedata.normalize('NFKC', text).replace('\u2212', '-').replace('\u03bc', 'u').replace('\u00b5', 'u')


def case_sensitive_unit_pattern(pattern):
    """Scope explicit unit spellings; no general dimensional inference."""
    pattern = normalized_answer(pattern)
    regions = [(match.start(), match.end(), match.group(0).startswith('['))
               for match in _REGEX_CONTROL.finditer(pattern)]
    parts, cursor = [], 0
    for match in _UNIT_FRAGMENT.finditer(pattern):
        if any(match.start() < end and match.end() > start and not (
            is_class and match.start() == start and match.end() >= end)
            for start, end, is_class in regions):
            continue
        parts.extend([pattern[cursor:match.start()], '(?-i:'+match.group(0)+')'])
        cursor = match.end()
    return ''.join(parts)+pattern[cursor:]


def rubric_search(pattern, text):
    return re.search(case_sensitive_unit_pattern(pattern), text, re.I | re.S)


def citation_audit(result):
    evidence = {item['id']: item for item in result.get('evidence', [])}
    claims = (result.get('output') or {}).get('claims', [])
    errors = []
    for i, claim in enumerate(claims):
        citations = claim.get('citations') or [{'evidence_id': claim.get('evidence_id'), 'quote': claim.get('quote', '')}]
        for citation in citations:
            source = evidence.get(citation.get('evidence_id'))
            quote = normalize_quote(citation.get('quote', ''))
            if source is None:
                errors.append({'claim': i, 'code': 'unknown_evidence'})
            elif len(quote) < 8 or quote not in normalize_quote(source['text']):
                errors.append({'claim': i, 'code': 'quote_not_in_exposed_source'})
    return {'valid': bool(claims) and not errors, 'errors': errors,
            'note': 'Provenance only; no semantic entailment guarantee.'}


def score_answer(case, result=None, error=None):
    if error is not None:
        return {'passed': False, 'complete': False, 'matched_atoms': 0, 'total_atoms': len(case['atoms']),
                'category': 'runtime_failure', 'failure': str(error), 'atom_results': []}
    result = result or {}
    refused = result.get('refused', not bool(result.get('answer', '').strip()))
    claims = (result.get('output') or {}).get('claims', [])
    texts = [normalized_answer(claim.get('text', '')) for claim in claims]
    if not claims and not refused:
        texts = [normalized_answer(result.get('answer', ''))]
    citation = citation_audit(result)
    if case['expected_status'] == 'insufficient_evidence':
        correct_refusal = bool(refused) and not claims and not result.get('answer', '').strip()
        return {'passed': correct_refusal, 'complete': correct_refusal, 'matched_atoms': 0, 'total_atoms': 0,
                'category': 'correct_refusal' if correct_refusal else 'unsupported_answer',
                'citation': citation, 'atom_results': []}
    atoms = []
    for atom in case['atoms']:
        groups = atom['accepted_regex']
        # All properties of one atomic requirement must bind in the same claim.
        matched = any(all(any(rubric_search(pattern, text) for pattern in group) for group in groups)
                      and not any(rubric_search(pattern, text) for pattern in atom.get('forbidden_regex', []))
                      for text in texts)
        forbidden = [pattern for pattern in atom.get('forbidden_regex', [])
                     if any(rubric_search(pattern, text) for text in texts)]
        atoms.append({'id': atom['id'], 'matched': matched and not forbidden,
                      'description': atom['description'], 'forbidden_matches': forbidden})
    matched_count = sum(atom['matched'] for atom in atoms)
    complete = bool(atoms) and matched_count == len(atoms) and not refused
    passed = complete and citation['valid']
    category = ('false_refusal' if refused else 'invalid_citation' if not citation['valid'] else
                'complete' if passed else 'wrong_fact' if any(a['forbidden_matches'] for a in atoms) else
                'partial' if matched_count else 'missing_or_mismatched_requirement')
    return {'passed': passed, 'complete': complete, 'matched_atoms': matched_count, 'total_atoms': len(atoms),
            'category': category, 'atom_results': atoms, 'citation': citation,
            'note': 'Strict claim-local numeric/condition rubric on an AI-reviewed diagnostic set; not human accuracy.'}
