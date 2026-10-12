"""Scarce page slots and overlapping evidence can defeat density greedy."""
import itertools
import random
import pytest
from chipclaw.orchestration import (select_coverage, select_coverage_multistart,
                                    coverage_score, exact_packet_reference, overlaps, page_key)

def packet(i, cost, page, start=0):
    return {'text': 'x' * cost, 'doc_id': 'ina226', 'page': page,
            'start': start, 'end': start + cost, 'rank': i}

def test_expensive_complete_packet_beats_two_cheap_partial_packets():
    pool = {'p0': packet(0, 40, 1), 'p1': packet(1, 60, 2), 'p2': packet(2, 100, 3)}
    req = [{'id': 'r0'}, {'id': 'r1'}]
    matrix = {'r0': {'p0', 'p1', 'p2'}, 'r1': {'p2'}}
    old, _ = select_coverage(pool, matrix, req, 100, 5)
    new, _, audit = select_coverage_multistart(pool, matrix, req, 100, 5)
    assert coverage_score(old, matrix, req) == 1.15
    assert new == ['p2'] and audit['selected_seed'] == 'p2'
    assert coverage_score(new, matrix, req) == 2

def test_randomized_constraints_and_nonregression_against_exact_reference():
    rng = random.Random(42)
    for _ in range(20):
        pool = {f'p{i}': packet(i, rng.randint(20, 100), rng.randint(1, 3), rng.choice([0, 100])) for i in range(6)}
        req = [{'id': f'r{i}'} for i in range(3)]
        matrix = {r['id']: {k for k in pool if rng.random() < .45} for r in req}
        budget, pages = rng.randint(100, 300), rng.randint(1, 3)
        old, _ = select_coverage(pool, matrix, req, budget, pages)
        new, steps, _ = select_coverage_multistart(pool, matrix, req, budget, pages)
        ref = exact_packet_reference(pool, matrix, req, budget, pages)
        assert sum(len(pool[k]['text']) for k in new) <= budget
        assert len({page_key(pool[k]) for k in new}) <= pages
        assert not any(overlaps(pool[a], pool[b]) for a, b in itertools.combinations(new, 2))
        score = coverage_score(new, matrix, req)
        assert score >= coverage_score(old, matrix, req)
        assert ref['optimal'] and score <= ref['objective'] + 1e-8
        assert [s['objective'] for s in steps] == sorted(s['objective'] for s in steps)

@pytest.mark.parametrize('cost,pages', [(0, 1), (100, 0)])
def test_invalid_budget_fails_explicitly(cost, pages):
    with pytest.raises(ValueError):
        select_coverage_multistart({}, {}, [], cost, pages)

def test_unknown_support_id_and_empty_packet_are_rejected():
    with pytest.raises(ValueError, match='matrix'):
        select_coverage_multistart({'p0': packet(0, 10, 1)}, {'r0': {'missing'}}, [{'id': 'r0'}], 100, 1)
    with pytest.raises(ValueError, match='nonempty'):
        select_coverage_multistart({'p0': packet(0, 0, 1)}, {'r0': set()}, [{'id': 'r0'}], 100, 1)
