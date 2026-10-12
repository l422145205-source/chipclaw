"""Query contracts, independent of an online translation model or QA gold labels."""
import json
import numpy as np
import pytest
from chipclaw.query_rewrite import (protect_query, restore_translation, terminology_query,
                                    validate_cached_translation)
from chipclaw.retrieval import Retriever, protect_view_candidates, rrf


def test_protection_retains_chip_signed_range_units_and_mode():
    query = 'INA226 在电源电压 3.3 V、温度 −40 至 125 °C 时，power-down 的典型电流是多少？'
    p = protect_query(query)
    raw = ' '.join(a['marker'] for a in p['atoms'])
    restored = restore_translation(p, raw)
    assert 'INA226' in restored and '3.3 V' in restored and '-40 to 125 °C' in restored
    assert 'power-down' in restored and 'typical' in restored and 'supply voltage' in restored
    assert all(a['source'] == query[a['start']:a['end']].strip() for a in p['atoms'])


@pytest.mark.parametrize('unit', [
    'µW', 'μW', 'uW', 'mW', 'MW', 'kW', 'W',
    'µF', 'μF', 'uF', 'nF', 'pF', 'mF', 'F',
    'µA', 'μA', 'uA', 'mA', 'MA', 'nA', 'A',
    'µV', 'μV', 'mV', 'V', 'MHz', 'kHz', 'GHz',
    'µs', 'μs', 'us', 'ms', 'ns', 's', 'kPa', 'hPa', 'Pa', 'kΩ', 'Ω', 'mA/bit',
])
@pytest.mark.parametrize('space', [' ', ''])
def test_quantity_protection_is_one_atom_and_preserves_entire_unit(unit, space):
    quantity = '1.25' + space + unit
    question = 'STM32F103 最大功耗为 ' + quantity + ' 时，电流是多少？'
    protection = protect_query(question)
    numbers = [atom for atom in protection['atoms'] if atom['kind'] == 'number']
    assert len(numbers) == 1 and numbers[0]['source'] == quantity
    assert numbers[0]['canonical'] == quantity
    assert unit not in protection['masked']
    raw = ' '.join(atom['marker'] for atom in protection['atoms'])
    assert quantity in restore_translation(protection, raw)
    english, audit = terminology_query(question)
    assert quantity in english and any(a['source'] == quantity for a in audit['protected_atoms'])


@pytest.mark.parametrize('unit', ['µUnsupportedUnit', 'μUnsupportedUnit'])
def test_unknown_unicode_prefixed_token_survives_without_dropping_prefix(unit):
    english, _ = terminology_query('TMP117 条件为 1.25 ' + unit + ' 时，电流是多少？')
    assert '1.25 ' + unit in english


@pytest.mark.parametrize('sign', ['-', '−', '–', '—'])
def test_temperature_negative_sign_is_never_converted_to_positive_or_range(sign):
    protection = protect_query('BME280 在 ' + sign + '40 °C 时的典型电流是多少？')
    numbers = [a for a in protection['atoms'] if a['kind'] == 'number']
    assert len(numbers) == 1 and numbers[0]['canonical'] == '-40 °C'
    raw = ' '.join(a['marker'] for a in protection['atoms'])
    assert '-40 °C' in restore_translation(protection, raw)
    english, _ = terminology_query(protection['question'])
    assert '-40 °C' in english and 'to 40' not in english


@pytest.mark.parametrize('separator', ['至', '到', '~', '～', '–', '—', 'to'])
def test_range_separator_and_endpoint_signs_are_normalized_separately(separator):
    question = 'BME280 温度 –40 °C ' + separator + ' +125 °C 时的典型电流是多少？'
    protection = protect_query(question)
    numbers = [a for a in protection['atoms'] if a['kind'] == 'number']
    assert len(numbers) == 1 and numbers[0]['canonical'] == '-40 °C to +125 °C'
    assert '-40 °C to +125 °C' in restore_translation(protection, ' '.join(a['marker'] for a in protection['atoms']))


@pytest.mark.parametrize('unit,canonical', [
    ('微瓦', 'µW'), ('毫瓦', 'mW'), ('瓦特', 'W'), ('微安', 'µA'), ('毫安', 'mA'),
    ('安培', 'A'), ('毫伏', 'mV'), ('伏特', 'V'), ('微法', 'µF'), ('纳法', 'nF'),
    ('皮法', 'pF'), ('摄氏度', '°C'), ('℃', '°C'),
])
def test_verifier_supported_chinese_units_are_explicitly_mapped_not_dropped(unit, canonical):
    question = 'TMP117 条件为 1.25 ' + unit + ' 时，电流是多少？'
    p = protect_query(question)
    assert any(a['source'] == '1.25 ' + unit and a['canonical'] == '1.25 ' + canonical for a in p['atoms'])
    english, _ = terminology_query(question)
    assert '1.25 ' + canonical in english


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'invented_marker', 'reordered', 'introduced_number', 'remaining_chinese'])
def test_translation_rejects_marker_corruption_and_new_facts(fault):
    p = protect_query('TMP117 在 3.3 V 下的最大电流是多少？')
    markers = [a['marker'] for a in p['atoms']]
    raw = ' '.join(markers)
    if fault == 'missing': raw = ' '.join(markers[1:])
    if fault == 'duplicate': raw += ' ' + markers[0]
    if fault == 'invented_marker': raw += ' CCP9999X'
    if fault == 'reordered': raw = ' '.join(reversed(markers))
    if fault == 'introduced_number': raw += ' at 25 C'
    if fault == 'remaining_chinese': raw += ' 最大'
    with pytest.raises(ValueError): restore_translation(p, raw)


def test_terminology_expansion_is_explicit_and_does_not_invent_temperature():
    english, audit = terminology_query('BME280 在常温、睡眠模式下的典型供电电流是多少？')
    assert english == 'BME280 room temperature sleep mode typical supply current'
    assert '25' not in english
    assert audit['method'] == 'hand_authored_terminology_expansion'
    assert not audit['semantic_equivalence_proven']
    assert any(term['source'] == '供电电流' for term in audit['matched_terms'])


def test_legacy_cache_requires_numeric_and_condition_fidelity():
    original = 'INA226 在 3.3 V 下的典型电流是多少？'
    assert validate_cached_translation(original, 'INA226 3.3 V typical current')
    assert not validate_cached_translation(original, 'INA226 3.3 V maximum current')
    assert not validate_cached_translation(original, 'INA226 5 V typical current')
    assert not validate_cached_translation(original, 'INA226 3.3 V typical current at 25 C')


@pytest.mark.parametrize('source_unit,wrong_unit', [('mA','MA'), ('mW','MW'), ('µW','W'), ('μF','F'), ('kW','KW')])
def test_legacy_cache_rejects_unit_prefix_case_or_micro_sign_loss(source_unit, wrong_unit):
    original = 'INA226 在 3 ' + source_unit + ' 下的典型电流是多少？'
    assert validate_cached_translation(original, 'INA226 3 ' + source_unit + ' typical current')
    assert not validate_cached_translation(original, 'INA226 3 ' + wrong_unit + ' typical current')


def test_legacy_cache_rejects_lost_negative_temperature_sign():
    original = 'BME280 在 –40 °C 时的典型电流是多少？'
    assert validate_cached_translation(original, 'BME280 -40 °C typical current')
    assert not validate_cached_translation(original, 'BME280 40 °C typical current')


def test_candidate_budget_retains_each_language_view_without_gold_labels():
    views = [list(range(50)), list(range(50, 100))]
    rank, scores = rrf(views)
    admitted, audit = protect_view_candidates(rank, views, budget=30)
    assert len(set(admitted[:30])) == 30
    assert set(views[0][:15] + views[1][:15]) == set(admitted[:30])
    assert admitted[:30] == sorted(admitted[:30], key=lambda i: (-scores[i], i))
    assert audit['per_view_quota'] == 15


@pytest.mark.parametrize('views,budget', [
    ([list(range(50)), list(range(40, 90)), list(range(80, 130))], 30),
    ([list(range(12)), list(range(5, 17))], 30),
    ([list(range(50)), list(range(50))], 30),
])
def test_view_quota_is_bounded_unique_and_preserves_every_reserved_prefix(views, budget):
    rank, _ = rrf(views)
    admitted, audit = protect_view_candidates(rank, views, budget)
    first = admitted[:budget]
    assert len(first) == min(budget, len(set(rank)))
    assert len(set(admitted)) == len(admitted) and set(admitted) == set(rank)
    for view in views:
        assert set(view[:audit['per_view_quota']]) <= set(first)


def make_retriever(tmp_path):
    chunks = [dict(id=str(i),doc_id='ina226',page=i+1,text=text,start=0,end=len(text))
              for i, text in enumerate(['Shunt voltage conversion time.', 'Bus voltage conversion time.'])]
    (tmp_path/'chunks.jsonl').write_text('\n'.join(json.dumps(c) for c in chunks),encoding='utf-8')
    return Retriever(dense=False, index_dir=tmp_path)


def test_bad_translation_falls_back_with_original_input_and_audit(monkeypatch, tmp_path):
    import chipclaw.generation as generation
    r = make_retriever(tmp_path)
    calls = []
    def bad_translate(messages, **kwargs):
        calls.append(messages)
        return '{"query":"INA226 current"}', {'latency_ms': 7, 'prompt_tokens': 20, 'output_tokens': 4}
    monkeypatch.setattr(generation, 'call_model', bad_translate)
    question = 'INA226 在 3.3 V 下的最大电流是多少？'
    english = r.translate(question)
    assert '3.3 V' in english and 'maximum' in english
    assert r.rewrite_audits[question]['protection_passed'] is False
    assert r.rewrite_metrics[question]['latency_ms'] == 7
    assert '3.3 V' not in calls[0][1]['content']
    assert r.translate(question) == english and len(calls) == 1


def test_retrieval_fuses_original_chinese_and_english_without_extra_llm(monkeypatch, tmp_path):
    import chipclaw.generation as generation
    r = make_retriever(tmp_path)
    question = 'INA226 的分流电压转换时间是多少？'
    english, _ = terminology_query(question)
    r.model = object(); r.vectors = np.array([[1.,0.],[0.,1.]])
    r.query_cache = {question: np.array([0.,1.]), english: np.array([1.,0.])}
    monkeypatch.setattr(generation,'call_model',lambda *args,**kwargs:pytest.fail('offline route must not call LLM'))
    hits = r.retrieve(question, 'ina226', method='protected_hybrid', top_k=2, diversify=False)
    assert {hit['id'] for hit in hits} == {'0','1'}
    assert r.retrieval_events[-1]['query_views'] == [question, english]
    assert r.rewrites[question] == english
    assert r.rewrite_metrics[question]['model'] is None and r.rewrite_metrics[question]['llm_calls'] == 0


def test_unsafe_legacy_translation_is_replaced_by_protected_query(monkeypatch, tmp_path):
    import chipclaw.generation as generation
    r = make_retriever(tmp_path)
    question = 'INA226 3.3 V 典型电流'
    r.rewrites[question] = 'INA226 5 V maximum current'
    p = protect_query(question)
    monkeypatch.setattr(generation, 'call_model', lambda *args, **kwargs:
        (json.dumps({'query': ' '.join(a['marker'] for a in p['atoms'])}), {'latency_ms': 1}))
    assert r.translate(question) == 'INA226 3.3 V typical current'
    assert r.rewrite_audits[question]['method'] == 'protected_llm_translation'
