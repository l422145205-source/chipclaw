"""Deterministic controls; these synthetic tables are not a QA benchmark."""
from copy import deepcopy
import json

import pytest

from chipclaw.fact_verifier import extract_facts, parse_quantities, verify_claim, verify_output


def source(rows, symbol='IQ', parameter='Quiescent current', unit='µA'):
    columns = [10, 70, 230, 350, 400, 450, 510]
    headers = ['Symbol', 'Parameter', 'Conditions', 'Min', 'Typ', 'Max', 'Unit']
    matrix = [headers] + [[symbol, parameter, conditions, lo, typ, hi, unit]
                          for conditions, lo, typ, hi in rows]
    records = []
    for row, cells in enumerate(matrix):
        records.append({'r': row, 'kind': 'table', 'table_index': 0,
            'c': [[x, text] for x, text in zip(columns, cells) if text],
            'provenance': [{'bbox': [x, row * 20, x + 45, row * 20 + 20], 'text': text,
                            'word_boxes': [[x + 2, row * 20 + 2, x + 40, row * 20 + 12, text]]}
                           for x, text in zip(columns, cells)]})
    encoded = [{'r': r['r'], 'c': r['c']} for r in records]
    return {'id': 'test:p1', 'span_id': 'test:p1', 'doc_id': 'test', 'page': 1,
            'source_sha256': 'a' * 64, 'geometry_records': records,
            'source_rows': encoded, 'text': '\n'.join(json.dumps(r) for r in encoded)}


def iq_source():
    return source([('', '', '330', '420'), ('power-down (shutdown)', '', '0.5', '2')])


@pytest.mark.parametrize('text,verdict', [
    ('normal operation IQ typical current is 330 µA.', 'supported'),
    ('IQ 在正常工作模式下典型值为 0.33 mA。', 'supported'),
    ('IQ 在 power-down 模式下典型值为 500 nA。', 'supported'),
    ('normal operation IQ typical current is 0.5 µA.', 'contradicted'),
    ('power-down IQ typical current is 330 µA.', 'contradicted'),
    ('normal operation IQ typical current is 420 µA.', 'contradicted'),
    ('power-down IQ typical current is 2 µA.', 'contradicted'),
    ('normal operation IQ maximum current is 420 µA.', 'supported'),
    ('power-down IQ maximum current is 0.5 µA.', 'contradicted'),
    ('IQ typical current is 330 µA.', 'unknown'),
    ('normal operation IQ current is 330 µA.', 'unknown'),
    ('The typical value is 330 µA.', 'unknown'),
    ('IQ sleep-mode typical current is 0.5 µA.', 'unknown'),
    ('normal operation IQ default current is 330 µA.', 'unknown'),
    ('IQ 在正常工作模式下不是 0.5 µA，而是典型 330 µA。', 'unknown'),
    ('IQ normal operation current is not 0.5 µA but typically 330 µA.', 'unknown'),
])
def test_mode_role_and_units_bind_before_value(text, verdict):
    result = verify_claim(text, [iq_source()])
    assert result['verdict'] == verdict, result
    if verdict in {'supported', 'contradicted'}:
        assert result['atoms'][0]['source_facts'][0]['binding']['value_cell']['bbox']
        assert result['atoms'][0]['source_facts'][0]['binding']['role_header']['text']


@pytest.mark.parametrize('text,verdict', [
    ('PD maximum power dissipation is 0.454 W.', 'supported'),
    ('PD maximum power dissipation is 454 mW.', 'supported'),
    ('PD maximum power dissipation is 454000 μW.', 'supported'),
    ('PD maximum power dissipation is 454000 微瓦。', 'supported'),
    ('PD maximum power dissipation is 454 W.', 'contradicted'),
    ('PD maximum power dissipation is 454 µW.', 'contradicted'),
    ('PD typical power dissipation is 454 mW.', 'unknown'),
    ('PD maximum power dissipation is 0.454 mA.', 'unknown'),
])
def test_power_dimension_and_si_prefix_are_not_just_numeric_presence(text, verdict):
    span = source([('', '', '', '454')], symbol='PD', parameter='Power dissipation', unit='mW')
    assert verify_claim(text, [span])['verdict'] == verdict


def test_conditions_select_package_before_value():
    span = source([('LQFP48', '', '', '363'), ('LQFP100', '', '', '434')],
                  symbol='PD', parameter='Power dissipation', unit='mW')
    assert verify_claim('LQFP48 PD maximum power dissipation is 363 mW.', [span])['verdict'] == 'supported'
    assert verify_claim('LQFP48 PD maximum power dissipation is 434 mW.', [span])['verdict'] == 'contradicted'
    assert verify_claim('PD maximum power dissipation is 363 mW.', [span])['verdict'] == 'unknown'


def test_numeric_condition_and_missing_condition():
    span = source([('at -40 °C', '', '714', '')], symbol='IDDP', parameter='Pressure measurement current')
    assert verify_claim('IDDP at -40 °C typical current is 714 µA.', [span])['verdict'] == 'supported'
    assert verify_claim('IDDP at 25 °C typical current is 714 µA.', [span])['verdict'] == 'contradicted'
    assert verify_claim('IDDP typical current is 714 µA.', [span])['verdict'] == 'unknown'
    iq = source([('at 25 °C', '', '330', '')])
    assert verify_claim('IQ at 25 °C typical current is 330 µA.', [iq])['verdict'] == 'supported'
    assert verify_claim('IQ at -40 °C typical current is 330 µA.', [iq])['verdict'] == 'contradicted'


def test_exact_range_endpoints_not_any_point_in_allowed_interval():
    span = source([('', '2.7', '', '5.5')], symbol='VS', parameter='Operating supply range', unit='V')
    assert verify_claim('Operating supply voltage range is 2.7 to 5.5 V.', [span])['verdict'] == 'supported'
    assert verify_claim('Operating supply voltage range is 3.0 to 5.5 V.', [span])['verdict'] == 'contradicted'
    assert verify_claim('Operating supply voltage typical value is 3.0 V.', [span])['verdict'] == 'unknown'


def test_multiple_different_rows_do_not_select_by_requested_number():
    span = source([('', '', '330', ''), ('', '', '500', '')])
    assert verify_claim('IQ typical current is 330 µA.', [span])['verdict'] == 'unknown'


def test_duplicate_equal_facts_do_not_change_numeric_verdict():
    span = source([('', '', '330', '')])
    assert verify_claim('IQ typical current is 330 µA.', [span, deepcopy(span)])['verdict'] == 'supported'


def test_missing_provenance_or_changed_original_rows_are_unknown():
    span = source([('', '', '330', '')])
    missing = deepcopy(span)
    missing['geometry_records'][1]['provenance'] = []
    assert verify_claim('IQ typical current is 330 µA.', [missing])['verdict'] == 'unknown'
    changed = deepcopy(span)
    changed['geometry_records'][1]['c'][2][1] = '999'
    assert verify_claim('IQ typical current is 999 µA.', [changed])['verdict'] == 'unknown'
    no_identity = deepcopy(span)
    no_identity['source_sha256'] = 'invalid'
    assert verify_claim('IQ typical current is 330 µA.', [no_identity])['verdict'] == 'unknown'


def test_layout_only_or_prose_never_promoted_to_physical_support():
    span = source([('', '', '330', '')])
    span.pop('geometry_records')
    assert verify_claim('IQ typical current is 330 µA.', [span])['verdict'] == 'unknown'
    assert verify_claim('Registers are configurable.', [span])['verdict'] == 'unknown'


def test_formula_and_missing_unit_cells_are_unknown():
    span = source([('', '', 'VDD+0.3', '')], symbol='VIH', parameter='Input voltage', unit='V')
    assert verify_claim('VIH typical input voltage is 3.6 V.', [span])['verdict'] == 'unknown'
    span = source([('', '', '330', '')], unit='')
    assert verify_claim('IQ typical current is 330 µA.', [span])['verdict'] == 'unknown'


def test_question_does_not_supply_missing_condition_or_role():
    assert verify_claim('IQ current is 330 µA.', [iq_source()],
                        question='What is normal operation IQ typical current?')['verdict'] == 'unknown'


def test_unrecognised_text_condition_is_not_silently_dropped():
    span = source([('all outputs disabled', '', '330', '')])
    assert verify_claim('IQ typical current is 330 µA.', [span])['verdict'] == 'unknown'
    assert verify_claim('With all outputs disabled IQ typical current is 330 µA.', [span])['verdict'] == 'supported'


def test_numeric_value_alone_does_not_bind_a_named_voltage_condition():
    span = source([('VDD = 3.3 V', '', '330', '')])
    assert verify_claim('At 3.3 V IQ typical current is 330 µA.', [span])['verdict'] == 'unknown'
    assert verify_claim('VDD = 3.3 V IQ typical current is 330 µA.', [span])['verdict'] == 'supported'


def test_verify_output_checks_only_claim_citations():
    span = source([('', '', '330', '')])
    other = deepcopy(span)
    other['id'] = other['span_id'] = 'other:p1'
    other['geometry_records'][1]['c'][2][1] = '500'
    other['geometry_records'][1]['provenance'][4]['text'] = '500'
    other['source_rows'][1]['c'][2][1] = '500'
    output = {'claims': [{'text': 'IQ typical current is 500 µA.', 'span_ids': ['test:p1']}]}
    assert verify_output(output, [span, other])['contradicted_claim_indices'] == [0]
    output['claims'][0]['span_ids'] = ['absent']
    assert verify_output(output, [span])['claims'][0]['verdict'] == 'unknown'


def test_sub_nano_difference_is_not_abs_tol_accepted():
    span = source([('', '', '1', '')], unit='nA')
    assert verify_claim('IQ typical current is 1.0001 nA.', [span])['verdict'] == 'contradicted'


def test_unrelated_same_page_number_cannot_satisfy_claim():
    span = source([('', '', '330', '420')])
    extra = source([('', '', '0.5', '2')], symbol='IO', parameter='Other current')
    extra['id'] = extra['span_id'] = 'test:p2'
    assert verify_claim('IQ typical current is 0.5 µA.', [span, extra])['verdict'] == 'contradicted'


def test_signed_quantities_and_case_significant_power_units():
    assert parse_quantities('-40 °C')[0]['value_si'] == -40
    assert parse_quantities('1 MW')[0]['value_si'] == 1e6
    assert parse_quantities('1 mW')[0]['value_si'] == 1e-3
    assert len(extract_facts([iq_source()])['facts']) == 4
