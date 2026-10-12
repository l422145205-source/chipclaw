"""Synthetic controls for reversible evidence encoding and table closure."""
import pytest
from chipclaw.layout_codec import encode_layout,decode_layout,content_text,VERSION
from chipclaw.reliable import build_spans,source_guard
from chipclaw.scope_guard import missing_instance_value

def page(text,number=1):
    return {"doc_id":"stm32f103","page":number,"text":encode_layout(text),"source_sha256":"a"*64,
            "url":"https://example.test","title":"Synthetic manual","extraction":VERSION,"raw_layout_chars":len(text)}

def test_roundtrip_unicode_signs_padding_and_virtual_columns():
    text=" Title\n        Min      Max        Unit\nVoltage  1.5      9.5        V\n\n        − 40 °C"
    encoded=encode_layout(text)
    assert decode_layout(encoded)==text
    assert "− 40 °C" in content_text(encoded)
    assert len(encode_layout(" "*700+"Value"))<len(" "*700+"Value")

def test_exact_table_selection_retains_every_row_header_condition_and_unit():
    source="Synthetic source\nTable 7. Voltage mode\nParameter       Condition       Min       Max       Unit\n"
    source+="Voltage         active           1         9         V\n"
    source+="Voltage         disabled         0         5         V\n"
    source+="Table 8. Other function\nSomething       unused           3         4         A"
    p=page(source)
    spans,budget=build_spans([p],"Table7: what voltage is allowed in active mode?",[],900)
    selected=[s for s in spans if s["table_id"]=="7"]
    assert len(selected)==1
    visible=content_text(selected[0]["text"])
    assert all(word in visible for word in ("Parameter","Min","Max","Unit","active","disabled","V"))
    assert selected[0]["text"]==p["text"][selected[0]["start"]:selected[0]["end"]]
    assert budget["used_chars"]<=900

def test_overbudget_table_is_not_sliced_to_isolated_values():
    source="Table 77. A large complete table\nParameter      Condition      Min      Max      Unit\n"
    source+="".join("Voltage       state%d        1        9        V\n"%i for i in range(40))
    spans,budget=build_spans([page(source)],"What is Table77 voltage?",[],256)
    assert not spans
    assert budget["rejected_packets"][0]["reason"]=="whole_packet_exceeds_source_budget"

def test_metadata_numbers_are_not_claim_support_and_shared_unit_is_only_deferred():
    # The value 120 exists only in a layout-column offset, not in the source facts.
    p=page("Table 1. Voltage\nParameter      Min      Max      Unit\n"+" "*120+"1      9      V")
    span={**p,"packet_kind":"table"}
    errors,deferred=source_guard("The maximum is 120 V.",[span])
    assert errors and not deferred
    # A zero min inherits MHz; narrow typed mismatches are deferred, not proved.
    span={**page("Table 2. Clock\nMinimum      Maximum     Unit\n0            10 MHz"),"packet_kind":"table"}
    errors,deferred=source_guard("The minimum is 0 MHz.",[span])
    assert not errors and deferred

def test_missing_quantity_is_rejected_even_when_no_same_dimension_literal_exists():
    span={**page("Graph caption\nCurrent (mA) versus voltage (V)\nTicks   0   5   10"),"packet_kind":"prose"}
    errors,_=source_guard("The current is 7.5 mA.",[span])
    assert any(e["code"]=="quantity_missing_from_cited_source" for e in errors)

def test_nanosecond_missing_value_is_checked_without_changing_legacy_guards():
    span={**page("Clock low period 200 ns"),"packet_kind":"prose"}
    errors,_=source_guard("The low period is 28 ns.",[span])
    assert any(e["code"]=="quantity_missing_from_cited_source" for e in errors)
    assert source_guard("The low period is 200 ns.",[span])==([],[])

def test_explicit_table_number_outranks_unrelated_extreme_role_bonus():
    text='Table 11. Serial timing\nParameter      Mode      Maximum      Unit\nRate           master    7            MHz\n'
    text+='Table 12. Maximum operating current under extreme conditions\nParameter      Mode      Maximum      Unit\nCurrent        active    9            mA\n'
    spans,_=build_spans([page(text)],'Table11 serial master maximum rate and operating conditions?',[],500)
    assert spans and spans[0]['table_id']=='11'

def test_operating_caption_is_detected_after_preface_and_prioritized():
    text='An unrelated long introductory source paragraph\n'*6
    text+='Table 21. Absolute maximum voltage\nParameter      Limit       Unit\nSupply         9           V\n'
    text+='Table 22. General operating conditions\nParameter      Min         Max        Unit\nSupply         1           5          V\n'
    spans,_=build_spans([page(text)],'normal working supply voltage lower upper limits?',[],700)
    assert any(s['table_id']=='22' for s in spans)

def test_absolute_ratings_cannot_establish_normal_working_voltage():
    span={**page('Table 3. Absolute maximum ratings\nSupply maximum 9 V'),'packet_kind':'table'}
    errors,_=source_guard('The normal supply maximum is 9 V.',[span],'normal operating supply voltage?')
    assert any(e['code']=='absolute_rating_is_not_operating_condition' for e in errors)

def test_empty_min_dash_cannot_negate_adjacent_max_value():
    span={**page('Table 1. Timing\nMode      Min      Max      Unit\nMaster    -        17       MHz'),'packet_kind':'table'}
    errors,deferred=source_guard('The master maximum is 17 MHz.',[span])
    assert not errors
    errors,_=source_guard('The master maximum is -17 MHz.',[span])
    assert any(e['code']=='quantity_missing_from_cited_source' for e in errors)

def test_real_negative_temperature_inside_merged_condition_cell_remains_negative():
    span={**page('Table 2. Current\nCondition      Max      Unit\nAt − 40 °C     714      μA'),'packet_kind':'table'}
    assert not source_guard('At -40 °C the maximum current is 714 μA.',[span])[0]
    assert source_guard('At +40 °C the maximum current is 714 μA.',[span])[0]

@pytest.mark.parametrize('claim,question,allowed',[
    ('工程师声称最短值是77 μs，这是错误的。实际最短值为20 μs。','工程师声称最短值是77 μs，对吗？',True),
    ('最短值是77 μs。','最短值是77 μs，对吗？',False),
    ('工程师声称最短值是88 μs，这是错误的。','工程师声称最短值是77 μs，对吗？',False),
    ('工程师声称最短值是77 μs，这是错误的。实际最短值为88 μs。','工程师声称最短值是77 μs，对吗？',False),
])
def test_denied_original_quantity_is_distinct_from_positive_or_invented_values(claim,question,allowed):
    span={**page('The minimum selectable time is 20 μs.'),'packet_kind':'prose'}
    errors,_=source_guard(claim,[span],question)
    assert (not errors)==allowed

@pytest.mark.parametrize("question,expected",[
    ("没有测量记录，给出我这块电源板精确实测电流。",True),
    ("Give the exact measured current of my board without measurements.",True),
    ("我的 EEPROM 被重写，没有提供值，给出下次恢复的确定十六进制值。",True),
    ("我没有示波器，如何测量这块板的实际电流？",False),
    ("传感器手册的正常供电范围和典型电流是多少？",False),
    ("Give the specified voltage range and factory default.",False),
])
def test_missing_instance_scope_has_a_narrow_explicit_input_policy(question,expected):
    assert missing_instance_value(question)["refuse"] is expected
