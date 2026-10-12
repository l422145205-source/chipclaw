"""Deterministic, source-row-bound numeric checking; never calls a model.

The verdict concerns only the parsed numeric atoms. ``supported`` is not a
verdict on an entire answer, its completeness, or arbitrary natural language.
Unresolved headers, expressions, conditions and subjects remain ``unknown``.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

VERSION = 'physical-row-numeric-verifier-v1'
METHOD = 'deterministic_physical_row_unit_role_condition'

# Case is significant: mW and MW are different units. Micro sign and Greek mu
# have the same multiplier. ASCII u is accepted only as a unit prefix.
UNITS = {
    'W': ('power', 1.0), 'mW': ('power', 1e-3), 'µW': ('power', 1e-6),
    'μW': ('power', 1e-6), 'uW': ('power', 1e-6), 'kW': ('power', 1e3),
    'MW': ('power', 1e6),
    'A': ('current', 1.0), 'mA': ('current', 1e-3), 'µA': ('current', 1e-6),
    'μA': ('current', 1e-6), 'uA': ('current', 1e-6), 'nA': ('current', 1e-9),
    'V': ('voltage', 1.0), 'mV': ('voltage', 1e-3), 'µV': ('voltage', 1e-6),
    'μV': ('voltage', 1e-6), 'uV': ('voltage', 1e-6),
    'Hz': ('frequency', 1.0), 'kHz': ('frequency', 1e3),
    'MHz': ('frequency', 1e6), 'GHz': ('frequency', 1e9),
    's': ('time', 1.0), 'ms': ('time', 1e-3), 'µs': ('time', 1e-6),
    'μs': ('time', 1e-6), 'us': ('time', 1e-6), 'ns': ('time', 1e-9),
    'Pa': ('pressure', 1.0), 'hPa': ('pressure', 1e2), 'kPa': ('pressure', 1e3),
    '°C': ('temperature', 1.0), '℃': ('temperature', 1.0),
    'Ω': ('resistance', 1.0), 'kΩ': ('resistance', 1e3),
    'F': ('capacitance', 1.0), 'µF': ('capacitance', 1e-6),
    'μF': ('capacitance', 1e-6), 'uF': ('capacitance', 1e-6),
    'nF': ('capacitance', 1e-9), 'pF': ('capacitance', 1e-12),
    '%': ('percentage', 1.0),
    '微安': ('current', 1e-6), '毫安': ('current', 1e-3), '安培': ('current', 1.0),
    '微瓦': ('power', 1e-6), '毫瓦': ('power', 1e-3), '瓦特': ('power', 1.0),
    '毫伏': ('voltage', 1e-3), '伏特': ('voltage', 1.0),
    '摄氏度': ('temperature', 1.0),
}
NUMBER = r'[+\-−–]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+\-]?\d+)?'
UNIT_PATTERN = '|'.join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))
QUANTITY = re.compile(r'(?<![A-Za-z0-9_.])(?P<number>' + NUMBER + r')\s*'
                      r'(?P<unit>' + UNIT_PATTERN + r')(?![A-Za-z/])')
RANGE = re.compile(r'(?<![A-Za-z0-9_.])(?P<low>' + NUMBER + r')\s*'
                   r'(?P<left>' + UNIT_PATTERN + r')?\s*(?:to|至|到|…|\.\.\.|~|～|–)\s*'
                   r'(?P<high>' + NUMBER + r')\s*(?P<right>' + UNIT_PATTERN + r')(?![A-Za-z/])')
ROLE_PATTERN = re.compile(r'\b(?:min(?:imum)?|typ(?:ical)?|max(?:imum)?|default|reset)\b'
                          r'|最小(?:值)?|最低(?:值)?|典型(?:值)?|最大(?:值)?|最高(?:值)?|默认(?:值)?|复位(?:值)?', re.I)
ROLE_NAMES = {'min': 'min', 'minimum': 'min', '最小': 'min', '最低': 'min',
              'typ': 'typ', 'typical': 'typ', '典型': 'typ',
              'max': 'max', 'maximum': 'max', '最大': 'max', '最高': 'max',
              'default': 'default', '默认': 'default', 'reset': 'reset', '复位': 'reset'}

# A small, inspectable bilingual engineering lexicon. It does not contain a
# document ID, a gold numeric value, a benchmark ID, or an answer label.
SUBJECTS = {
    'quiescent_current': ('current', r'\bI\s*Q\b|quiescent\s+current|静态电流|静态功耗电流'),
    'current': ('current', r'\bcurrent\b|电流'),
    'power_dissipation': ('power', r'\bP\s*D\b|power\s+dissipation|耗散功率|功率耗散|功耗|功率'),
    'supply_voltage': ('voltage', r'operating\s+supply\s+range|supply\s+voltage|operating\s+voltage|供电电压|工作电压'),
    'input_voltage': ('voltage', r'input\s+voltage|输入电压'),
    'output_voltage': ('voltage', r'output\s+voltage|输出电压'),
    'voltage': ('voltage', r'\bvoltage\b|电压'),
    'frequency': ('frequency', r'\bfrequency\b|\bclock\b|频率|时钟'),
    'time': ('time', r'conversion\s+time|cycle\s+time|period|转换时间|转换周期|周期|时间'),
    'pressure': ('pressure', r'\bpressure\b|压力|气压'),
    'temperature': ('temperature', r'\btemperature\b|温度'),
    'resistance': ('resistance', r'\bresistance\b|电阻'),
    'capacitance': ('capacitance', r'\bcapacitance\b|电容'),
    'percentage': ('percentage', r'\bhumidity\b|湿度'),
}
SUBJECT_PATTERNS = {k: (d, re.compile(p, re.I)) for k, (d, p) in SUBJECTS.items()}
SYMBOL_EQUIVALENTS = {'IQ': 'quiescent_current', 'PD': 'power_dissipation'}
MODE_PATTERNS = {
    'power_down': r'power[\s-]*down|shutdown|关断|关闭模式|掉电模式',
    'normal': r'normal\s+(?:operation|operating|mode|working)|正常(?:工作|运行|模式)',
    'run': r'\brun[\s-]*mode\b|运行模式',
    'sleep': r'\bsleep\b|睡眠', 'standby': r'\bstandby\b|待机',
    'stop': r'\bstop[\s-]*mode\b|停止模式',
    'forced': r'\bforced\s+mode\b|强制模式',
    'master': r'\bmaster\s+mode\b|主机模式',
    'slave': r'\bslave\s+mode\b|从机模式',
    'flash': r'(?:code\s+(?:from|in)|execut(?:ion|ing)\s+from)\s+flash|从\s*Flash\s*执行',
    'ram': r'(?:code\s+(?:from|in)|execut(?:ion|ing)\s+from)\s+ram|从\s*RAM\s*执行',
    'peripherals_on': r'all\s+peripherals\s+(?:enabled|on)|全部外设(?:开启|启用)',
    'peripherals_off': r'all\s+peripherals\s+(?:disabled|off)|全部外设(?:关闭|禁用)',
    'adc_used': r'ADC\s+(?:used|enabled)|(?:使用|启用)\s*ADC',
    'adc_unused': r'ADC\s+(?:not\s+used|disabled)|(?:不使用|禁用)\s*ADC',
}
MODE_PATTERNS = {k: re.compile(p, re.I) for k, p in MODE_PATTERNS.items()}


def _num(value):
    return float(value.replace('−', '-').replace('–', '-'))


def _close(a, b):
    # No generic 1e-12 absolute tolerance: that would silently accept small but
    # real nano/micro-scale mismatches. Zero is compared exactly.
    return a == b or math.isclose(a, b, rel_tol=1e-8, abs_tol=0.0)


def _unit(text):
    value = re.sub(r'\s+', '', text)
    return value if value in UNITS else None


def _role(text):
    value = text.casefold().strip().rstrip('.')
    # A whole column label may carry a numeric footnote. Keep compound prose
    # such as MAXIMUM SAMPLE RATE outside the strict role grammar.
    value = re.sub(r'\s*\(\d+(?:\s*[,;]\s*\d+)*\)\s*$', '', value).strip()
    value = value.removesuffix('值')
    return ROLE_NAMES.get(value)


def _subjects(text, dimension=None):
    found = {key for key, (d, pattern) in SUBJECT_PATTERNS.items()
             if (dimension is None or d == dimension) and pattern.search(text)}
    if 'quiescent_current' in found:
        found.discard('current')
    if {'input_voltage', 'output_voltage', 'supply_voltage'} & found:
        found.discard('voltage')
    return found


def _symbol(text):
    value = re.sub(r'\([^)]*\)|\s|_', '', text)
    return value if re.fullmatch(r'[A-Z][A-Za-z0-9+−-]{0,12}', value) else None


def _modes(text):
    found = {key for key, pattern in MODE_PATTERNS.items() if pattern.search(text)}
    if 'adc_unused' in found:
        found.discard('adc_used')
    return found


def _packages(text):
    return set(re.findall(r'\b(?:LQFP|LFBGA|UFBGA|TFBGA|UFQFPN|VFQFPN|QFN|BGA)\s*\d+\b', text, re.I))


def parse_quantities(text):
    """Parse strict quantities and exact range endpoints, preserving offsets."""
    result = []
    occupied = []
    for m in RANGE.finditer(text):
        left, right = m['left'] or m['right'], m['right']
        if UNITS[left][0] != UNITS[right][0]:
            continue
        low, high = _num(m['low']) * UNITS[left][1], _num(m['high']) * UNITS[right][1]
        if low > high:
            continue
        for name, unit, value, role in [('low', left, low, 'min'), ('high', right, high, 'max')]:
            result.append({'raw': m.group(0), 'number': _num(m[name]), 'unit': unit,
                           'value_si': value, 'dimension': UNITS[unit][0],
                           'start': m.start(name), 'end': m.end(name), 'range_role': role})
        occupied.append((m.start(), m.end()))
    for m in QUANTITY.finditer(text):
        if any(a <= m.start() < b for a, b in occupied):
            continue
        result.append({'raw': m.group(0), 'number': _num(m['number']), 'unit': m['unit'],
                       'value_si': _num(m['number']) * UNITS[m['unit']][1],
                       'dimension': UNITS[m['unit']][0], 'start': m.start(), 'end': m.end()})
    return sorted(result, key=lambda q: q['start'])


def _provenance(record, column, text):
    refs = [r for r in record.get('provenance', []) if r and r.get('text') == text
            and isinstance(r.get('bbox'), list) and len(r['bbox']) == 4
            and r['bbox'][0] - 1.5 <= column <= r['bbox'][2] + 1.5]
    if not refs:
        return None
    ref = min(refs, key=lambda r: abs(r['bbox'][0] - column))
    return {'row': record['r'], 'column': column, 'text': text,
            'bbox': ref['bbox'], 'word_boxes': ref.get('word_boxes', [])}


def _cell(cells, column):
    matches = [(x, text) for x, text in cells if abs(x - column) <= 1.5]
    return matches[0] if len(matches) == 1 else None


def extract_facts(spans):
    """Build facts only from explicit physical table headers and source cells.

    There is no nearest-number search, arbitrary label forward-fill, or
    numeric-value-based subject/row selection. Layout-only evidence is deferred.
    """
    facts, deferred = [], []
    for span in spans:
        records = span.get('geometry_records') or []
        if not records:
            deferred.append({'span_id': span.get('id') or span.get('span_id'),
                             'reason': 'no_physical_table_records'})
            continue
        fingerprint = span.get('source_sha256', '')
        if not re.fullmatch(r'[a-f0-9]{64}', fingerprint) or not span.get('doc_id') or not isinstance(span.get('page'), int):
            deferred.append({'reason': 'missing_source_identity'})
            continue
        encoded_rows = span.get('source_rows')
        if not isinstance(encoded_rows, list):
            deferred.append({'reason': 'missing_original_encoded_rows'})
            continue
        originals = {r.get('r'): r.get('c') for r in encoded_rows if isinstance(r, dict)}
        headers = {}
        table_facts = defaultdict(list)
        for index, record in enumerate(records):
            if record.get('kind') != 'table':
                continue
            if originals.get(record.get('r')) != record.get('c'):
                deferred.append({'reason': 'geometry_record_differs_from_cited_source_row', 'row': record.get('r')})
                headers.pop(record.get('table_index'), None)
                continue
            table = record.get('table_index')
            cells = record.get('c', [])
            role_cols = {x: _role(text) for x, text in cells if _role(text)}
            unit_cols = [x for x, text in cells if re.fullmatch(r'units?', text, re.I)]
            if role_cols and len(unit_cols) == 1:
                refs = {x: _provenance(record, x, text) for x, text in cells if x in role_cols or x in unit_cols}
                if all(refs.values()):
                    # A table's global condition rows are context only. In
                    # particular they are never interpreted as value columns.
                    leading = [r for r in records[max(0, index - 4):index] if r.get('kind') == 'prose']
                    headers[table] = {'roles': role_cols, 'unit': unit_cols[0], 'refs': refs,
                        'labels': [x for x, text in cells if re.search(r'parameter|symbol', text, re.I)],
                        'conditions': [x for x, text in cells if re.search(r'conditions?', text, re.I)],
                        'default_context': ' '.join(t for r in leading for _, t in r.get('c', []))}
                else:
                    headers.pop(table, None)
                continue
            header = headers.get(table)
            if not header:
                continue
            unit_cell = _cell(cells, header['unit'])
            unit = _unit(unit_cell[1]) if unit_cell else None
            if not unit:
                continue
            labels = [v for x, v in cells if x in header['labels']]
            conditions = [v for x, v in cells if x in header['conditions'] and v not in {'-', '–', '—'}]
            # Some tables have a single combined parameter column. A label must
            # be explicit in this row; merged cells are already rectangle-resolved.
            if not labels:
                continue
            subject_text = ' '.join(labels)
            symbols = {s for label in labels if (s := _symbol(label))}
            concepts = _subjects(subject_text, UNITS[unit][0])
            for s in symbols:
                if s in SYMBOL_EQUIVALENTS:
                    concepts.add(SYMBOL_EQUIVALENTS[s])
            unit_ref = _provenance(record, *unit_cell)
            label_refs = [_provenance(record, x, v) for x, v in cells if x in header['labels']]
            condition_refs = [_provenance(record, x, v) for x, v in cells if x in header['conditions'] and v not in {'-', '–', '—'}]
            if not unit_ref or not all(label_refs + condition_refs):
                continue
            for column, role in header['roles'].items():
                value_cell = _cell(cells, column)
                if not value_cell or not re.fullmatch(NUMBER, re.sub(r'\s+', '', value_cell[1])):
                    continue  # formulas (e.g. VDD+0.3) are deliberately unknown
                value_ref = _provenance(record, *value_cell)
                if not value_ref:
                    continue
                value = _num(re.sub(r'\s+', '', value_cell[1]))
                context = ' '.join(labels + conditions)
                fact = {'doc_id': span['doc_id'], 'page': span['page'],
                    'source_sha256': fingerprint, 'span_id': span.get('id') or span.get('span_id'),
                    'table_index': table, 'row': record['r'], 'role': role,
                    'subject': subject_text, 'symbols': sorted(symbols), 'concepts': sorted(concepts),
                    'conditions': ' '.join(conditions), 'modes': sorted(_modes(context)),
                    'packages': sorted(_packages(context)),
                    'condition_quantities': parse_quantities(context),
                    'default_context': header['default_context'],
                    'dimension': UNITS[unit][0], 'unit': unit, 'number': value,
                    'value_si': value * UNITS[unit][1],
                    'binding': {'kind': 'physical_table_cells', 'value_cell': value_ref,
                        'role_header': header['refs'][column], 'unit_cell': unit_ref,
                        'unit_header': header['refs'][header['unit']],
                        'subject_cells': label_refs, 'condition_cells': condition_refs}}
                facts.append(fact)
                table_facts[table].append(fact)
        # Generic quiescent-current row versus explicitly named shutdown row is
        # a deterministic table contrast, not a freely invented normal mode.
        # Keep this inference visible and narrowly scoped to that table/subject.
        for table, entries in table_facts.items():
            has_down = any('quiescent_current' in f['concepts'] and 'power_down' in f['modes'] for f in entries)
            if has_down:
                for f in entries:
                    if 'quiescent_current' in f['concepts'] and not f['modes']:
                        f['modes'] = ['normal']
                        f['condition_inference'] = 'generic_quiescent_row_contrasted_with_explicit_shutdown_row_in_same_table'
    return {'facts': facts, 'deferred': deferred}


def _sentences(text):
    # Preserve decimal points. Semicolons and commas are retained so context
    # such as "at 25 °C, current ..." stays attached to the same statement.
    return [s.strip() for s in re.split(r'[。\n;；]|(?<!\d)\.(?!\d)', text) if s.strip()]


def _quantity_role(sentence, quantity, quantities):
    if quantity.get('range_role'):
        return quantity['range_role']
    previous_end = max([q['end'] for q in quantities if q['end'] <= quantity['start']] or [0])
    next_start = min([q['start'] for q in quantities if q['start'] >= quantity['end']] or [len(sentence)])
    before = [m for m in ROLE_PATTERN.finditer(sentence, previous_end, quantity['start'])]
    after = [m for m in ROLE_PATTERN.finditer(sentence, quantity['end'], next_start)]
    # Prefer the immediately preceding role. A following role is allowed for
    # Chinese "330 μA 为典型值" and English "330 μA (typical)".
    if before:
        return _role(before[-1].group(0))
    if after and after[0].start() - quantity['end'] <= 18:
        return _role(after[0].group(0))
    return None


def _claim_symbols(sentence, facts, dimension):
    found = set()
    all_symbols = {s for f in facts if f['dimension'] == dimension for s in f['symbols']}
    all_symbols.update(s for s, concept in SYMBOL_EQUIVALENTS.items()
                       if SUBJECTS[concept][0] == dimension)
    for symbol in all_symbols:
        # Match English symbol boundaries without treating Chinese adjacency
        # as a word boundary failure. Spaced PDF subscripts are accepted.
        pattern = r'(?<![A-Za-z0-9])' + r'\s*'.join(re.escape(c) for c in symbol) + r'(?![A-Za-z0-9])'
        if re.search(pattern, sentence):
            found.add(symbol)
    return found


def _matching_subject(fact, concepts, symbols):
    if symbols:
        equivalent = {SYMBOL_EQUIVALENTS[s] for s in symbols if s in SYMBOL_EQUIVALENTS}
        return bool(symbols & set(fact['symbols']) or equivalent & set(fact['concepts']))
    if not concepts:
        return False
    if 'current' in concepts:
        return fact['dimension'] == 'current' and bool(fact['concepts'])
    if 'voltage' in concepts:
        return fact['dimension'] == 'voltage' and bool(fact['concepts'])
    return bool(concepts & set(fact['concepts']))


def _condition_match(fact, modes, packages, qualifiers, claim_text):
    source_modes, source_packages = set(fact['modes']), set(fact['packages'])
    if modes and not modes <= source_modes:
        return 'requested_mode_not_in_source_row'
    if packages and not packages <= source_packages:
        return 'requested_package_not_in_source_row'
    if source_modes - modes or source_packages - packages:
        return 'source_condition_omitted_in_claim'
    source_q = fact['condition_quantities']
    defaults = parse_quantities(fact['default_context'])
    for q in qualifiers:
        local = [s for s in source_q if s['dimension'] == q['dimension']]
        choices = local or [s for s in defaults if s['dimension'] == q['dimension']]
        if not choices:
            return 'numeric_condition_not_resolved'
        if len({s['value_si'] for s in choices}) > 1:
            # Bind a complete, explicit interval by its endpoint roles. A
            # lone point, two anonymous voltages, or multiple intervals remain
            # ambiguous; never accept membership as equality to a range.
            if len(choices)==2 and {s.get('range_role') for s in choices}=={'min','max'}:
                claimed=[s for s in qualifiers if s['dimension']==q['dimension']]
                if len(claimed)!=2 or {s.get('range_role') for s in claimed}!={'min','max'}:
                    return 'ambiguous_numeric_condition_binding'
                choices=[s for s in choices if s['range_role']==q.get('range_role')]
                if not choices:return 'ambiguous_numeric_condition_binding'
            else:
                return 'ambiguous_numeric_condition_binding'
        if not any(_close(s['value_si'], q['value_si']) for s in choices):
            return 'numeric_condition_conflicts_with_source_row'
    # Do not hide explicit condition-column numbers when the claim omits them.
    for q in source_q:
        if not any(q['dimension'] == s['dimension'] and _close(q['value_si'], s['value_si'])
                   and (not q.get('range_role') or q.get('range_role')==s.get('range_role')) for s in qualifiers):
            return 'source_numeric_condition_omitted_in_claim'
    # Unrecognised textual conditions are not silently discarded. They can
    # match only as an explicit literal phrase; otherwise defer to semantic
    # review. Named electrical conditions (e.g. VIN versus VDD) cannot be bound
    # by sharing an anonymous voltage value alone.
    condition_text = fact['conditions']
    remainder = condition_text
    for pattern in MODE_PATTERNS.values():
        remainder = pattern.sub('', remainder)
    for package in _packages(remainder):
        remainder = remainder.replace(package, '')
    remainder = QUANTITY.sub('', RANGE.sub('', remainder))
    remainder = ROLE_PATTERN.sub('', remainder)
    remainder = re.sub(r'\b(?:at|value|under|in|the|mode|and)\b|条件|温度|时|为', '', remainder, flags=re.I)
    if re.search(r'[A-Za-z\u4e00-\u9fff]', remainder):
        normalize = lambda value: re.sub(r'\s+', '', value).casefold()
        if normalize(condition_text) not in normalize(claim_text):
            return 'unresolved_textual_condition_binding'
    return None


def verify_claim(text, selected_spans, question=''):
    """Three-state numeric verdict with source bindings and explicit abstention.

    ``question`` is accepted for integration but never supplies a missing claim
    subject, role or condition. Cited evidence must be the selected original
    spans, not an LLM-generated quotation. Callers should veto contradictions;
    unknown must retain its meaning and must not be presented as rule-verified.
    """
    extracted = extract_facts(selected_spans)
    facts = extracted['facts']
    atoms = []
    for sentence in _sentences(text):
        quantities = parse_quantities(sentence)
        if not quantities:
            continue
        # Negations/comparisons are not safely reducible to affirmative equality.
        if re.search(r'并非|不是|不等于|而非|\b(?:not|isn\x27t|isn’t|less\s+than|greater\s+than)\b|[<>≤≥]', sentence, re.I):
            atoms.extend({'quantity': q, 'verdict': 'unknown', 'reason': 'nonaffirmative_or_comparative_statement'} for q in quantities)
            continue
        target_dimensions = {d for d in {q['dimension'] for q in quantities}
                             if _subjects(sentence, d) or _claim_symbols(sentence, facts, d)}
        if len(target_dimensions) > 1:
            atoms.extend({'quantity': q, 'verdict': 'unknown', 'reason': 'multiple_subject_dimensions_in_one_statement'} for q in quantities)
            continue
        for quantity in quantities:
            dimension = quantity['dimension']
            if target_dimensions and dimension not in target_dimensions:
                continue  # this quantity will be checked as a condition qualifier
            concepts = _subjects(sentence, dimension)
            symbols = _claim_symbols(sentence, facts, dimension)
            role = _quantity_role(sentence, quantity, quantities)
            modes, packages = _modes(sentence), _packages(sentence)
            qualifiers = [q for q in quantities if q['dimension'] != dimension]
            atom = {'quantity': quantity, 'subject_concepts': sorted(concepts),
                    'subject_symbols': sorted(symbols), 'role': role,
                    'modes': sorted(modes), 'packages': sorted(packages),
                    'condition_quantities': qualifiers}
            candidates = [f for f in facts if f['dimension'] == dimension and _matching_subject(f, concepts, symbols)]
            if not candidates:
                atom.update(verdict='unknown', reason='subject_or_physical_source_not_resolved')
            elif not role:
                atom.update(verdict='unknown', reason='claim_typ_min_max_role_missing')
            else:
                role_candidates = [f for f in candidates if f['role'] == role]
                if not role_candidates:
                    atom.update(verdict='unknown', reason='requested_role_not_in_source_header')
                else:
                    matched, failures = [], []
                    for f in role_candidates:
                        failure = _condition_match(f, modes, packages, qualifiers, sentence)
                        if failure:
                            failures.append(failure)
                        else:
                            matched.append(f)
                    if not matched:
                        # Exact wrong numeric condition is a contradiction only
                        # when its subject/role/mode/package candidate is unique.
                        distinct = {(f['doc_id'], f['page'], f['table_index'], f['row']) for f in role_candidates}
                        contradiction = len(distinct) == 1 and failures == ['numeric_condition_conflicts_with_source_row']
                        atom.update(verdict='contradicted' if contradiction else 'unknown',
                            reason=failures[0] if len(set(failures)) == 1 else 'conditions_not_uniquely_bound',
                            condition_failures=sorted(set(failures)))
                        if contradiction:
                            atom['source_facts'] = role_candidates
                    else:
                        values = {f['value_si'] for f in matched}
                        if len(values) != 1:
                            atom.update(verdict='unknown', reason='multiple_source_rows_have_different_values',
                                        candidate_rows=[{'doc_id': f['doc_id'], 'page': f['page'], 'row': f['row']} for f in matched])
                        else:
                            supported = _close(next(iter(values)), quantity['value_si'])
                            atom.update(verdict='supported' if supported else 'contradicted',
                                reason='bound_value_unit_role_condition_match' if supported else 'bound_row_numeric_value_mismatch',
                                source_facts=matched)
            atoms.append(atom)
    verdict = ('contradicted' if any(a['verdict'] == 'contradicted' for a in atoms)
               else 'supported' if atoms and all(a['verdict'] == 'supported' for a in atoms)
               else 'unknown')
    return {'version': VERSION, 'method': METHOD, 'verdict': verdict,
            'scope': 'parsed_affirmative_numeric_atoms_only', 'atoms': atoms,
            'source_fact_count': len(facts), 'deferred_sources': extracted['deferred'],
            'reason': 'no_supported_numeric_atom' if not atoms else 'see_atom_verdicts',
            'limits': 'Does not prove whole-answer entailment or completeness. No model, gold labels or question-derived facts are used.'}


def verify_output(output, spans):
    """Check each claim against only the span IDs that it actually cites."""
    by_id = {s.get('id') or s.get('span_id'): s for s in spans}
    rows = []
    for index, claim in enumerate(output.get('claims', [])):
        ids = claim.get('span_ids', [])
        selected = [by_id[sid] for sid in ids if sid in by_id]
        if not ids or len(selected) != len(ids):
            check = {'version': VERSION, 'method': METHOD, 'verdict': 'unknown',
                     'atoms': [], 'reason': 'unknown_or_missing_cited_span_id'}
        else:
            check = verify_claim(claim.get('text', ''), selected)
        rows.append({'claim_index': index, 'claim_id': claim.get('claim_id'), **check})
    return {'version': VERSION, 'claims': rows,
            'contradicted_claim_indices': [r['claim_index'] for r in rows if r['verdict'] == 'contradicted']}
