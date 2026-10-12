"""Auditable query protection and deterministic Chinese terminology expansion.

The glossary is hand-authored retrieval engineering, not a trained translator.
The contract preserves lexical atoms; it cannot prove relational equivalence.
"""
from collections import Counter
import hashlib
import json
import re

VERSION = 'protected-query-v1.1-unitfix'
CONDITIONS = {
    '正常工作': 'normal operating', '正常模式': 'normal mode',
    '强制模式': 'forced mode', '睡眠模式': 'sleep mode', '休眠模式': 'sleep mode',
    '掉电模式': 'power-down mode', '关断模式': 'shutdown mode',
    '典型值': 'typical', '最大值': 'maximum', '最小值': 'minimum',
    '典型': 'typical', '最大': 'maximum', '最小': 'minimum',
    '睡眠': 'sleep', '休眠': 'sleep', '关断': 'shutdown', '掉电': 'power-down',
    '常温': 'room temperature', '上限': 'upper limit', '下限': 'lower limit',
    '不超过': 'not exceeding', '不能': 'cannot', '不得': 'must not',
    '禁止': 'prohibited', '只在': 'only when', '只有': 'only', '未使能': 'disabled',
}
TERMS = {
    '分流电压': 'shunt voltage', '总线电压': 'bus voltage', '供电电流': 'supply current',
    '电源电流': 'supply current', '工作电流': 'operating current', '电源电压': 'supply voltage',
    '供电电压': 'supply voltage', '转换时间': 'conversion time', '转换周期': 'conversion time',
    '转换速率': 'conversion rate', '输出数据率': 'output data rate', '相对湿度': 'relative humidity',
    '湿度': 'humidity', '压力': 'pressure', '精度': 'accuracy', '误差': 'error',
    '主频': 'clock frequency', '频率': 'frequency', '内核': 'core processor',
    '容量': 'memory size', '闪存': 'flash', '温度': 'temperature', '电压': 'voltage',
    '电流': 'current', '通道': 'channel', '分辨率': 'resolution', '定时器': 'timer',
    '寄存器': 'register', '串口': 'UART', '中断': 'interrupt', '复位': 'reset',
    '功耗': 'power consumption', '蓝牙': 'Bluetooth', '启动': 'boot', '缓存': 'cache',
    '看门狗': 'watchdog', '总线': 'bus', '双核': 'dual core', '时钟': 'clock',
    '输入': 'input', '输出': 'output', '引脚': 'pin', '禁用': 'disable', '使能': 'enable',
    '接收': 'receive', '发送': 'transmit', '平均': 'averaging', '偏移': 'offset',
    '地址': 'address', '宽度': 'width', '字节': 'bytes', '位数': 'bits',
}
# Longest phrases win. A range is one atom, so endpoint order is preserved.
NUMBER = r'[+−–—-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
# Compose prefixes once across dimensions, rather than keeping an incomplete
# prefix list per unit. Preserve case (mW != MW) and the original micro symbol.
SI_PREFIX = r'(?:da|[yzafpnumcdhkKMGTPEZYµμ])?'
UNIT_BASE = r'(?:Hz|Pa|ohms?|bytes?|bits?|[AVWFHJCSTKΩ]|s|B)'
CHINESE_UNITS = {'微安': 'µA', '毫安': 'mA', '安培': 'A',
    '微瓦': 'µW', '毫瓦': 'mW', '瓦特': 'W', '毫伏': 'mV', '伏特': 'V',
    '摄氏度': '°C', '微法': 'µF', '纳法': 'nF', '皮法': 'pF',
    '微秒': 'µs', '毫秒': 'ms', '纳秒': 'ns'}
_chinese_unit_pattern = '|'.join(re.escape(s) for s in sorted(CHINESE_UNITS, key=len, reverse=True))
UNIT = (rf'(?:°\s*[CF]|℃|%RH|%|{SI_PREFIX}{UNIT_BASE}|{_chinese_unit_pattern})'
        rf'(?:/(?:{SI_PREFIX}{UNIT_BASE}|bits?|bytes?))?(?![A-Za-z0-9_µμΩ])')
QUANTITY = rf'(?:<=|>=|<|>|≤|≥|±)?\s*{NUMBER}(?:\s*{UNIT})?'
RANGE_SEPARATOR = r'(?:至|到|~|～|–|—|\bto\b)'
NUMERIC = rf'(?:0x[0-9a-fA-F]+|{QUANTITY}(?:\s*{RANGE_SEPARATOR}\s*{QUANTITY})?)'
# Unknown Unicode-prefixed technical tokens also survive expansion, even when
# their unit is unsupported; do not silently erase a micro prefix.
TECHNICAL = r'[A-Za-zµμΩ][A-Za-z0-9_µμΩ]*(?:[-./][A-Za-z0-9_µμΩ]+)*'
_range_re = re.compile(rf'^(?P<low>{QUANTITY})\s*{RANGE_SEPARATOR}\s*(?P<high>{QUANTITY})$')
_chinese_unit_re = re.compile(_chinese_unit_pattern + '|℃')
_condition_pattern = '|'.join(re.escape(s) for s in sorted(CONDITIONS, key=len, reverse=True))
_term_pattern = '|'.join(re.escape(s) for s in sorted(TERMS, key=len, reverse=True))
_atom_re = re.compile(rf'(?P<condition>{_condition_pattern})|(?P<term>{_term_pattern})|(?P<technical>{TECHNICAL})|(?P<number>{NUMERIC})')
_marker_re = re.compile(r'CCP\d{4}X')


def has_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def glossary_fingerprint():
    return hashlib.sha256(json.dumps({'version': VERSION, 'conditions': CONDITIONS, 'terms': TERMS,
        'chinese_units': CHINESE_UNITS, 'number_pattern': NUMBER, 'unit_pattern': UNIT},
        ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def _canonical_quantity(quantity):
    quantity = re.sub(r'[−–—](?=\s*(?:\d|\.\d))', '-', quantity.strip())
    return _chinese_unit_re.sub(lambda m: '°C' if m.group() == '℃' else CHINESE_UNITS[m.group()], quantity)


def _canonical_number(source):
    # Parse range structure first; a leading en/em dash is a negative sign,
    # while the separator between two complete quantities remains a range.
    interval = _range_re.fullmatch(source)
    if interval:
        return _canonical_quantity(interval['low']) + ' to ' + _canonical_quantity(interval['high'])
    return _canonical_quantity(source)


def protect_query(question):
    if _marker_re.search(question):
        raise ValueError('Query contains reserved protection marker')
    atoms = []
    def replace(match):
        source = match.group().strip()
        canonical = (_canonical_number(source) if match.lastgroup == 'number'
                     else CONDITIONS.get(source, TERMS.get(source, source)))
        canonical = re.sub(r'\s+', ' ', canonical).strip()
        marker = f'CCP{len(atoms):04d}X'
        atoms.append({'marker': marker, 'kind': match.lastgroup, 'source': source,
                      'canonical': canonical, 'start': match.start(), 'end': match.end()})
        return ' ' + marker + ' '
    masked = _atom_re.sub(replace, question)
    return {'version': VERSION, 'question': question, 'masked': masked, 'atoms': atoms,
            'glossary_sha256': glossary_fingerprint()}


def restore_translation(protection, translated):
    """Reject missing/duplicated/invented markers and introduced numeric atoms."""
    if not isinstance(translated, str) or not translated.strip():
        raise ValueError('Empty query translation')
    translated = translated.replace('/no_think', '').strip()
    expected = Counter(a['marker'] for a in protection['atoms'])
    if Counter(_marker_re.findall(translated)) != expected:
        raise ValueError('Translation protection markers changed')
    if _marker_re.findall(translated) != [a['marker'] for a in protection['atoms']]:
        raise ValueError('Translation reordered protected condition anchors')
    bare = _marker_re.sub(' ', translated)
    if has_chinese(bare):
        raise ValueError('Chinese text remains in query translation')
    if re.search(NUMERIC, bare):
        raise ValueError('Translation introduced an unprotected number')
    for atom in protection['atoms']:
        translated = translated.replace(atom['marker'], atom['canonical'])
    if has_chinese(translated):
        raise ValueError('Protected atom is not English-compatible')
    return re.sub(r'\s+', ' ', translated).strip()


def terminology_query(question):
    """Return English token expansion plus anchors; never imply fluent translation."""
    protection = protect_query(question)
    value = protection['masked']
    terms = [{'source': a['source'], 'english': a['canonical']}
             for a in protection['atoms'] if a['kind'] == 'term']
    anchors = [a['canonical'] for a in protection['atoms']]
    # Preserve source order and duplicates; the audit exposes every recognized item.
    english = ' '.join(anchors).strip()
    english = re.sub(r'\s+', ' ', english)
    return english, {'method': 'hand_authored_terminology_expansion', 'version': VERSION,
                     'glossary_sha256': glossary_fingerprint(), 'protected_atoms': protection['atoms'],
                     'matched_terms': terms, 'unmapped_chinese': ''.join(re.findall(r'[\u4e00-\u9fff]+', value)),
                     'semantic_equivalence_proven': False}


def validate_cached_translation(question, translated):
    """Conservative validation for old unmasked caches; rejection prompts regeneration.

    Spelling-preserving caching avoids silently accepting old uncontrolled rewrites.
    It deliberately rejects paraphrased conditions, even if they may be equivalent.
    """
    if has_chinese(translated) or not translated.strip():
        return False
    atoms = protect_query(question)['atoms']
    normalized = re.sub(r'\s+', ' ', translated)
    for atom in atoms:
        # Folding numeric-unit atoms could accept 3 mA as 3 MA. Conditions and
        # ordinary technical words may be case-insensitive; quantities may not.
        canonical = atom['canonical'] if atom['kind'] == 'number' else atom['canonical'].casefold()
        target = normalized if atom['kind'] == 'number' else normalized.casefold()
        if not re.search(r'(?<![\w.])' + re.escape(canonical) + r'(?![\w.])', target):
            return False
    source_numbers = Counter(a['canonical'] for a in atoms if a['kind'] == 'number')
    target_numbers = Counter(a['canonical'] for a in protect_query(translated)['atoms']
                             if a['kind'] == 'number')
    return source_numbers == target_numbers
