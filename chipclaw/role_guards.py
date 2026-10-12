"""Narrow, source-derived role binding for explicit clock-frequency ceilings.

Supports literal English maximum-frequency statements and Chinese AHB/APB
ceiling claims. Unknown grammar is deferred, never treated as proof. No chip
parameters, gold labels or other evidence documents are consulted.
"""
import math
import re
from .guards import UNIT

ALIASES = re.compile(r'(?P<alias>(?:high|low)[- ]speed APB)\s*\((?P<symbol>APB[1-9]\d*)\)', re.I)
CEILING = re.compile(
    r'\bmaximum(?: allowed)? frequency of (?P<objects>[^.!?]+?) is '
    r'(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>GHz|MHz|kHz|Hz)\b')
CLAIM = re.compile(
    r'(?P<objects>(?:AHB|APB[1-9]\d*)(?:(?:和|、|及|与|/)\s*(?:AHB|APB[1-9]\d*))*)'
    r'\s*(?:的)?(?:最高(?:频率|时钟频率)?|最大(?:频率|时钟频率))\s*(?:为|是)?\s*'
    r'(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>GHz|MHz|kHz|Hz)\b')
SYMBOL = re.compile(r'\b(?:AHB|APB[1-9]\d*)\b')


def source_bindings(quote, source):
    """Bind values in the actual quote; resolve aliases only in its own source."""
    source = ' '.join(source.split())
    quote = ' '.join(quote.split())
    if quote not in source:
        return []
    aliases = {}
    for match in ALIASES.finditer(source):
        key = match['alias'].lower()
        aliases.setdefault(key, []).append((match['symbol'].upper(), match.group(0)))
    facts = []
    for match in CEILING.finditer(quote):
        objects = match['objects']
        used = []
        for alias, definitions in aliases.items():
            # Conflicting alias definitions are unsupported, not guessed.
            symbols = {symbol for symbol, _ in definitions}
            if len(symbols) != 1:
                continue
            if re.search(re.escape(alias), objects, re.I):
                symbol = next(iter(symbols))
                objects = re.sub(re.escape(alias), symbol, objects, flags=re.I)
                used.append(definitions[0][1])
        for symbol in SYMBOL.findall(objects):
            facts.append({'object': symbol, 'attribute': 'maximum_frequency',
                          'value_hz': float(match['value']) * UNIT[match['unit']][1],
                          'source_quote': match.group(0), 'alias_quotes': used})
    return facts


def check_role_claim(text, quote, source):
    facts = source_bindings(quote, source)
    errors = []
    for match in CLAIM.finditer(text):
        value = float(match['value']) * UNIT[match['unit']][1]
        for symbol in SYMBOL.findall(match['objects']):
            bound = [fact for fact in facts if fact['object'] == symbol]
            # A role with multiple different ceilings may have conditions we
            # have not parsed; defer rather than collapse those conditions.
            expected = {fact['value_hz'] for fact in bound}
            if len(expected) == 1 and not math.isclose(value, next(iter(expected)), rel_tol=1e-9):
                errors.append({'code': 'role_bound_quantity_mismatch', 'object': symbol,
                               'claim_value_hz': value, 'source_bindings': bound})
    return errors


def check_role_output(output, evidence):
    sources = {e['id']: e['text'] for e in evidence}
    return [{'claim_index': i, **error}
            for i, claim in enumerate(output.get('claims', []))
            for error in check_role_claim(claim['text'], claim['quote'],
                                          sources.get(claim['evidence_id'], ''))]
