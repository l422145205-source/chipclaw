"""Conservative, typed checks for datasheet claims.

These reject contradictions detectable without an LLM. A passing check is not
a proof: relational/conditional errors still require semantic review.
"""
import math
import re

# Case matters: MB/s is bytes/second, Mbps is bits/second.
UNIT = {
    'GHz':('frequency',1e9), 'MHz':('frequency',1e6), 'kHz':('frequency',1e3), 'Hz':('frequency',1),
    'V':('voltage',1), 'mV':('voltage',1e-3),
    's':('time',1), 'ms':('time',1e-3), 'us':('time',1e-6), 'µs':('time',1e-6), 'μs':('time',1e-6),
    'Mbit/s':('bit_rate',1e6), 'Mbits/s':('bit_rate',1e6), 'Mbit':('bit_capacity',1e6),
    'Mbps':('bit_rate',1e6), 'Mbs':('bit_rate',1e6), 'MB/s':('bit_rate',8e6),
    'KB':('memory_kbyte',1), 'kB':('memory_kbyte',1), 'Kbytes':('memory_kbyte',1),
    'MB':('memory_mbyte',1), 'Mbytes':('memory_mbyte',1),
    'bit':('bit_width',1), 'bits':('bit_width',1), '位':('bit_width',1),
    'A':('current',1), 'mA':('current',1e-3), 'µA':('current',1e-6), 'μA':('current',1e-6),
    'µV':('voltage',1e-6), 'μV':('voltage',1e-6),
    '°C':('temperature',1), 'm°C':('temperature',1e-3),
    'Pa':('pressure',1), 'kPa':('pressure',1e3), 'hPa':('pressure',1e2),
    'Ω':('resistance',1), 'kΩ':('resistance',1e3),
    'F':('capacitance',1), 'µF':('capacitance',1e-6), 'μF':('capacitance',1e-6),
    'nF':('capacitance',1e-9), 'pF':('capacitance',1e-12),
    '%':('percentage',1), 'ksps':('sample_rate',1e3), 'MSps':('sample_rate',1e6),
}
NUMBER=r'[+−–-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+\-]?\d+)?'
UNIT_PATTERN='|'.join(re.escape(u) for u in sorted(UNIT,key=len,reverse=True))
QUANTITY=re.compile(r'(?<![A-Za-z0-9_+−–-])(?P<value>'+NUMBER+r')\s*[-‑]?\s*(?P<unit>'+UNIT_PATTERN+r')(?![A-Za-z/])')
RANGE=re.compile(r'(?<![A-Za-z0-9_])(?P<low>'+NUMBER+r')\s*(?P<left_unit>'+UNIT_PATTERN+r')?\s*'
                 r'(?:to|through|and|…|\.{2,3}|–|—|-|至|到)\s*(?P<high>'+NUMBER+r')\s*'
                 r'(?P<right_unit>'+UNIT_PATTERN+r')(?![A-Za-z/])',re.I)
HEX=re.compile(r'(?<![A-Za-z0-9_])0x[0-9a-fA-F]+(?![A-Za-z0-9_])')
HEX_SUFFIX=re.compile(r'(?<![A-Za-z0-9_])[0-9][0-9a-fA-F]{1,7}[hH](?![A-Za-z0-9_])')
REGISTER=re.compile(r'(?<![A-Za-z0-9_])[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+(?![A-Za-z0-9_])')


def quantities(text):
    result=[]
    for m in QUANTITY.finditer(text):
        dimension,scale=UNIT[m['unit']]
        result.append({'raw':m.group(0),'dimension':dimension,'value':number(m['value'])*scale})
    # A unit on the right can explicitly apply to both range endpoints.
    for interval in ranges(text):
        for endpoint in ['low','high']:
            q={'raw':interval['raw'],'dimension':interval['dimension'],'value':interval[endpoint]}
            if not any(x['dimension']==q['dimension'] and math.isclose(x['value'],q['value'],rel_tol=1e-9,abs_tol=1e-12) for x in result):result.append(q)
    return result


def number(value):return float(value.replace('−','-').replace('–','-'))


def ranges(text):
    result=[]
    for m in RANGE.finditer(text):
        if re.search(r'\band\b',m.group(0),re.I) and not re.search(r'\bbetween\s*$',text[max(0,m.start()-16):m.start()],re.I):continue
        right=m['right_unit'];left=m['left_unit'] or right
        # Case remains meaningful (mV vs MV); regex case folding never invents a unit.
        if right not in UNIT or left not in UNIT or UNIT[right][0]!=UNIT[left][0]:continue
        low,high=number(m['low'])*UNIT[left][1],number(m['high'])*UNIT[right][1]
        if low<=high:result.append({'raw':m.group(0),'dimension':UNIT[right][0],'low':low,'high':high})
    return result


def check_claim(text, quote):
    errors=[]
    reference=quantities(quote)
    intervals=ranges(quote)
    for q in quantities(text):
        same_dimension=[x for x in reference if x['dimension']==q['dimension']]
        # Without a comparable quantity, defer to semantic verification (tables,
        # number words, implied units). Do not invent an unsupported numeric match.
        in_interval=any(i['dimension']==q['dimension'] and i['low']-1e-12<=q['value']<=i['high']+1e-12 for i in intervals)
        if same_dimension and not in_interval and not any(math.isclose(x['value'],q['value'],rel_tol=1e-9,abs_tol=1e-12) for x in same_dimension):
            errors.append({'code':'quantity_mismatch','claim':q,'source_values':same_dimension})
    source_hex={int(x,16) for x in HEX.findall(quote)}|{int(x[:-1],16) for x in HEX_SUFFIX.findall(quote)}
    for addr in HEX.findall(text)+HEX_SUFFIX.findall(text):
        if int(addr[:-1] if addr.lower().endswith('h') else addr,16) not in source_hex:
            errors.append({'code':'hex_not_supported','value':addr})
    source_registers=set(REGISTER.findall(quote))
    for register in REGISTER.findall(text):
        if register not in source_registers:
            errors.append({'code':'register_not_supported','value':register})
    return errors


def check_output(output):
    return [{'claim_index':i,**error} for i,c in enumerate(output.get('claims',[]))
            for error in check_claim(c['text'],c['quote'])]
