"""Narrowly separate a denied user assertion from positive numeric facts."""
import math
import re

def mask_denied_user_quantities(text,question,quantities):
    """No outside value may be exempted; every positive correction remains checked."""
    if not question:return text,[]
    original=quantities(question);masked=text;records=[]
    boundary=re.compile(r'[。！？;；]|(?<!\d)[.!?](?!\d)')
    clauses=boundary.split(text)
    for clause in clauses:
        denial=re.search(r'错误|有误|不正确|不成立|不支持|\bincorrect\b|\bwrong\b|not\s+(?:correct|supported|true)',clause,re.I)
        attributed=re.search(r'声称|认为|说法|断言|\bclaim(?:ed|s)?\b|\bassert(?:ed|s)?\b',clause,re.I)
        if not (denial and attributed):continue
        contrast=re.search(r'但|实际|应为|正确值|\bbut\b|\bcorrect\s+(?:value|minimum|maximum)',clause,re.I)
        reference=clause[:contrast.start()] if contrast else clause
        for q in quantities(reference):
            if not any(x['dimension']==q['dimension'] and math.isclose(x['value'],q['value'],rel_tol=1e-9,abs_tol=1e-12) for x in original):continue
            # Replace only this explicitly attributed and denied clause, never
            # another occurrence of the same value in a positive assertion.
            replacement=reference.replace(q['raw'],'[denied user value]')
            if replacement!=reference:
                masked=masked.replace(reference,replacement,1)
                records.append({'code':'denied_original_premise','quantity':q,
                    'note':'Only the denied user assertion is exempted; semantic correction still requires source support.'})
                reference=replacement
    return masked,records
