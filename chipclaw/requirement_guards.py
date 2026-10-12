"""Fail-closed explicit default-condition binding; not general entailment."""
import re

DEFAULT_REQUEST=re.compile(r'\b(default|factory|reset value)\b|默认|出厂|上电默认',re.I)
DEFAULT_SOURCE=re.compile(r'\b(default|factory|power[- ]?on|POR|reset|start[- ]?up)\b',re.I)

def check_requirement_context(requirement,output):
    if not DEFAULT_REQUEST.search(requirement):return []
    return [{'claim_index':i,'code':'missing_explicit_default_binding',
             'note':'General possible settings do not establish a factory/default value; implicit table headers may be over-refused.'}
            for i,claim in enumerate(output.get('claims',[])) if not DEFAULT_SOURCE.search(claim['quote'])]
