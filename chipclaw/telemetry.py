"""Request-scoped accounting of actual model calls, including failures."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any
from uuid import uuid4

_ACTIVE_LEDGER = ContextVar('evidenceplan_request_ledger', default=None)
_ACTIVE_STAGE = ContextVar('evidenceplan_model_stage', default='unspecified')


def _nonnegative_number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else None


def _token_count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


@dataclass
class RequestLedger:
    request_id: str = field(default_factory=lambda: uuid4().hex)
    calls: list[dict[str, Any]] = field(default_factory=list)
    started: float = field(default_factory=perf_counter, repr=False)
    finished: float | None = field(default=None, repr=False)
    status: str = 'running'
    error_type: str | None = None

    def to_dict(self):
        calls = [dict(event, ollama_ms=dict(event['ollama_ms'])) for event in self.calls]
        known = [event for event in calls if event['usage_known']]
        by_stage = {}
        for event in calls:
            row = by_stage.setdefault(event['stage'], {'calls': 0, 'failed_calls': 0,
                'usage_unknown_calls': 0, 'known_prompt_tokens': 0,
                'known_output_tokens': 0, 'wall_ms': 0.0})
            row['calls'] += 1
            row['failed_calls'] += event['status'] == 'failed'
            row['usage_unknown_calls'] += not event['usage_known']
            row['known_prompt_tokens'] += event['prompt_tokens'] or 0
            row['known_output_tokens'] += event['output_tokens'] or 0
            row['wall_ms'] += event['wall_ms']
        prompt = sum(event['prompt_tokens'] or 0 for event in calls)
        output = sum(event['output_tokens'] or 0 for event in calls)
        return {'request_id': self.request_id, 'status': self.status,
            'error_type': self.error_type,
            'request_ms': ((self.finished or perf_counter()) - self.started) * 1000,
            'call_count': len(calls), 'failed_calls': sum(event['status'] == 'failed' for event in calls),
            'usage_unknown_calls': len(calls) - len(known), 'usage_complete': len(known) == len(calls),
            'known_prompt_tokens': prompt, 'known_output_tokens': output,
            'known_total_tokens': prompt + output, 'model_call_wall_ms': sum(event['wall_ms'] for event in calls),
            'by_stage': by_stage, 'calls': calls,
            'accounting_note': 'Only actual calls in this request; unknown failed-call usage is not zero. Offline judging is separate.'}


@contextmanager
def capture_request(request_id=None):
    ledger = RequestLedger(request_id=str(request_id) if request_id is not None else uuid4().hex)
    token = _ACTIVE_LEDGER.set(ledger)
    try:
        yield ledger
    except BaseException as error:
        ledger.status = 'failed'
        ledger.error_type = type(error).__name__
        raise
    else:
        ledger.status = 'complete'
    finally:
        ledger.finished = perf_counter()
        _ACTIVE_LEDGER.reset(token)


@contextmanager
def stage(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Stage name must be a nonempty string')
    token = _ACTIVE_STAGE.set(name)
    try:
        yield
    finally:
        _ACTIVE_STAGE.reset(token)


@dataclass
class ModelCall:
    model: str
    response_data: dict | None = None

    def observe_response(self, data):
        if isinstance(data, dict):
            self.response_data = data


@contextmanager
def model_call(model):
    ledger = _ACTIVE_LEDGER.get()
    label = _ACTIVE_STAGE.get()
    started = perf_counter()
    call = ModelCall(model=str(model))
    error_type = None
    try:
        yield call
    except BaseException as error:
        error_type = type(error).__name__
        raise
    finally:
        if ledger is not None:
            data = call.response_data or {}
            prompt = _token_count(data.get('prompt_eval_count'))
            output = _token_count(data.get('eval_count'))
            durations = {}
            for target, key in [('total', 'total_duration'), ('load', 'load_duration'),
                                ('prompt_eval', 'prompt_eval_duration'), ('eval', 'eval_duration')]:
                value = _nonnegative_number(data.get(key))
                durations[target] = value / 1_000_000 if value is not None else None
            ledger.calls.append({'call_id': len(ledger.calls) + 1, 'stage': label,
                'model': str(data.get('model') or model), 'status': 'failed' if error_type else 'complete',
                'error_type': error_type, 'done': data.get('done'), 'done_reason': data.get('done_reason'),
                'wall_ms': (perf_counter() - started) * 1000, 'prompt_tokens': prompt,
                'output_tokens': output, 'usage_known': prompt is not None and output is not None,
                'ollama_ms': durations})
