"""Label-free FinanceBench typed output and transport checks.

``value`` is the model's final DISPLAY value, never a hidden/raw program value.
For unit=percent_fraction the arithmetic program returns a decimal fraction and
value must include '%'. For unit=percentage_points, value is a plain point-unit
scalar. Unit is explicitly declared: merely mentioning '%' in a question does
not choose an output unit. The source validator must check the declared unit
and amount unit_label against the question/source; this module cannot infer
financial semantics or establish factual support.

Parser preserves declared value, program output, display output, raw response
and every transport normalization. A mismatch becomes an operational issue;
no field is selected according to a reference answer. No labels are accepted.
"""
from __future__ import annotations

import json
import math
import re
from decimal import Decimal

from .calculator import execute
from .compatibility import flatten_program, render_numeric

UNITS = frozenset({'amount', 'scalar', 'percent_fraction', 'percentage_points'})
ANSWER_TYPES = frozenset({'number', 'boolean', 'text', 'unknown'})
_TAILS = (
    re.compile(r'\s*</parameter>\s*'),
    re.compile(r'\s*</｜DSML｜ parameter>\s*</｜DSML｜ invoke>\s*</｜DSML｜ calls>\s*'),
)
_SCALAR = re.compile(r'[+-]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')


def decode_provider_object(text):
    """Return (one object, normalization records); reject unrecognized tails."""
    if not isinstance(text, str):
        raise ValueError('Expected completion text')
    value = text.strip()
    changes = []
    if value.startswith('```json') and value.endswith('```'):
        value = value[7:-3].strip()
        changes.append({'kind': 'markdown_json_fence'})
    elif value.startswith('```') and value.endswith('```'):
        value = value[3:-3].strip()
        changes.append({'kind': 'markdown_fence'})
    obj, end = json.JSONDecoder().raw_decode(value)
    if not isinstance(obj, dict):
        raise ValueError('Expected one JSON object')
    tail = value[end:]
    if tail.strip():
        if not any(pattern.fullmatch(tail) for pattern in _TAILS):
            raise ValueError('Unexpected trailing completion content')
        changes.append({'kind': 'provider_closing_tags', 'removed_tail': tail})
    return obj, changes


def object_from_reply(text):
    """Object-only compatibility entry point; prefer preserving metadata."""
    return decode_provider_object(text)[0]


def present_numeric(raw_value, unit, question):
    """Format a program result only from explicit unit and question rounding."""
    if unit not in UNITS:
        raise ValueError('Missing or invalid explicit numeric unit')
    numeric = Decimal(str(raw_value))
    if not numeric.is_finite():
        raise ValueError('Non-finite program result')
    if unit == 'percent_fraction':
        return render_numeric(str(numeric * 100), question) + '%'
    return render_numeric(str(raw_value), question)


def _display_number(value, unit):
    text = str(value).strip().replace('−', '-')
    if unit == 'percent_fraction':
        if not text.endswith('%'):
            raise ValueError('percent_fraction display value must include %')
        text = text[:-1].strip()
    elif text.endswith('%'):
        raise ValueError('Only percent_fraction display values may include %')
    if not _SCALAR.fullmatch(text):
        raise ValueError('Declared numerical value must be a whole display scalar')
    number = float(text.replace(',', ''))
    if not math.isfinite(number):
        raise ValueError('Non-finite display scalar')
    return number


def is_abstention(answer):
    """Exact contract abstention is recognized independent of declared type."""
    return bool(answer and (answer.get('answer_type') == 'unknown' or
                            str(answer.get('value', '')).strip().casefold() == 'unknown'))


def parse_typed_answer(raw_reply, question):
    """Return answer/calculator/issues/raw metadata without silently repairing data.

    Numeric consistency uses explicit question-requested rounding first. A
    1e-12 relative/absolute float allowance handles only the existing executor's
    binary arithmetic representation, not benchmark/reference precision.
    ``answer['value']`` remains the declared final value even on disagreement.
    """
    result = {'answer': None, 'calculator': {'status': 'no_answer'},
              'operational_issues': [], 'normalization': [], 'raw_reply': raw_reply}
    try:
        obj, changes = decode_provider_object(raw_reply)
        result['normalization'] = changes
        required = {'answer_type', 'value', 'program', 'citations', 'explanation', 'unit'}
        if not required.issubset(obj):
            raise ValueError('Missing typed answer fields: ' + ', '.join(sorted(required - set(obj))))
        if obj['answer_type'] not in ANSWER_TYPES:
            raise ValueError('Invalid answer_type')
        if obj['answer_type'] == 'boolean' and type(obj['value']) is bool:
            original_boolean = obj['value']
            obj = {**obj, 'value': 'yes' if original_boolean else 'no'}
            result['normalization'].append({'kind': 'json_boolean_value',
                'original_value': original_boolean, 'normalized_value': obj['value']})
        if not isinstance(obj['value'], (str, int, float)) or isinstance(obj['value'], bool):
            raise ValueError('Invalid final value type')
        if not isinstance(obj['citations'], list) or not all(isinstance(x, str) for x in obj['citations']):
            raise ValueError('Invalid citations field')
        if not isinstance(obj['explanation'], str):
            raise ValueError('Invalid explanation field')
        if obj['answer_type'] != 'number' and obj['program'] == []:
            obj = {**obj, 'program': ''}
            result['normalization'].append({'kind': 'empty_non_numeric_program_list'})
        if not isinstance(obj['program'], str):
            raise ValueError('Program must be a string')
        if obj.get('unit_label') is not None and not isinstance(obj['unit_label'], str):
            raise ValueError('unit_label must be a string or null')
        answer = {key: obj[key] for key in required}
        answer['value'] = str(obj['value'])
        if 'unit_label' in obj:
            answer['unit_label'] = obj['unit_label']
        result['answer'] = answer
        issues = result['operational_issues']
        if is_abstention(answer):
            result['calculator'] = {'status': 'abstained'}
            issues.append('No supported usable answer: explicit abstention.')
            if answer['answer_type'] == 'unknown' and answer['value'].strip().casefold() != 'unknown':
                issues.append('unknown answer_type requires value unknown.')
            return result
        if answer['answer_type'] == 'number':
            if answer['unit'] not in UNITS:
                issues.append('Numeric answer requires an explicit valid unit.')
                result['calculator'] = {'status': 'invalid_unit'}
                return result
            if answer['unit'] == 'amount' and not str(answer.get('unit_label') or '').strip():
                issues.append('Numeric amount requires a nonempty unit_label checked against the question and source.')
            try:
                compiled = flatten_program(answer['program'])
                calculated = execute(compiled)
                # A comparison program is Boolean, not a numeric display value.
                displayed = present_numeric(calculated['value'], answer['unit'], question)
                result['calculator'] = {'status': 'executed', **calculated,
                    'declared_value': answer['value'], 'raw_value': calculated['value'],
                    'display_value': displayed, 'compiled_program': compiled,
                    'original_program': answer['program']}
                declared_number = _display_number(answer['value'], answer['unit'])
                display_number = _display_number(displayed, answer['unit'])
                if not math.isclose(declared_number, display_number, rel_tol=1e-12, abs_tol=1e-12):
                    issues.append('Declared final value disagrees with executed displayed value.')
            except (ValueError, ArithmeticError, TypeError) as exc:
                if result['calculator']['status'] != 'executed':
                    result['calculator'] = {'status': 'invalid_program'}
                issues.append('Numeric output contract failed: ' + str(exc))
        else:
            result['calculator'] = {'status': 'not_applicable'}
            if answer['unit'] is not None:
                issues.append('Nonnumeric answer unit must be null.')
            if answer['answer_type'] == 'boolean' and answer['program']:
                try:
                    compiled = flatten_program(answer['program'])
                    calculated = execute(compiled)
                    result['calculator'] = {'status': 'executed', **calculated,
                        'declared_value': answer['value'], 'raw_value': calculated['value'],
                        'display_value': calculated['value'], 'compiled_program': compiled,
                        'original_program': answer['program']}
                    if calculated['value'] not in {'yes', 'no'}:
                        issues.append('Boolean program must return yes or no, not a numerical result.')
                    elif answer['value'].strip().lower() != calculated['value']:
                        issues.append('Declared Boolean final value disagrees with executed program result.')
                except (ValueError, ArithmeticError, TypeError) as exc:
                    result['calculator'] = {'status': 'invalid_program'}
                    issues.append('Boolean output contract failed: ' + str(exc))
            elif answer['program']:
                issues.append('Nonnumeric answer must use an empty program.')
            if answer['answer_type'] == 'boolean' and answer['value'].strip().lower() not in {'yes', 'no'}:
                issues.append('Boolean final value must be yes or no.')
            if not answer['value'].strip():
                issues.append('Empty final answer value.')
    except (ValueError, TypeError, KeyError) as exc:
        result['answer'] = None
        result['operational_issues'].append('Malformed typed answer: ' + str(exc))
    return result


def validated_output(answer, accepted, issues=()):
    """Expose validation status without claiming rejected values are validated.

    Preserve a provisional answer for reproducibility. Product callers can
    suppress it by consulting usable/validated; scoring policy stays explicit.
    """
    status = ('invalid' if answer is None else 'abstained' if is_abstention(answer)
              else 'validated' if accepted and not issues else 'unvalidated')
    return {'answer': answer, 'status': status,
            'usable': status in {'validated', 'unvalidated'},
            'validated': status == 'validated'}
