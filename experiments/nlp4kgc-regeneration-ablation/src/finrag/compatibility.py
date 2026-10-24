"""Conservative, label-free compatibility fixes; original experiments stay frozen."""
import json
import re
from decimal import Decimal, ROUND_HALF_EVEN

OPS = {'add', 'subtract', 'multiply', 'divide', 'exp', 'greater'}
TOKEN = re.compile(r'\s*(?:(add|subtract|multiply|divide|exp|greater)|(#\d+|const_m1|const_[+-]?(?:\d+(?:\.\d*)?|\.\d+)|[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?)|([(),]))')

def flatten_program(program):
    """Compile nested calls to the existing bounded executor's flat steps.

    #i always means the i-th original top-level step, including inside a nested
    expression. No Python eval, names, indexing, strings or arbitrary calls.
    """
    if not isinstance(program, str) or not program or len(program) > 2000:
        raise ValueError('Missing or oversized arithmetic program')
    tokens=[];pos=0
    while pos < len(program.rstrip()):
        m=TOKEN.match(program,pos)
        if not m:raise ValueError('Invalid arithmetic token')
        tokens.append(next(x for x in m.groups() if x is not None));pos=m.end()
    out=[];top=[];cursor=0
    def take(expected=None):
        nonlocal cursor
        if cursor>=len(tokens):raise ValueError('Incomplete arithmetic expression')
        token=tokens[cursor];cursor+=1
        if expected is not None and token!=expected:raise ValueError('Unexpected arithmetic token')
        return token
    def expr(depth=0):
        if depth>16:raise ValueError('Nested expression too deep')
        token=take()
        if token in OPS:
            take('(');a=expr(depth+1);take(',');b=expr(depth+1);take(')')
            if len(out)>=24:raise ValueError('Too many arithmetic steps')
            out.append(f'{token}({a}, {b})');return '#'+str(len(out)-1)
        if token.startswith('#'):
            i=int(token[1:])
            if i>=len(top):raise ValueError('Forward arithmetic reference')
            return top[i]
        if token in ('(',')',','):raise ValueError('Expected arithmetic operand')
        return token
    while cursor<len(tokens):
        if tokens[cursor] not in OPS:raise ValueError('Expected top-level call')
        top.append(expr())
        if cursor<len(tokens):take(',')
        if cursor==len(tokens) and tokens[-1]==',':raise ValueError('Trailing separator')
    return ', '.join(out)

def normalize_reply(raw):
    """Accept [] only for an unused non-numeric program field."""
    obj=json.loads(raw)
    changes=[]
    if isinstance(obj,dict) and obj.get('answer_type') in ('text','boolean','unknown') and obj.get('program')==[]:
        obj={**obj,'program':''};changes.append('empty_non_numeric_program_list')
    return obj,changes

def requested_decimals(question):
    words={'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6}
    m=re.search(r'round\b[^.?!]{0,65}?\bto\s+(zero|one|two|three|four|five|six|\d+)\s+decimal\s+places?',question,re.I)
    if not m:return None
    value=m.group(1).lower();n=words[value] if value in words else int(value)
    return n if 0<=n<=6 else None

def render_numeric(value,question):
    """Honor explicit question rounding; never inspect the reference answer.

    No automatic factor-100 choice, sign change, number extraction from prose,
    implicit reference-precision tolerance or reuse of model-stated arithmetic.
    """
    n=requested_decimals(question)
    if n is None:return str(value)
    d=Decimal(str(value))
    if not d.is_finite():raise ValueError('Non-finite answer')
    return format(d.quantize(Decimal(1).scaleb(-n),rounding=ROUND_HALF_EVEN),'f')

def financebench_presentation(value,question):
    """Input program follows decimal-ratio contract; output follows question units."""
    if re.search(r'%|\bpercent(?:s|age)?\b',question,re.I):
        return render_numeric(str(Decimal(str(value))*100),question)+'%'
    return render_numeric(value,question)
