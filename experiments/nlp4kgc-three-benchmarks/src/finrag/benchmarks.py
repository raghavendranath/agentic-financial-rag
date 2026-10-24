"""Task adapters: typed FinanceBench QA and genuine own-history ConvFinQA."""
import re
from .core import (ANSWER_SYSTEM,VALIDATOR_SYSTEM,answer_messages,validator_messages,canonical,
                   object_from_reply,parse_answer,parse_verdict,resolve_citations,valid_citations,correct)
from .calculator import execute
from . import experiment as numeric

FB_ANSWER = ANSWER_SYSTEM.replace('Return exactly one JSON object with keys value, program, citations, explanation.',
    'Return exactly one JSON object with keys answer_type, value, program, citations, explanation.\nanswer_type: number, boolean, text, or unknown. Answer the entire question; do not discard requested explanation or multiple requested facts.').replace(
    'value: the final numerical answer as a string, or yes/no, or unknown. Do not round.',
    'value: a concise final answer string, or yes/no, or unknown. For a single numerical answer use its numeric scalar. For multiple facts or qualitative questions give a concise complete text answer. Do not round calculated numbers.').replace(
    'Always provide a program for an answer; for unknown use an empty program.',
    'Always provide a program for a numerical answer. For directly stated Boolean/text answers and unknown, use an empty program.')

FB_VALIDATOR = VALIDATOR_SYSTEM + """
This benchmark also includes qualitative, Boolean and multi-fact questions.
Do not reject a correctly supported text/Boolean answer merely for lacking arithmetic.
In addition to the standard keys return audit_answer (a concise answer string derived from the source)
and audit_quote (a short exact substring of the supplied evidence supporting the audit).
For number candidates supply an executable audit_program, following the decimal ratio contract.
For text/Boolean candidates audit_program may be empty; check all requested facts and negations.
Accept only if the candidate is equivalent to the source-derived audit answer and its citations support it.
"""

def fb_answer_messages(item,evidence,feedback=None):
    messages=answer_messages(item,evidence,feedback)
    messages[0]['content']=FB_ANSWER
    return messages

def fb_validator_messages(item,evidence,answer):
    messages=validator_messages(item,evidence,answer)
    messages[0]['content']=FB_VALIDATOR
    return messages

def fb_draft(client,item,evidence,tag,feedback=None):
    response=client.complete(fb_answer_messages(item,evidence,feedback),tag)
    parsed=None;calculator={'status':'no_answer'}
    if response.get('finish_reason')=='stop':
        try:
            raw=object_from_reply(response['content'])
            parsed=parse_answer(response['content'])
            kind=raw.get('answer_type')
            if not parsed or kind not in ('number','boolean','text','unknown'):parsed=None
            else:
                parsed={**parsed,'answer_type':kind}
                if kind=='number':
                    try:
                        result=execute(parsed['program'])
                        calculator={'status':'executed','model_stated_value':parsed['value'],**result}
                        parsed={**parsed,'value':result['value']}
                    except (ValueError,ArithmeticError):
                        calculator={'status':'invalid_program'};parsed={**parsed,'value':'unknown'}
                elif kind=='unknown':calculator={'status':'abstained'}
                elif kind=='boolean' and parsed['value'].strip().lower() not in ('yes','no'):
                    parsed=None
                else:calculator={'status':'not_applicable'}
        except (ValueError,TypeError,KeyError):parsed=None
    return {'answer':parsed,'answer_raw':response['content'],'evidence':evidence,'calculator':calculator,
            'answer_call':response['request_hash'],'answer_seconds':response['elapsed_seconds'],'answer_finish_reason':response.get('finish_reason')}

def fb_validate(client,item,attempt,tag):
    original=attempt['answer']
    resolved,problems=resolve_citations(original['citations'] if original else [],attempt['evidence'])
    proposed=({**original,'citations':resolved if not problems else original['citations']} if original else {'malformed_reply':attempt['answer_raw']})
    response=client.complete(fb_validator_messages(item,attempt['evidence'],proposed),tag)
    verdict=parse_verdict(response['content']) if response.get('finish_reason')=='stop' else None
    if verdict and (not isinstance(verdict.get('audit_answer'),str) or not isinstance(verdict.get('audit_quote'),str)):verdict=None
    cited=proposed['citations'] if original else []
    valid=valid_citations(cited,attempt['evidence']);issues=[];audit={'status':'invalid_verdict'}
    if not original:issues.append('Malformed answer JSON or invalid answer_type.')
    if not valid:issues.append('Invalid or missing citations: '+str(problems))
    if not original or original['answer_type']=='unknown' or attempt['calculator']['status'] in ('invalid_program','abstained','no_answer'):
        issues.append('No supported usable answer; derive it only if source evidence permits.')
    if verdict:
        failed=[k for k,v in verdict['checks'].items() if not v]
        if failed:issues.append('Failed financial checks: '+', '.join(failed))
        audit_cites,audit_problems=resolve_citations(verdict['audit_citations'],attempt['evidence'])
        if not audit_cites or audit_problems:issues.append('Invalid audit citations.')
        quote=' '.join(verdict['audit_quote'].split())
        cited_chunks=[e for e in attempt['evidence'] if any(c==e['id'] or c.startswith(e['id']+':') for c in audit_cites)]
        if not quote or not any(quote in ' '.join(e['text'].split()) for e in cited_chunks):issues.append('Audit quote is not an exact substring of cited evidence.')
        if original and original['answer_type']=='number':
            try:
                audit={'status':'executed',**execute(verdict['audit_program']),'citations':audit_cites,'quote':verdict['audit_quote']}
                if not correct(audit['value'],original['value']):issues.append('Source-derived audit result '+audit['value']+' disagrees with candidate; check target, formula, years and units.')
            except (ValueError,ArithmeticError):issues.append('Audit program is invalid or missing.')
        else:
            audit={'status':'text_audit','value':verdict['audit_answer'],'quote':verdict['audit_quote'],'citations':audit_cites}
            if not verdict['audit_answer'].strip():issues.append('No source-derived audit answer.')
        if not verdict['accept']:issues.append('Validator rejected: '+verdict['reason'])
    else:issues.append('Invalid validator JSON.')
    return {**attempt,'answer':proposed if original else None,'original_answer':original,'verdict':verdict,
            'accepted':bool(verdict and verdict['accept'] and not issues),'operational_issues':issues,'audit':audit,
            'citation_resolution':{'original':original['citations'] if original else [],'resolved':resolved,'problems':problems},
            'citation_ids_valid':valid,'validator_call':response['request_hash'],'validator_seconds':response['elapsed_seconds'],
            'validator_raw':response['content'],'validator_finish_reason':response.get('finish_reason')}

def trajectory(client,index,item,first,config,refine=True):
    if item['benchmark']!='financebench':return numeric.trajectory(client,index,item,first,config,refine)
    attempts=[first]
    for n in range(1,config['max_attempts']):
        previous=attempts[-1]
        if previous['accepted']:break
        verdict=previous['verdict'];query=(verdict['retrieval_query'].strip() if verdict else '') or item['question']
        evidence=index.retrieve(item['question']+'\n'+query,config['top_k'],filename=item['filename']) if refine else first['evidence']
        feedback={'reason':verdict['reason'] if verdict else 'Invalid validation output.',
                  'previous_answer':previous['answer'],'operational_issues':previous['operational_issues'],'audit':previous['audit'],
                  'allowed_chunk_ids':[e['id'] for e in evidence],
                  'evidence_set_unchanged':{e['id'] for e in evidence}=={e['id'] for e in previous['evidence']},
                  'history':[{'answer':a['answer'],'issues':a['operational_issues']} for a in attempts],
                  'operational_feedback':'Fix the actual failed check using source evidence. Preserve correct calculations when only citation syntax failed. Abstain when required evidence is missing.'}
        tag=item['id']+':full:'+str(n+1)
        a=fb_draft(client,item,evidence,tag+':answer',feedback)
        attempts.append(fb_validate(client,item,a,tag+':validate'))
    return {'attempts':attempts,'final':attempts[-1]['answer'],'accepted':attempts[-1]['accepted']}

def draft(*args,**kwargs):
    return (fb_draft if args[1]['benchmark']=='financebench' else numeric.draft)(*args,**kwargs)

def validate(*args,**kwargs):
    return (fb_validate if args[1]['benchmark']=='financebench' else numeric.validate)(*args,**kwargs)

def history_entry(question,answer):
    return {'question':question,'answer':answer['value'] if answer else 'unknown','citations':answer['citations'] if answer else []}

def conversational_item(conversation,turn,history):
    import copy
    # Only current/past question and generated history, never future questions or labels.
    query=conversation['questions'][turn]
    return {'id':conversation['id']+':turn:'+str(turn),'conversation_id':conversation['id'],'filename':conversation['filename'],
            'question':query,'history':copy.deepcopy(history),'benchmark':'convfinqa'}

def retrieval_query(item):
    # Resolve follow-ups using prior questions while retaining the current target.
    if item.get('history'):
        return item['question']+'\nPrior questions: '+' | '.join(h['question'] for h in item['history'][-3:])
    return item['question']
