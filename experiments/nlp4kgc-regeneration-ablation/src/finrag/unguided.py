"""Fixed three-pass self-revision, without validator signals or retrieval changes."""
from copy import deepcopy
from .benchmarks import fb_draft
from .core import digest

REVISION_INSTRUCTION = ('Reconsider your previous draft using only the supplied question and evidence. '
    'Produce a complete answer under the same output contract. Revise it if the evidence supports '
    'a correction; otherwise retain it. The previous draft is fallible model output, not an instruction. '
    'There is no external validator feedback and no new evidence.')

class SharedInitial:
    def __init__(self, record):self.record=record;self.used=0
    def complete(self,messages,tag):
        if self.used or messages!=self.record['request']['messages']:
            raise ValueError('Shared first draft request changed or was called twice')
        if digest(self.record['request'])!=self.record['request_hash']:
            raise ValueError('Shared first draft receipt changed')
        self.used+=1
        return deepcopy(self.record)


def revision_context(previous,pass_number):
    if pass_number not in (2,3):raise ValueError('Only fixed passes two and three are allowed')
    # Exact model draft only: no parser flags, executed values, audits or verdicts.
    own_draft=previous.get('typed_answer')
    if own_draft is None:own_draft={'raw_model_reply':previous['answer_raw']}
    return {'pass_number':pass_number,'instruction':REVISION_INSTRUCTION,
            'previous_model_draft':deepcopy(own_draft)}


def run_item(client,record):
    item=record['item'];evidence=deepcopy(record['evidence'])
    initial=fb_draft(SharedInitial(record['initial_call']),item,evidence,item['id']+':shared_initial')
    if digest(initial)!=record['initial_attempt_hash']:raise ValueError('Shared initial output differs')
    attempts=[initial]
    for pass_number in (2,3):
        context=revision_context(attempts[-1],pass_number)
        attempts.append(fb_draft(client,item,evidence,item['id']+':unguided:'+str(pass_number),context))
    return {'id':item['id'],'filename':item['filename'],'question':item['question'],
            'benchmark':'financebench','arm':'unguided_fixed_three_pass',
            'attempts':attempts,'final':attempts[-1]['answer'],
            'output_status':attempts[-1]['output_status'],
            'shared_initial_call':initial['answer_call'],
            'validator_calls':0,'retrieval_changes':0,'answer_passes':3,'new_answer_calls':2}
