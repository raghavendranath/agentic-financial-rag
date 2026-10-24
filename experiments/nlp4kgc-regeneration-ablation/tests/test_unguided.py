import json
import unittest
from copy import deepcopy
from finrag.benchmarks import fb_draft,fb_answer_messages
from finrag.core import digest
from finrag.unguided import SharedInitial,run_item,revision_context

ITEM={'id':'synthetic','filename':'filing/report','question':'What is the value?','benchmark':'financebench'}
EVIDENCE=[{'id':'filing/report::0','filename':'filing/report','text':'Value: 5.'}]

def answer(value='5',program='add(5,0)'):
    return json.dumps({'answer_type':'number','value':value,'program':program,'citations':['filing/report::0'],
                      'unit':'scalar','explanation':'Value is 5.'})

def completion(raw,messages):
    request={'messages':messages}
    return {'request':request,'request_hash':digest(request),'content':raw,'elapsed_seconds':0.1,'finish_reason':'stop'}

def make_record(raw=None):
    call=completion(answer() if raw is None else raw,fb_answer_messages(ITEM,EVIDENCE))
    initial=fb_draft(SharedInitial(call),ITEM,EVIDENCE,'initial')
    return {'item':deepcopy(ITEM),'evidence':deepcopy(EVIDENCE),'initial_call':call,'initial_attempt_hash':digest(initial)}

class Spy:
    def __init__(self,replies):self.replies=list(replies);self.messages=[]
    def complete(self,messages,tag):
        self.messages.append(deepcopy(messages));return completion(self.replies.pop(0),messages)

class UnguidedTests(unittest.TestCase):
    def test_exact_shared_initial_not_billed_and_two_new_calls(self):
        client=Spy([answer(),answer()]);row=run_item(client,make_record())
        self.assertEqual(len(client.messages),2);self.assertEqual(len(row['attempts']),3)
        self.assertEqual(row['answer_passes'],3);self.assertEqual(row['validator_calls'],0)
    def test_always_third_pass_even_when_first_was_usable(self):
        client=Spy([answer(),answer('6','add(6,0)')]);row=run_item(client,make_record())
        self.assertEqual(row['final']['value'],'6')
    def test_malformed_last_pass_does_not_fall_back_to_good_earlier(self):
        row=run_item(Spy([answer(),'not JSON']),make_record())
        self.assertIsNone(row['final']);self.assertEqual(row['output_status'],'invalid')
    def test_unknown_or_malformed_initial_still_gets_two_revisions(self):
        for initial in ['not JSON',json.dumps({'answer_type':'unknown','value':'unknown','program':'',
            'unit':None,'explanation':'Missing information','citations':[]})]:
            client=Spy([answer(),answer()]);row=run_item(client,make_record(initial))
            self.assertEqual(len(client.messages),2);self.assertEqual(row['final']['value'],'5')
    def test_validator_and_execution_signals_never_enter_context(self):
        previous={'typed_answer':json.loads(answer()),'answer_raw':answer(),
            'verdict':{'reason':'SECRET_VALIDATOR'},'audit':{'value':'SECRET_AUDIT'},
            'operational_issues':['SECRET_FLAGS'],'draft_issues':['SECRET_PARSER'],
            'calculator':{'value':'SECRET_EXECUTED'},'accepted':True}
        context=revision_context(previous,2);payload=json.dumps(context)
        self.assertEqual(set(context),{'pass_number','instruction','previous_model_draft'})
        self.assertNotIn('SECRET_',payload);self.assertEqual(context['previous_model_draft'],previous['typed_answer'])
    def test_raw_model_fallback_excludes_parser_feedback(self):
        context=revision_context({'typed_answer':None,'answer_raw':'raw model draft','draft_issues':['SECRET']},2)
        self.assertEqual(context['previous_model_draft'],{'raw_model_reply':'raw model draft'})
    def test_same_evidence_in_every_pass_and_request(self):
        client=Spy([answer(),answer()]);row=run_item(client,make_record())
        for a in row['attempts']:self.assertEqual(a['evidence'],EVIDENCE)
        for messages in client.messages:
            payload=json.loads(messages[1]['content']);self.assertEqual(payload['evidence'],EVIDENCE)
            self.assertEqual(set(payload),{'question','report_page','evidence','feedback'})
    def test_pass_three_sees_own_second_draft(self):
        client=Spy([answer('6','add(6,0)'),answer()]);run_item(client,make_record())
        second=json.loads(client.messages[1][1]['content'])
        self.assertEqual(second['feedback']['previous_model_draft']['value'],'6')
    def test_pass_numbers_frozen_not_best_of_three(self):
        client=Spy([answer(),answer()]);run_item(client,make_record())
        self.assertEqual([json.loads(m[1]['content'])['feedback']['pass_number'] for m in client.messages],[2,3])
        with self.assertRaises(ValueError):revision_context({},4)
    def test_initial_request_or_output_tampering_detected(self):
        record=make_record();record['item']['question']='Changed question'
        with self.assertRaises(ValueError):run_item(Spy([]),record)
        record=make_record();record['initial_attempt_hash']='bad'
        with self.assertRaises(ValueError):run_item(Spy([]),record)
    def test_initial_receipt_integrity_and_single_use(self):
        record=make_record();shared=SharedInitial(record['initial_call'])
        shared.complete(fb_answer_messages(ITEM,EVIDENCE),'one')
        with self.assertRaises(ValueError):shared.complete(fb_answer_messages(ITEM,EVIDENCE),'two')
        record['initial_call']['request_hash']='bad'
        with self.assertRaises(ValueError):SharedInitial(record['initial_call']).complete(fb_answer_messages(ITEM,EVIDENCE),'one')
    def test_input_record_not_mutated(self):
        record=make_record();before=deepcopy(record)
        run_item(Spy([answer(),answer()]),record);self.assertEqual(record,before)

if __name__=='__main__':unittest.main()
