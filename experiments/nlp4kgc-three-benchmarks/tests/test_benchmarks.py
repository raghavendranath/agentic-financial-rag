import copy
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
from finrag import benchmarks as b
from finrag.core import answer_messages,validator_messages,ANSWER_SYSTEM
from test_experiment import FakeClient,FakeIndex,answer,verdict
from run_benchmark import answer_conversation,answer_one
import threading,json

def fb_answer(value,kind='text'):
    return {**answer(value),'answer_type':kind,'program':'' if kind!='number' else 'add('+str(value)+', 0)'}

def fb_verdict(value,quote='110'):
    return {**verdict(True),'audit_answer':str(value),'audit_quote':quote,'audit_program':'add('+str(value)+', 0)' if str(value).isdigit() else ''}

class BenchmarkTests(unittest.TestCase):
    def test_no_gold_or_future_turn_in_conversational_item(self):
        conv={'id':'c','filename':'co/page','questions':['first?','second?','SECRET_FUTURE'],'answers':['GOLD']}
        history=[{'question':'first?','answer':'10','citations':['table_1']}]
        item=b.conversational_item(conv,1,history)
        payload=json.dumps(answer_messages(item,[]))
        self.assertIn('second?',payload);self.assertIn('first?',payload)
        self.assertNotIn('SECRET_FUTURE',payload);self.assertNotIn('GOLD',payload)
        history[0]['answer']='changed';self.assertEqual(item['history'][0]['answer'],'10')

    def test_finqa_generator_prompt_unchanged_without_history(self):
        item={'question':'q','filename':'co/page'}
        self.assertEqual(answer_messages(item,[])[0]['content'],ANSWER_SYSTEM)
        self.assertNotIn('conversation_history',json.loads(answer_messages(item,[])[1]['content']))

    def test_financebench_text_and_boolean_are_not_forced_into_calculator(self):
        item={'id':'q','filename':'co/page','question':'answer?','benchmark':'financebench'}
        evidence=FakeIndex().retrieve('q')
        for value,kind in [('yes','boolean'),('net revenue','text')]:
            client=FakeClient([fb_answer(value,kind),fb_verdict(value)])
            a=b.fb_validate(client,item,b.fb_draft(client,item,evidence,'a'),'v')
            self.assertEqual(a['calculator']['status'],'not_applicable');self.assertTrue(a['accepted'])

    def test_fabricated_audit_quote_cannot_pass(self):
        item={'id':'q','filename':'co/page','question':'answer?','benchmark':'financebench'}
        client=FakeClient([fb_answer('yes','boolean'),fb_verdict('yes','INVENTED EVIDENCE')])
        a=b.fb_validate(client,item,b.fb_draft(client,item,FakeIndex().retrieve('q'),'a'),'v')
        self.assertFalse(a['accepted'])

    def test_invalid_numeric_program_cannot_be_relabelled_as_supported_number(self):
        item={'id':'q','filename':'co/page','question':'answer?','benchmark':'financebench'}
        reply=fb_answer(10,'number');reply['program']='dangerous(1,2)'
        client=FakeClient([reply,fb_verdict(10)])
        a=b.fb_validate(client,item,b.fb_draft(client,item,FakeIndex().retrieve('q'),'a'),'v')
        self.assertFalse(a['accepted']);self.assertEqual(a['answer']['value'],'unknown')

    def test_histories_follow_each_arms_own_generated_answers(self):
        conv={'id':'c','filename':'co/page','questions':['first?','second?'],'benchmark':'convfinqa'}
        # First baseline 20; agentic corrects its own first answer from 20 to 10.
        client=FakeClient([answer(20),answer(20),verdict(False),answer(10),verdict(True),
                           answer(30),answer(10),verdict(True)])
        result=answer_conversation(client,FakeIndex(),threading.RLock(),conv,{'top_k':10,'max_attempts':3})
        items=result['turns'][1]['items']
        self.assertEqual(items['single_shot']['history'][0]['answer'],'20.0')
        self.assertEqual(items['agentic']['history'][0]['answer'],'10.0')
        self.assertNotIn('GOLD',json.dumps(client.messages))

if __name__=='__main__':unittest.main()
