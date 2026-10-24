import json
import unittest
from finrag.benchmarks import (fb_draft, fb_validate, fb_source_messages, COMPARE_CHECKS, BASE_CHECKS, trajectory)

ITEM={'id':'synthetic','filename':'report','question':'Which security is debt?','benchmark':'financebench'}
EVIDENCE=[{'id':'chunk1','filename':'report','text':'Debt securities: None. Common stock is equity.'}]

def answer(value='None',kind='text',program='',unit=None):
    return dict(answer_type=kind,value=value,program=program,unit=unit,citations=['chunk1'],explanation='Debt securities: None.')

def source(value='None',kind='text',program='',unit=None):
    return {**answer(value,kind,program,unit),'checks':dict.fromkeys(BASE_CHECKS,True),
            'audit_quote':'Debt securities: None.','retrieval_query':''}

def verdict(**checks):
    return {'accept':True,'reason':'supported','retrieval_query':'',
            'checks':{**dict.fromkeys(COMPARE_CHECKS,True),**checks}}

class Client:
    def __init__(self, replies):self.replies=list(replies);self.messages=[]
    def complete(self,messages,tag):
        self.messages.append(messages)
        raw=self.replies.pop(0)
        return {'content':raw if isinstance(raw,str) else json.dumps(raw),'request_hash':str(len(self.messages)),
                'elapsed_seconds':0,'finish_reason':'stop'}

class ValidationTests(unittest.TestCase):
    def validate(self,candidate,audit,compare):
        c=Client([candidate,audit,compare]);draft=fb_draft(c,ITEM,EVIDENCE,'a')
        return fb_validate(c,ITEM,draft,'v'),c
    def test_source_audit_has_no_candidate_or_feedback(self):
        result,c=self.validate(answer('Common stock'),source(),verdict(equivalent=False))
        payload=json.loads(c.messages[1][1]['content'])
        self.assertEqual(set(payload),{'question','report_page','evidence'})
        self.assertNotIn('proposed_answer',payload)
        self.assertFalse(result['accepted'])
    def test_false_equivalence_cannot_pass_even_accept_true(self):
        result,_=self.validate(answer('Common stock'),source(),verdict(equivalent=False))
        self.assertFalse(result['accepted']);self.assertEqual(result['output_status'],'unvalidated')
    def test_boolean_contradiction_blocks_optimistic_comparator(self):
        result,_=self.validate(answer('yes','boolean'),source('no','boolean'),verdict())
        self.assertFalse(result['accepted'])
        self.assertTrue(any('contradict' in x for x in result['operational_issues']))
    def test_numeric_program_disagreement_blocks_acceptance(self):
        result,_=self.validate(answer('3','number','add(3,0)','scalar'),source('4','number','add(4,0)','scalar'),verdict())
        self.assertFalse(result['accepted'])
    def test_small_value_magnitude_or_sign_not_erased_by_scorer_rounding(self):
        for value in ('0.000004','-0.000001'):
            result,_=self.validate(answer('0.000001','number','add(0.000001,0)','scalar'),
                source(value,'number','add('+value+',0)','scalar'),verdict())
            self.assertFalse(result['accepted'])
    def test_own_program_disagreement_never_replaces_declared_value(self):
        result,_=self.validate(answer('4','number','add(3,0)','scalar'),source('3','number','add(3,0)','scalar'),verdict())
        self.assertIsNone(result['answer']);self.assertEqual(result['typed_answer']['value'],'4')
        self.assertFalse(result['accepted']);self.assertEqual(result['output_status'],'invalid')
    def test_valid_text_can_pass(self):
        result,_=self.validate(answer(),source(),verdict())
        self.assertTrue(result['accepted']);self.assertEqual(result['output_status'],'validated')
    def test_quote_outside_evidence_rejected(self):
        audit=source();audit['audit_quote']='Nonexistent quote'
        result,_=self.validate(answer(),audit,verdict())
        self.assertFalse(result['accepted'])
    def test_unknown_text_cannot_bypass_abstention(self):
        result,_=self.validate(answer('unknown'),source(),verdict())
        self.assertEqual(result['answer']['value'],'unknown');self.assertFalse(result['accepted'])
        self.assertEqual(result['output_status'],'abstained')
    def test_source_query_survives_bad_comparison(self):
        audit=source();audit['retrieval_query']='report debt securities table'
        result,_=self.validate(answer(),audit,'malformed')
        self.assertFalse(result['accepted'])
        self.assertEqual(result['verdict']['retrieval_query'],audit['retrieval_query'])
    def test_loop_does_not_mark_rejected_final_validated(self):
        first,_=self.validate(answer(),source(),verdict(evidence_sufficient=False))
        result=trajectory(None,None,ITEM,first,{'max_attempts':1,'top_k':10})
        self.assertEqual(result['output_status'],'unvalidated');self.assertFalse(result['accepted'])

if __name__=='__main__':unittest.main()
