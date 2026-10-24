"""Replay every ablation request offline, then score frozen reference agreement."""
from collections import Counter
from copy import deepcopy
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unicodedata
from unittest.mock import patch
from common import ROOT,assert_frozen,file_hash
from finrag.core import read_json,write_json,jsonl,digest,number,correct
from finrag.client import OpenRouterClient
from finrag.report import cluster_interval,fraction
from finrag.unguided import run_item

# Same metric as v4. No question-dependent rounding or semantic adjudication added.
def normalise_text(value):
    return ' '.join(unicodedata.normalize('NFKC',str(value)).casefold().split()).rstrip('.?!;')

def financebench_em(value,reference):
    a,b=number(value),number(reference)
    if a is not None and b is not None:return correct(value,reference)
    if value is None or normalise_text(value) in ('unknown',''):return False
    return normalise_text(value)==normalise_text(reference)

def compare(records,a,b):
    changes=[x-y for x,y in zip(a,b)]
    return {'difference':sum(changes)/len(changes),
        '95ci_filing_cluster_bootstrap':cluster_interval(records,changes),
        'reference_mismatch_to_match':sum(x==1 and y==0 for x,y in zip(a,b)),
        'reference_match_to_mismatch':sum(x==0 and y==1 for x,y in zip(a,b))}

def main():
    folder=ROOT/'runs/evaluation';manifest=read_json(folder/'manifest.json')
    if manifest['status']!='inference_complete':raise ValueError('Do not access assessment labels before inference completes')
    frozen=assert_frozen()
    if manifest['frozen']!=frozen:raise ValueError('Manifest freeze mismatch')
    for name,expected in frozen['assessment_files'].items():
        if file_hash(ROOT/'assessment'/name)!=expected:raise ValueError('Frozen assessment changed')
    inputs=read_json(ROOT/'data/inputs.json')
    rows=jsonl(folder/'results.jsonl');byid={r['id']:r for r in rows}
    if len(rows)!=50 or len(byid)!=50 or set(byid)!={r['item']['id'] for r in inputs}:
        raise ValueError('Selected/incomplete/duplicate results')
    calls=jsonl(folder/'calls.jsonl');bycall={c['request_hash']:c for c in calls}
    if len(calls)!=100 or len(bycall)!=100:raise ValueError('Expected exactly 100 new calls')
    ledger=jsonl(folder/'budget.jsonl')
    if len(ledger)!=100 or {r['request_hash'] for r in ledger}!=set(bycall):
        raise ValueError('Unresolved or duplicated paid reservation')
    config=frozen['config'];referenced=set()
    for c in calls:
        request=c['request'];payload=json.loads(request['messages'][1]['content'])
        if digest(request)!=c['request_hash'] or request['model']!=c['returned_model'] or c['returned_model']!=config['model']:
            raise ValueError('Changed model/request')
        if c['returned_provider']!=manifest['preflight']['endpoint']['provider_name']:
            raise ValueError('Provider changed')
        if request['provider']!={'only':[config['provider']],'allow_fallbacks':False,'require_parameters':True}:
            raise ValueError('Routing changed')
        for k in ('temperature','seed','max_tokens','reasoning'):
            if request.get(k)!=config.get(k):raise ValueError('Decoding changed')
        if set(payload)!={'question','report_page','evidence','feedback'}:
            raise ValueError('Unexpected inference fields')
        context=payload['feedback']
        if set(context)!={'pass_number','instruction','previous_model_draft'} or context['pass_number'] not in (2,3):
            raise ValueError('Forbidden feedback or pass count')
        if c['usage'].get('cost') is None:raise ValueError('Missing billing receipt')
    with tempfile.TemporaryDirectory(prefix='unguided-replay-') as tmp:
        for name in ('calls.jsonl','budget.jsonl'):shutil.copyfile(folder/name,Path(tmp)/name)
        client=OpenRouterClient(config,tmp,'OFFLINE_NO_KEY',manifest['preflight'])
        with patch('finrag.client.fetch_json',side_effect=AssertionError('Unrecorded request: network forbidden')):
            replay={r['item']['id']:run_item(client,r) for r in inputs}
        if replay!=byid:raise ValueError('Offline request/trajectory replay differs')
    official_spec=importlib.util.spec_from_file_location('official_ablation',ROOT/'provenance/finqa-official-evaluate.py')
    official=importlib.util.module_from_spec(official_spec);official_spec.loader.exec_module(official)
    program_checks=0
    for row in rows:
        if row['answer_passes']!=3 or len(row['attempts'])!=3 or row['validator_calls']!=0 or row['retrieval_changes']!=0:
            raise ValueError('Iteration/validation/retrieval policy changed')
        if row['final']!=row['attempts'][2]['answer'] or row['output_status']!=row['attempts'][2]['output_status']:
            raise ValueError('Final-policy selection changed')
        if any(digest(a['evidence'])!=digest(row['attempts'][0]['evidence']) for a in row['attempts']):
            raise ValueError('Evidence changed')
        for n,a in enumerate(row['attempts']):
            if any(k in a for k in ('validator_call','source_audit_call','verdict','accepted','audit')):
                raise ValueError('Validator signal present in control attempt')
            if a['output_status'] not in ('invalid','abstained','unvalidated'):raise ValueError('Control incorrectly marked validated')
            if n:referenced.add(a['answer_call'])
            calc=a['calculator']
            if calc.get('status')=='executed':
                invalid,value=official.eval_program(calc['program_tokens'],[])
                expected=number(calc['value']);expected=round(expected,5) if isinstance(expected,(int,float)) else expected
                if invalid or value!=expected:raise ValueError('Official arithmetic disagrees')
                program_checks+=1
    if referenced!=set(bycall):raise ValueError('Unreferenced paid response')
    integrity={'status':'passed','questions':50,'shared_initial_receipts':50,'new_paid_responses':100,
        'pending_reservations':0,'every_request_and_trajectory_replayed_offline':True,
        'no_labels_accessed_in_generation':True,'no_validator_signals_in_revision_context':True,
        'fixed_evidence_and_final_third_pass_verified':True,'official_program_execution_checks':program_checks}
    write_json(folder/'INTEGRITY.json',integrity)
    # Reference access begins only after complete inference and replay verification.
    labels=read_json(ROOT/'assessment/evaluation.labels.json')
    if digest(labels)!=frozen['source_receipt']['source_data_manifest']['splits']['evaluation']['labels_hash']:
        raise ValueError('Reference labels differ from original benchmark')
    comparators=read_json(ROOT/'assessment/comparators.json')
    source_costs=read_json(ROOT/'assessment/source-call-costs.json')
    records=[byid[r['item']['id']] for r in inputs]
    names=['single_shot','unguided_fixed_three_pass','fixed_retrieval','agentic']
    hits={name:[] for name in names};summary={};details=[];trajectories=[]
    for row in records:
        ident=row['id'];ref=labels[ident]['answer']
        for name in names:
            result=row if name=='unguided_fixed_three_pass' else comparators[ident]['arms'][name]
            answer=result['final'];value=(answer or {}).get('value');hit=int(financebench_em(value,ref))
            hits[name].append(hit)
            details.append({'id':ident,'filename':row['filename'],'arm':name,'reference':ref,'value':value,
                'literal_reference_match':bool(hit),'output_status':result['output_status'],
                'answer':answer,'human_factual_correctness_assessed':False})
        trajectories.append({'id':ident,'filename':row['filename'],
            'pass_values':[(a['answer'] or {}).get('value') for a in row['attempts']],
            'pass_reference_matches':[int(financebench_em((a['answer'] or {}).get('value'),ref)) for a in row['attempts']]})
    for name in names:
        statuses=Counter();passes=[];requests=set()
        for row in records:
            ident=row['id'];result=row if name=='unguided_fixed_three_pass' else comparators[ident]['arms'][name]
            statuses[result['output_status']]+=1;passes.append(result['answer_passes'])
            if name=='unguided_fixed_three_pass':requests.update(a['answer_call'] for a in row['attempts'])
            else:requests.update(result['request_hashes'])
        costs={h:(bycall[h]['usage']['cost'] if h in bycall else source_costs[h]['cost']) for h in requests}
        numeric=[i for i,row in enumerate(records) if isinstance(number(labels[row['id']]['answer']),(int,float))]
        numeric_hits=[hits[name][i] for i in numeric]
        summary[name]={'questions':50,'literal_matches':sum(hits[name]),'normalized_literal_em':sum(hits[name])/50,
            '95ci_filing_cluster_bootstrap':cluster_interval(records,hits[name]),
            'whole_numeric_reference_questions':len(numeric),'numeric_reference_matches':sum(numeric_hits),
            'scalar_numeric_reference_f1':fraction(sum(numeric_hits),len(numeric_hits)),
            'mean_answer_passes':sum(passes)/50,'unique_model_requests_attributed':len(requests),
            'attributed_response_cost_usd':float(sum(Decimal(str(x)) for x in costs.values())),
            'output_status_counts':dict(statuses)}
    paired={name:compare(records,hits['unguided_fixed_three_pass'],hits[name]) for name in names if name!='unguided_fixed_three_pass'}
    new_cost=sum(Decimal(str(c['usage']['cost'])) for c in calls)
    result={'status':'complete','benchmark':'financebench','evaluation_role':'post-hoc additional component ablation on previously inspected fixed sample',
        'questions':50,'arms':summary,'unguided_minus_comparator':paired,
        'pass_level_literal_agreement':{str(n+1):sum(t['pass_reference_matches'][n] for t in trajectories)/50 for n in range(3)},
        'incremental_billing':{'new_calls':100,'new_reported_cost_usd':str(new_cost),'reused_first_drafts':50,'first_drafts_not_billed_again':True},
        'human_assessment_completed':False,'integrity':integrity,
        'limitations':['Literal EM/reference agreement is diagnostic, not human-assessed financial correctness or grounding.',
            'Maximum answer-attempt budget is shared; total model calls/tokens and stopping rules differ.',
            'The validated fixed-retrieval arm combines validator feedback and adaptive stopping. This comparison does not isolate those two mechanisms.',
            'Full agentic comparison additionally changes retrieval. No independent effect of every component or EDGAR/Python tool is established.',
            'This sample has been inspected before; no untouched-holdout or model-selection claim. Final third pass was frozen before new calls; no best-of-three selection.',
            'All historical runs remain unchanged. No human-label study or CL error-injection/dependency experiment was performed.']}
    write_json(folder/'SCORES.json',result);write_json(folder/'ITEM-SCORES.json',details);write_json(folder/'PASS-TRAJECTORIES.json',trajectories)
    lines=['FinanceBench additional ablation: self-revision without validator feedback','',
        'Same fifty questions, exact shared first drafts and initial retrieved evidence. No validator feedback or new retrieval; exactly three answer passes; last pass always scored.','',
        'Method | Literal matches / 50 | Literal EM | Mean answer passes | Attributed model calls',
        '--- | --- | --- | --- | ---']
    for name,s in summary.items():lines.append(f"{name} | {s['literal_matches']}/50 | {s['normalized_literal_em']:.3f} | {s['mean_answer_passes']:.2f} | {s['unique_model_requests_attributed']}")
    lines+=['','Reference-agreement comparisons (unguided minus comparator):',json.dumps(paired,indent=2),
        '',f'Additional paid requests: 100. Reported additional cost: ${new_cost}. The 50 initial drafts were reused, not charged again.',
        '', 'Every new request and final-selection decision replayed offline; bounded arithmetic independently checked.',
        '', 'Limitations:']+['- '+x for x in result['limitations']]
    (folder/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    print({name:s['literal_matches'] for name,s in summary.items()});print('New cost USD',new_cost)

if __name__=='__main__':main()
