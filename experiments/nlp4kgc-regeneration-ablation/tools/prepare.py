"""Prepare allowlisted shared first drafts; separate assessment from inference."""
from copy import deepcopy
import inspect
import json
import sys
from common import ROOT,file_hash,frozen_state
from finrag.core import read_json,write_json,digest,jsonl,canonical
from finrag.benchmarks import FB_ANSWER
from finrag.unguided import run_item

SOURCE=ROOT.parent/'nlp4kgc-validation-v4'

def main():
    if (ROOT/'docs/FREEZE.json').exists():raise ValueError('Freeze already exists; do not reprepare after inference')
    folder=SOURCE/'runs/financebench-evaluation'
    manifest=read_json(folder/'manifest.json')
    if manifest['status']!='inference_complete' or read_json(folder/'INTEGRITY.json')['status']!='passed':
        raise ValueError('Verified source run required')
    inputs=read_json(SOURCE/'data/financebench/evaluation.inputs.json')
    rows={r['id']:r for r in jsonl(folder/'results.jsonl')}
    calls={r['request_hash']:r for r in jsonl(folder/'calls.jsonl')}
    if set(rows)!={i['id'] for i in inputs}:raise ValueError('Source sample changed')
    records=[];comparators={};source_call_costs={}
    for item in inputs:
        initial=rows[item['id']]['arms']['single_shot']['attempts'][0]
        call=calls[initial['answer_call']]
        payload=json.loads(call['request']['messages'][1]['content'])
        allowed_item={k:item[k] for k in ('id','filename','question','benchmark')}
        expected={'question':item['question'],'report_page':item['filename'],'evidence':initial['evidence']}
        if payload!=expected or call['request']['messages'][0]['content']!=FB_ANSWER:
            raise ValueError('Initial source request differs from original label-free question/evidence contract')
        records.append({'item':allowed_item,'evidence':initial['evidence'],'initial_call':call,
                        'initial_attempt_hash':digest(initial)})
        comparators[item['id']]={'filename':item['filename'],'arms':{}}
        for arm,result in rows[item['id']]['arms'].items():
            referenced={a[k] for a in result['attempts'] for k in ('answer_call','source_audit_call','validator_call') if k in a}
            source_call_costs.update({h:{'cost':calls[h]['usage']['cost'],'prompt_tokens':calls[h]['usage'].get('prompt_tokens'),
                'completion_tokens':calls[h]['usage'].get('completion_tokens')} for h in referenced})
            comparators[item['id']]['arms'][arm]={'final':result['final'],'output_status':result['output_status'],
                'answer_passes':len(result['attempts']),'request_hashes':sorted(referenced)}
    (ROOT/'assessment').mkdir(exist_ok=True)
    write_json(ROOT/'data/inputs.json',records)
    write_json(ROOT/'assessment/comparators.json',comparators)
    write_json(ROOT/'assessment/source-call-costs.json',source_call_costs)
    write_json(ROOT/'assessment/evaluation.labels.json',read_json(SOURCE/'data/financebench/evaluation.labels.json'))
    unchanged={p.name:file_hash(p)==file_hash(SOURCE/'src/finrag'/p.name) for p in sorted((ROOT/'src/finrag').glob('*.py')) if p.name!='unguided.py'}
    if not all(unchanged.values()):raise ValueError('Prior parser/model prompt/scoring module changed')
    write_json(ROOT/'provenance/SOURCE.json',{'source_project':SOURCE.name,'source_manifest_hash':digest(manifest),
        'source_results_hash':digest(jsonl(folder/'results.jsonl')),'source_inputs_hash':digest(inputs),
        'source_data_manifest':read_json(SOURCE/'data/financebench/manifest.json'),
        'provider_name':manifest['preflight']['endpoint']['provider_name'],'source_config':manifest['frozen']['config'],'shared_source_modules_unchanged':unchanged,
        'generation_inputs_contain_no_reference_answers_or_validator_feedback':True})
    write_json(ROOT/'provenance/ASSESSMENT-HASHES.json',{p.name:file_hash(p) for p in sorted((ROOT/'assessment').glob('*.json'))})
    write_json(ROOT/'docs/FREEZE.json',frozen_state())
    print('Frozen:',len(records),'questions; 50 reused first drafts; 100 new answer calls planned.')

if __name__=='__main__':main()
