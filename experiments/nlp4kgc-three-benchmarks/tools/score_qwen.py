"""Replay and score every Qwen baseline answer with the same fixed metrics."""
import argparse
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from finrag.core import read_json,write_json,jsonl,digest,number,correct
from finrag.client import OpenRouterClient
from finrag.report import cluster_interval,fraction
from qwen_baseline import sources,frozen_for,RecordedIndex,one
from score_benchmark import financebench_em,transitions

def score(benchmark):
    data,source,split=sources(benchmark);folder=ROOT/'runs'/('qwen-'+benchmark)
    manifest=read_json(folder/'manifest.json');frozen=manifest['frozen']
    if manifest['status']!='inference_complete' or frozen!=frozen_for(benchmark):raise ValueError('Incomplete/changed baseline')
    dataset=read_json(data/'manifest.json');inputs=read_json(data/(split+'.inputs.json'));labels=read_json(data/(split+'.labels.json'))
    if digest(inputs)!=dataset['splits'][split]['inputs_hash'] or digest(labels)!=dataset['splits'][split]['labels_hash']:raise ValueError('Data content changed')
    rows=jsonl(folder/'results.jsonl');byid={r['id']:r for r in rows};source_rows={r['id']:r for r in jsonl(source/'results.jsonl')}
    if len(rows)!=len(byid) or set(byid)!={i['id'] for i in inputs}:raise ValueError('Incomplete/duplicate results')
    calls=jsonl(folder/'calls.jsonl');bycall={c['request_hash']:c for c in calls}
    if len(bycall)!=len(calls) or {c['request_hash'] for c in jsonl(folder/'budget.jsonl')}!=set(bycall):raise ValueError('Unresolved/duplicate call')
    for c in calls:
        if digest(c['request'])!=c['request_hash'] or c['returned_model']!=frozen['config']['model'] or c['returned_provider']!=manifest['preflight']['endpoint']['provider_name']:raise ValueError('Request/model/provider changed')
        if c['usage'].get('cost') is None:raise ValueError('Missing cost')
    with tempfile.TemporaryDirectory(prefix='qwen-baseline-replay-') as tmp:
        tmp=Path(tmp)
        for name in ('calls.jsonl','budget.jsonl'):shutil.copyfile(folder/name,tmp/name)
        client=OpenRouterClient(frozen['config'],tmp,'OFFLINE_NO_KEY',manifest['preflight'])
        index=RecordedIndex(data,source,tmp) if benchmark=='convfinqa' else None
        with patch('finrag.client.fetch_json',side_effect=AssertionError('Unrecorded request')):
            replay={item['id']:one(client,benchmark,item,source_rows[item['id']],index,frozen['config']) for item in inputs}
        if replay!=byid:raise ValueError('Offline replay differs')
        if benchmark=='convfinqa' and {r['query_hash']:r for r in jsonl(tmp/'retrievals.jsonl')}!={r['query_hash']:r for r in jsonl(folder/'retrievals.jsonl')}:raise ValueError('Retrieval changed')
    rows=[byid[i['id']] for i in inputs];items=[];referenced=set();program_checks=0
    spec=importlib.util.spec_from_file_location('official_qwen',ROOT/'provenance/finqa-official-evaluate.py');upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    for row in rows:
        turns=row['turns'] if benchmark=='convfinqa' else [row]
        for turn in turns:
            arm=turn['arms']['qwen_single_shot'];answer=arm['final'];attempt=arm['attempts'][0];referenced.add(attempt['answer_call'])
            if len(arm['attempts'])!=1:raise ValueError('Baseline refined')
            gold=(labels[row['id']]['answers'][turn['turn']] if benchmark=='convfinqa' else labels[row['id']]['answer'] if benchmark=='financebench' else labels[row['id']]['gold'])
            calc=attempt['calculator']
            if calc['status']=='executed':
                invalid,result=upstream.eval_program(calc['program_tokens'],[]);saved=number(calc['value']);saved=round(saved,5) if isinstance(saved,(int,float)) else saved
                if invalid or result!=saved:raise ValueError('Official arithmetic execution differs')
                program_checks+=1
            items.append({'id':row['id']+(':turn:'+str(turn['turn']) if benchmark=='convfinqa' else ''),'conversation_id':row['id'],
                          'filename':row['filename'],'turn':turn.get('turn'),'value':(answer or {}).get('value'),'reference':gold})
    if referenced!=set(bycall):raise ValueError('Unreferenced/missing baseline reply')
    metric=financebench_em if benchmark=='financebench' else correct
    hits=[int(metric(i['value'],i['reference'])) for i in items]
    numerical=[i for i in items if isinstance(number(i['reference']),(int,float))];nh=[int(correct(i['value'],i['reference'])) for i in numerical]
    result={'benchmark':benchmark,'units':len(rows),'items':len(items),'correct':sum(hits),'normalized_exact_match':sum(hits)/len(hits),
            'accuracy_95ci':cluster_interval(items,hits),'numerical_items':len(nh),'numerical_correct':sum(nh),'scalar_numerical_f1':fraction(sum(nh),len(nh)),
            'numerical_f1_95ci':cluster_interval(numerical,nh) if numerical else None,'official_arithmetic_checks':program_checks,
            'integrity':'passed: all baseline requests and histories replayed offline','source_evidence_shared_with_deepseek':True,
            'billing':{'unique_calls':len(calls),'reported_cost_usd':sum(float(c['usage']['cost']) for c in calls)}}
    comparisons={}
    for arm in ('single_shot','agentic'):
        dh=[]
        for i in items:
            src=source_rows[i['conversation_id']];turn=src['turns'][i['turn']] if benchmark=='convfinqa' else src
            dh.append(int(metric((turn['arms'][arm]['final'] or {}).get('value'),i['reference'])))
        comparisons['qwen_minus_deepseek_'+arm]=transitions(items,hits,dh)
    result['paired_comparisons']=comparisons
    if benchmark=='convfinqa':
        episodes=[int(all(metric(i['value'],i['reference']) for i in items if i['conversation_id']==r['id'])) for r in rows]
        result['all_turns_correct_conversation_rate']=sum(episodes)/len(rows)
        first3=[hit for i,hit in zip(items,hits) if i['turn']<3]
        result['first_three_turns_accuracy']=sum(first3)/len(first3)
    write_json(folder/'SCORES.json',result);write_json(folder/'ITEM-SCORES.json',items)
    write_json(folder/'INTEGRITY.json',{'status':'passed','units':len(rows),'requests':len(calls),'all_requests_replayed_offline':True,'pending_reservations':0,'frozen':frozen})
    (folder/'RESULTS.md').write_text(f"# Qwen3-32B single-shot: {benchmark}\n\n{sum(hits)}/{len(hits)} exact/reference matches ({result['normalized_exact_match']:.3f}). Scalar numerical F1 {result['scalar_numerical_f1']} on {len(nh)} numerical questions.\n\nAll requests and generated histories replayed offline. Same inputs and initial source retrieval as the DeepSeek comparison. No refinements or historical model scores reused.\n")
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--benchmark',required=True,choices=['finqa','financebench','convfinqa']);r=score(p.parse_args().benchmark)
    print({k:r[k] for k in ('benchmark','items','correct','normalized_exact_match','scalar_numerical_f1')})
