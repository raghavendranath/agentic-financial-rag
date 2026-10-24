"""Score every final answer; separate exact match from model-judged correctness."""
import argparse
import importlib.util
import re
import sys
import unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from finrag.core import read_json,write_json,jsonl,number,correct,digest
from finrag.report import cluster_interval,fraction
from verify_benchmark import verify

def normalise_text(value):
    return ' '.join(unicodedata.normalize('NFKC',str(value)).casefold().split()).rstrip('.?!;')

def financebench_em(value,reference):
    # Only whole scalar references receive numerical normalization. No arbitrary number extraction.
    a,b=number(value),number(reference)
    if a is not None and b is not None:return correct(value,reference)
    if value is None or normalise_text(value) in ('unknown',''):return False
    return normalise_text(value)==normalise_text(reference)

def transitions(rows,a,b):
    ds=[x-y for x,y in zip(a,b)]
    return {'difference':sum(ds)/len(ds),'95ci_filing_cluster_bootstrap':cluster_interval(rows,ds),
            'wrong_to_right':sum(d==1 for d in ds),'right_to_wrong':sum(d==-1 for d in ds)}

def finish(folder):
    folder=Path(folder);integrity=verify(folder)
    manifest=read_json(folder/'manifest.json');benchmark=manifest['frozen']['benchmark'];split=manifest['split']
    data=ROOT/'data'/benchmark;dataset=read_json(data/'manifest.json');labels=read_json(data/(split+'.labels.json'))
    if digest(labels)!=dataset['splits'][split]['labels_hash']:raise ValueError('Labels changed')
    byid={r['id']:r for r in jsonl(folder/'results.jsonl')};rows=[byid[i] for i in dataset['splits'][split]['ids']]
    upstream_path=ROOT/'provenance/finqa-official-evaluate.py'
    spec=importlib.util.spec_from_file_location('finqa_official_score',upstream_path);upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    items=[]
    for row in rows:
        if benchmark=='convfinqa':
            for t in row['turns']:
                items.append({'id':row['id']+':turn:'+str(t['turn']),'conversation_id':row['id'],'filename':row['filename'],
                    'question':t['question'],'turn':t['turn'],'reference':labels[row['id']]['answers'][t['turn']],'arms':t['arms']})
        else:items.append({**row,'reference':labels[row['id']]['answer']})
    metric=correct if benchmark=='convfinqa' else financebench_em
    outcomes={};summary={};details=[];program_checks=0
    for arm in items[0]['arms']:
        hits=[];numeric_hits=[];numeric_rows=[];attempts=[];unknown=0
        for item in items:
            result=item['arms'][arm];answer=result['final'];value=(answer or {}).get('value');hit=int(metric(value,item['reference']))
            hits.append(hit);attempts.append(len(result['attempts']));unknown+=value is None or str(value).lower()=='unknown'
            if isinstance(number(item['reference']),(int,float)):
                numeric_hits.append(int(correct(value,item['reference'])));numeric_rows.append(item)
            for attempt in result['attempts']:
                calc=attempt['calculator']
                if calc['status']=='executed':
                    invalid,got=upstream.eval_program(calc['program_tokens'],[])
                    saved=number(calc['value']);saved=round(saved,5) if isinstance(saved,(int,float)) else saved
                    if invalid or got!=saved:raise ValueError('Official arithmetic execution differs')
                    program_checks+=1
            details.append({'id':item['id'],'arm':arm,'reference':item['reference'],'value':value,'correct':bool(hit),'attempts':len(result['attempts'])})
        outcomes[arm]=hits
        summary[arm]={'items':len(items),'correct':sum(hits),'normalized_exact_match':sum(hits)/len(hits),
                      'exact_match_95ci':cluster_interval(items,hits),'numerical_items':len(numeric_hits),'numerical_correct':sum(numeric_hits),
                      'scalar_numerical_f1':fraction(sum(numeric_hits),len(numeric_hits)),
                      'numerical_f1_95ci':cluster_interval(numeric_rows,numeric_hits) if numeric_rows else None,
                      'mean_attempts':sum(attempts)/len(attempts),'unknown_or_malformed':unknown}
        if benchmark=='convfinqa':
            episode_hits=[int(all(metric((t['arms'][arm]['final'] or {}).get('value'),labels[row['id']]['answers'][t['turn']]) for t in row['turns'])) for row in rows]
            summary[arm]['all_turns_correct_conversation_rate']=sum(episode_hits)/len(rows)
            summary[arm]['conversation_rate_95ci']=cluster_interval(rows,episode_hits)
            positions={}
            for pos in sorted({i['turn'] for i in items}):
                at=[hit for item,hit in zip(items,hits) if item['turn']==pos]
                positions[str(pos+1)]={'n':len(at),'correct':sum(at),'accuracy':sum(at)/len(at)}
            summary[arm]['by_turn']=positions
            first3=[hit for item,hit in zip(items,hits) if item['turn']<3]
            summary[arm]['first_three_turns_accuracy']=sum(first3)/len(first3)
    changes={arm:transitions(items,hits,outcomes['single_shot']) for arm,hits in outcomes.items() if arm!='single_shot'}
    initial=[int(metric((i['arms']['agentic']['attempts'][0]['answer'] or {}).get('value'),i['reference'])) for i in items]
    tp=fp=tn=fn=0
    for i,hit in zip(items,initial):
        rejected=not i['arms']['agentic']['attempts'][0]['accepted']
        tp+=rejected and not hit;fp+=rejected and bool(hit);tn+=not rejected and bool(hit);fn+=not rejected and not hit
    validator={'reference_metric':'strict numerical answer agreement' if benchmark=='convfinqa' else 'normalized exact match; paraphrases may count as disagreements',
        'true_reject_reference_mismatch':tp,'false_reject_reference_match':fp,'true_accept_reference_match':tn,'false_accept_reference_mismatch':fn,
        'detection_precision':fraction(tp,tp+fp),'detection_recall':fraction(tp,tp+fn),
        'false_acceptance_rate_among_reference_mismatches':fraction(fn,tp+fn),'false_rejection_rate_among_reference_matches':fraction(fp,fp+tn)}
    validator['first_attempt_acceptance_consistency_score']=(tn+fn)/len(items)
    loop_stats={}
    for arm in summary:
        refinements=repeated=unchanged=0
        for item in items:
            attempts=item['arms'][arm]['attempts']
            for previous,nxt in zip(attempts,attempts[1:]):
                refinements+=1
                a,b=previous['answer'] or {},nxt['answer'] or {}
                repeated+=(a.get('value'),a.get('program'))==(b.get('value'),b.get('program'))
                unchanged+={e['id'] for e in previous['evidence']}=={e['id'] for e in nxt['evidence']}
        loop_stats[arm]={'refinement_steps':refinements,'unchanged_evidence_sets':unchanged,'repeated_value_and_program':repeated}
    first_initial_to_final=transitions(items,outcomes['agentic'],initial)
    calls=jsonl(folder/'calls.jsonl')
    result={'benchmark':benchmark,'split':split,'units':len(rows),'items':len(items),'arms':summary,'end_to_end_vs_single_shot':changes,
        'within_agentic_turn_initial_to_final':first_initial_to_final,'validator_first_attempt_reference_evaluation':validator,
        'loop_statistics':loop_stats,
        'official_arithmetic_program_checks':program_checks,'integrity':integrity,
        'billing':{'unique_calls':len(calls),'reported_cost_usd':sum(float(c['usage']['cost']) for c in calls)},
        'limitations':['Small fixed public sample; newer hosted model is not a reproduction of the lost historical run.',
          'No benchmark label edits, sign/factor-100 forgiveness, best-of-attempt selection or test-driven tuning.',
          'Same-model validation can accept wrong financial interpretations; no independent human grounding audit.',
          'Confidence intervals resample filings (seed 18, 2000 draws).',
          'FinanceBench normalized exact match uses whole-scalar 5-decimal comparison or case/Unicode/whitespace normalization and terminal punctuation removal; semantic judgment is separate.',
          'ConvFinQA uses generated history per arm; end-to-end differences combine current-turn refinement with effects of earlier generated answers.']}
    judge=folder/'automated-judge/results.jsonl'
    if benchmark=='financebench' and judge.exists():
        judgments=jsonl(judge);mapping={(x['id'],x['arm']):x for x in judgments}
        expected={(i['id'],arm) for i in items for arm in i['arms']}
        if len(mapping)!=len(judgments) or set(mapping)!=expected:raise ValueError('Incomplete/duplicate automated judgments')
        semantic={};grounding={}
        for arm in summary:
            js=[mapping[(i['id'],arm)]['judgment'] for i in items]
            hits=[int(bool(j and j['correct'])) for j in js];supported=[int(bool(j and j['claims_supported'])) for j in js]
            answered=[bool(j and j['answered']) for j in js]
            semantic[arm]=hits;grounding[arm]=supported
            summary[arm]['automated_semantic_correctness']=sum(hits)/len(hits)
            summary[arm]['automated_semantic_95ci']=cluster_interval(items,hits)
            summary[arm]['automated_answer_grounding_rate_all_items']=sum(supported)/len(supported)
            summary[arm]['automated_grounding_rate_among_answered']=fraction(sum(supported),sum(answered))
            summary[arm]['malformed_judgments']=sum(j is None for j in js)
        result['automated_semantic_vs_single_shot']={arm:transitions(items,hits,semantic['single_shot']) for arm,hits in semantic.items() if arm!='single_shot'}
        result['automated_judge']={'model':read_json(folder/'automated-judge/manifest.json')['frozen']['config']['model'],
                                  'human_audit':False,'metric_is_exact_match':False,'judge_can_be_wrong':True}
    write_json(folder/'SCORES.json',result);write_json(folder/'ITEM-SCORES.json',details)
    lines=['# '+benchmark+' evaluation','',f"Split: {split}. {len(rows)} {'complete conversations' if benchmark=='convfinqa' else 'questions'}, {len(items)} scored answers.",'',
      '| Method | Correct / items | Normalized EM | Scalar numerical F1 | Mean attempts |','|---|---:|---:|---:|---:|']
    for arm,s in summary.items():
        f1='n/a' if s['scalar_numerical_f1'] is None else f"{s['scalar_numerical_f1']:.3f}"
        lines.append(f"| {arm} | {s['correct']}/{s['items']} | {s['normalized_exact_match']:.3f} | {f1} (n={s['numerical_items']}) | {s['mean_attempts']:.2f} |")
    if 'automated_judge' in result:
        lines+=['','## Supplemental automated assessment','',
           '| Method | Model-judged semantic correctness | Model-judged grounding, all items |','|---|---:|---:|']
        for arm,s in summary.items():lines.append(f"| {arm} | {s['automated_semantic_correctness']:.3f} | {s['automated_answer_grounding_rate_all_items']:.3f} |")
        lines+=['','These are Qwen3-32B judgments, not exact match, human annotations or guarantees.']
    lines+=['','## Paired changes','',str(changes),'','## Limitations','']+['- '+x for x in result['limitations']]
    (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();r=finish(ROOT/a.run)
    print({'benchmark':r['benchmark'],'units':r['units'],'items':r['items'],'scores':r['arms'],'changes':r['end_to_end_vs_single_shot']})
