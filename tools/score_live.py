#!/usr/bin/env python3
"""Score a completed fresh run; reference access starts after completion checks."""
import argparse,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT={'finqa':'nlp4kgc-validation-v3','convfinqa':'nlp4kgc-three-benchmarks','financebench':'nlp4kgc-validation-v4'}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();folder=a.run.resolve()
    config=json.loads((folder/'config.json').read_text());manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['status']!='inference_complete':raise ValueError('Complete inference before opening references')
    benchmark=manifest['frozen'].get('benchmark')
    if benchmark is None and manifest.get('live_wrapper_version') is None:benchmark='finqa'
    project=ROOT/'experiments'/PROJECT[benchmark];sys.path[:0]=[str(project/'src'),str(project/'tools')]
    from finrag.core import read_json,jsonl,digest,number,correct,write_json
    from finrag.experiment import source_hash
    from finrag.report import cluster_interval
    if source_hash()!=manifest['frozen']['source_hash']:raise ValueError('Generation source changed')
    data=project/('data/fresh50' if benchmark=='finqa' else 'data/'+benchmark)
    dataset=read_json(data/'manifest.json');split=manifest.get('split',manifest['frozen'].get('split'))
    if digest(dataset)!=manifest['frozen']['data_hash']:raise ValueError('Data manifest changed')
    rows=jsonl(folder/'results.jsonl');inputs=read_json(data/(split+'.inputs.json'))
    if {r['id'] for r in rows}!={r['id'] for r in inputs} or len(rows)!=len(inputs):raise ValueError('Incomplete/duplicate results')
    calls=jsonl(folder/'calls.jsonl');call_ids={c['request_hash'] for c in calls}
    if len(call_ids)!=len(calls) or any(digest(c['request'])!=c['request_hash'] for c in calls):raise ValueError('Changed/duplicate requests')
    if {c['request_hash'] for c in jsonl(folder/'budget.jsonl')}!=call_ids:raise ValueError('Unresolved paid reservations')
    # Only now access the separately stored references.
    labels=read_json(data/(split+'.labels.json'))
    if digest(labels)!=dataset['splits'][split]['labels_hash']:raise ValueError('References changed')
    if benchmark=='financebench':
        from score_benchmark import financebench_em
        metric=financebench_em
    else:metric=correct
    items=[]
    for row in rows:
        for turn in (row['turns'] if benchmark=='convfinqa' else [row]):
            gold=labels[row['id']]['answers'][turn['turn']] if benchmark=='convfinqa' else labels[row['id']]['gold' if benchmark=='finqa' else 'answer']
            items.append({'filename':row['filename'],'arms':turn['arms'],'reference':gold})
    outcomes={}
    for arm in items[0]['arms']:
        hits=[int(metric((i['arms'][arm]['final'] or {}).get('value'),i['reference'])) for i in items]
        numeric=[i for i in items if isinstance(number(i['reference']),(int,float))]
        nh=[int(correct((i['arms'][arm]['final'] or {}).get('value'),i['reference'])) for i in numeric]
        outcomes[arm]={'items':len(hits),'matches':sum(hits),'reference_agreement':sum(hits)/len(hits),
            '95ci_filing_cluster':cluster_interval(items,hits),'numerical_items':len(nh),'numerical_matches':sum(nh),
            'scalar_numerical_agreement':sum(nh)/len(nh) if nh else None}
    result={'status':'scored_fresh_run','benchmark':benchmark,'arms':outcomes,'human_evaluation':False,
            'caution':'Reference agreement, not independent human factual correctness. This fresh run does not replace published scores.'}
    write_json(folder/'LIVE-SCORES.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError,KeyError) as e:print(e,file=sys.stderr);sys.exit(1)
