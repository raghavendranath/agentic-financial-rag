"""Verify complete paired records and score every frozen item without label changes."""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from finrag.core import correct, jsonl, number, read_json, write_json
from finrag.report import make_report, cluster_interval, fraction
from verify_run import verify
from score_official import score


def finish(folder):
    folder = Path(folder)
    integrity = verify(folder)
    data = ROOT / 'data/fresh50'
    split = read_json(folder / 'manifest.json')['frozen']['split']
    basic = make_report(data, split, folder)
    official = score(data, folder)
    dataset = read_json(data / 'manifest.json')
    labels = read_json(data / (split + '.labels.json'))
    by_id = {r['id']:r for r in jsonl(folder / 'results.jsonl')}
    rows = [by_id[i] for i in dataset['splits'][split]['ids']]
    outcomes = {name:[int(correct((r['arms'][name]['final'] or {}).get('value'), labels[r['id']]['gold'])) for r in rows]
                for name in ('single_shot','baseline_v2','agentic')}
    differences = [a-b for a,b in zip(outcomes['agentic'],outcomes['baseline_v2'])]
    numeric_indices = [i for i,r in enumerate(rows) if isinstance(number(labels[r['id']]['gold']), (int,float))]
    numeric_rows = [rows[i] for i in numeric_indices]
    numeric_differences = [differences[i] for i in numeric_indices]
    numeric_intervals = {name:cluster_interval(numeric_rows,[hits[i] for i in numeric_indices]) for name,hits in outcomes.items()}
    validation, loops = {}, {}
    diagnostics = []
    for name in ('baseline_v2','agentic'):
        tp=fp=tn=fn=malformed=0
        refinements=repeated=unchanged=overridden=aliases=0
        for row,hit in zip(rows,outcomes['single_shot']):
            attempts = row['arms'][name]['attempts']
            first = attempts[0]
            rejected = not first['accepted']
            tp += rejected and not hit
            fp += rejected and bool(hit)
            tn += not rejected and bool(hit)
            fn += not rejected and not hit
            malformed += first['verdict'] is None
            for previous, nxt in zip(attempts,attempts[1:]):
                refinements += 1
                pa, na = previous['answer'] or {}, nxt['answer'] or {}
                repeated += (pa.get('value'),pa.get('program')) == (na.get('value'),na.get('program'))
                unchanged += {c['id'] for c in previous['evidence']} == {c['id'] for c in nxt['evidence']}
            for a in attempts:
                overridden += bool(a['verdict'] and a['verdict']['accept'] and not a['accepted'])
                resolution = a.get('citation_resolution',{})
                aliases += bool(resolution and not resolution['problems'] and resolution['resolved'] != resolution['original'])
        validation[name] = {'true_reject_wrong':tp,'false_reject_correct':fp,'true_accept_correct':tn,'false_accept_wrong':fn,
            'error_detection_precision':fraction(tp,tp+fp),'error_detection_recall':fraction(tp,tp+fn),
            'false_acceptance_rate_among_wrong':fraction(fn,tp+fn),'false_rejection_rate_among_correct':fraction(fp,fp+tn),
            'malformed_first_verdicts':malformed}
        loops[name] = {'refinement_steps':refinements,'repeated_value_and_program':repeated,'unchanged_evidence_sets':unchanged,
                       'validator_accepts_overridden':overridden,'resolved_real_citation_aliases':aliases}
    for row in rows:
        diagnostics.append({'id':row['id'],'question':row['question'],'reference':labels[row['id']]['gold'],
            'arms':{name:{'final':arm['final'],'accepted':arm.get('accepted'),'attempts':[
                {'answer':a['answer'],'accepted':a.get('accepted'),'verdict':a.get('verdict'),
                 'issues':a.get('operational_issues'),'audit':a.get('audit'),'citation_resolution':a.get('citation_resolution')}
                for a in arm['attempts']]} for name,arm in row['arms'].items()}})
    result = {'split':split,'n':len(rows),'arms':basic['arms'], 'official_scoring':official['arms'],
       'v3_minus_v2':{'difference':sum(differences)/len(rows),'95ci':cluster_interval(rows,differences),
                     'corrected':sum(d==1 for d in differences),'corrupted':sum(d==-1 for d in differences)},
       'v3_minus_v2_numerical':{'n':len(numeric_rows),'difference':sum(numeric_differences)/len(numeric_rows),
           '95ci':cluster_interval(numeric_rows,numeric_differences),'corrected':sum(d==1 for d in numeric_differences),
           'corrupted':sum(d==-1 for d in numeric_differences)},
       'scalar_numerical_f1_95ci':numeric_intervals,
       'within_version_vs_single_shot':basic['paired_transitions'],'first_validator_reference_evaluation':validation,
       'loop_diagnostics':loops,'billing':basic['billing'],'integrity':{k:integrity[k] for k in ('status','questions','all_requests_and_decisions_replayed_offline')},
       'interpretation':'Development only; no generalization claim.' if split=='pilot' else 'One new holdout. Compare policies within these 50 questions; do not directly compare its point estimate to the earlier v2 100-question score.',
       'limitations':official['limitations']+['Same-model source audit can still agree with an incorrect interpretation.',
          'Canonical percentage output is a decimal; benchmark unit conventions can disagree. Frozen labels/scorer were not changed.',
          'All filings in the holdout are disjoint from prior v1/v2 development/evaluation and the new development set.']}
    write_json(folder / 'PAIRED-RESULTS.json', result)
    write_json(folder / 'ALL-ITEM-DIAGNOSTICS.json', diagnostics)
    lines = ['# Validation/refinement comparison','',result['interpretation'],'',
        '| Method | Execution correct / total | Scalar numerical F1 | Numerical items | Mean attempts |','|---|---:|---:|---:|---:|']
    for name,label in [('single_shot','Identical first answer'),('baseline_v2','Preserved v2 loop'),('agentic','Revised v3 loop')]:
        a=basic['arms'][name]
        o=official['arms'][name]
        lines.append(f"| {label} | {o['correct']} / {o['n']} | {a['numerical_f1_final_scalar']:.3f} | {a['numerical_f1_n']} | {a['mean_attempts']:.2f} |")
    d=result['v3_minus_v2_numerical']
    lines += ['',f"Numerical F1, v3 minus v2: {d['difference']:+.3f}; paired filing-bootstrap 95% interval {d['95ci']}. Corrected {d['corrected']}, corrupted {d['corrupted']} among {d['n']} numerical questions. Boolean items are included in execution accuracy and excluded from numerical F1.",'',
              'All inference, formulas, citation normalization and loop decisions replayed successfully offline. Arithmetic programs also checked against the pinned upstream FinQA evaluator.',
              '',f"This paired stage used {basic['billing']['unique_calls']} unique paid requests; reported cost ${basic['billing']['known_reported_cost_usd']:.6f}.",'',
              '## Limitations','']+['- '+x for x in result['limitations']]
    (folder / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    result=finish(p.parse_args().run)
    print({'n':result['n'],'scores':{k:v['numerical_f1_final_scalar'] for k,v in result['arms'].items()},
           'v3_minus_v2':result['v3_minus_v2'],'validator':result['first_validator_reference_evaluation']})
