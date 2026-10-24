"""Run one development or frozen holdout stage with budget receipt and stall guard."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT.parent / 'nlp4kgc-reconstruction'
sys.path.insert(0, str(ROOT / 'src'))
from finrag.client import API, fetch_json, read_key
from finrag.core import digest, jsonl, prompt_hash, read_json, write_json
from finrag.experiment import source_hash
from finrag.baseline import baseline_hash


def account():
    data = fetch_json(API + '/key', key=read_key(ORIGINAL / '.secrets/openrouter.key'), timeout=25)['data']
    receipt = {k: data.get(k) for k in ('usage', 'limit', 'limit_remaining')}
    receipt['utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    if receipt['limit'] != 20 or not receipt['limit_remaining'] or receipt['limit_remaining'] <= 0:
        raise ValueError('Dedicated $20 limit or remaining credit check failed')
    return receipt


def freeze():
    return {'source_hash': source_hash(), 'prompt_hash': prompt_hash(),
            'baseline_v2_source_hash': baseline_hash(),
            'config': read_json(ROOT / 'configs/deepseek.json'),
            'data_hash': digest(read_json(ROOT / 'data/fresh50/manifest.json')),
            'index_hash': digest(read_json(ROOT / 'data/fresh50/index/manifest.json'))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['pilot', 'evaluation'], required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError('Use a simple run name')
    folder = ROOT / 'runs' / args.name
    folder.mkdir(exist_ok=True)
    current = freeze()
    if args.split == 'evaluation' and current != read_json(ROOT / 'docs/HOLDOUT-FREEZE.json'):
        raise ValueError('Freeze code/config/data before the holdout')
    write_json(folder / 'stage-freeze.json', current)
    write_json(folder / 'account-before.json', account())
    if (folder / 'budget.jsonl').exists():
        received = {c['request_hash'] for c in jsonl(folder / 'calls.jsonl')}
        if any(c['request_hash'] not in received for c in jsonl(folder / 'budget.jsonl')):
            raise ValueError('Unresolved request; account reconciliation required before any retry')
    env = os.environ.copy()
    env.update(HF_HOME=str(ORIGINAL / '.cache/huggingface'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    command = [sys.executable, str(ROOT / 'run.py'), '--config', str(ROOT / 'configs/deepseek.json'),
               'run', '--data', str(ROOT / 'data/fresh50'), '--split', args.split,
               '--out', str(folder), '--key-file', str(ORIGINAL / '.secrets/openrouter.key'), '--workers', '1']
    with (folder / 'console.log').open('a') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        sizes, progress, stalled = None, time.monotonic(), False
        try:
            while child.poll() is None:
                time.sleep(2)
                now = tuple((folder / n).stat().st_size if (folder / n).exists() else 0 for n in ('calls.jsonl', 'results.jsonl'))
                if now != sizes:
                    sizes, progress = now, time.monotonic()
                if time.monotonic() - progress > 150:
                    stalled = True
                    child.terminate()
                    break
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=15)
    write_json(folder / 'account-after.json', account())
    write_json(folder / 'execution-status.json', {'returncode': child.returncode, 'stalled': stalled,
               'questions_saved': len(jsonl(folder / 'results.jsonl')), 'automatic_ambiguous_retries': 0})
    if child.returncode or stalled:
        raise RuntimeError('Stage stopped; saved replies retained. Inspect status before any retry.')
    print('Stage complete:', args.name)


if __name__ == '__main__':
    main()
