"""Replay every saved request and loop decision offline with no labels or network."""
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from finrag.client import OpenRouterClient
from finrag.core import digest, jsonl, read_json, write_json
from finrag.experiment import run_experiment
from run_stage import freeze


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(folder):
    folder = Path(folder)
    data = ROOT / 'data/fresh50'
    manifest = read_json(folder / 'manifest.json')
    current = freeze()
    require(current == read_json(folder / 'stage-freeze.json'), 'Stage source/config/data changed')
    require(manifest['status'] == 'inference_complete', 'Incomplete run')
    for name in ('source_hash', 'prompt_hash', 'baseline_v2_source_hash', 'config', 'data_hash', 'index_hash'):
        require(current[name] == manifest['frozen'][name], 'Run freeze differs: ' + name)
    dataset = read_json(data / 'manifest.json')
    split = manifest['frozen']['split']
    inputs = read_json(data / (split + '.inputs.json'))
    require(digest(inputs) == dataset['splits'][split]['inputs_hash'], 'Changed inference inputs')
    require(digest(read_json(data / 'corpus.json')) == dataset['corpus_hash'], 'Changed corpus')
    index = read_json(data / 'index/manifest.json')
    require(hashlib.sha256((data / 'index/index.faiss').read_bytes()).hexdigest() == index['index_sha256'], 'Changed FAISS index')
    chunks = read_json(data / 'index/chunks.json')
    require(digest(chunks) == index['chunks_hash'], 'Changed chunks')
    chunks = {c['id']: c for c in chunks}
    rows, calls = jsonl(folder / 'results.jsonl'), jsonl(folder / 'calls.jsonl')
    by_id = {r['id']: r for r in rows}
    require(len(by_id) == len(rows) and set(by_id) == {i['id'] for i in inputs}, 'Missing/duplicate results')
    by_call = {c['request_hash']: c for c in calls}
    require(len(by_call) == len(calls), 'Duplicate paid request')
    reservations = jsonl(folder / 'budget.jsonl')
    require({r['request_hash'] for r in reservations} == set(by_call), 'Unresolved or unreserved call')
    expected_retrieval = []
    referenced = set()
    for item in inputs:
        row = by_id[item['id']]
        first = row['arms']['single_shot']['attempts'][0]
        expected_retrieval.append((item['question'], item['filename'], first['evidence']))
        for name in ('agentic', 'baseline_v2'):
            arm = row['arms'][name]
            require(arm['attempts'][0]['answer_call'] == first['answer_call'], 'Initial answers unpaired')
            require(1 <= len(arm['attempts']) <= 3, 'Attempt limit exceeded')
            for previous, nxt in zip(arm['attempts'], arm['attempts'][1:]):
                verdict = previous['verdict']
                query = (verdict['retrieval_query'].strip() if verdict else '') or item['question']
                expected_retrieval.append((item['question'] + '\n' + query, item['filename'], nxt['evidence']))
        for arm in row['arms'].values():
            for attempt in arm['attempts']:
                for key in ('answer_call', 'validator_call'):
                    if key in attempt:
                        referenced.add(attempt[key])
                for evidence in attempt['evidence']:
                    require(evidence['id'] in chunks, 'Unknown retrieved chunk')
                    require({k:v for k,v in evidence.items() if k != 'score'} == chunks[evidence['id']], 'Changed evidence')
    require(referenced == set(by_call), 'Unreferenced or missing paid replies')
    for call in calls:
        request = call['request']
        config = current['config']
        require(digest(request) == call['request_hash'], 'Changed request')
        require(request['model'] == call['returned_model'] == config['model'], 'Changed model')
        require(request['provider'] == {'only':[config['provider']], 'allow_fallbacks':False, 'require_parameters':True}, 'Changed routing')
        require(call['returned_provider'] == manifest['preflight']['endpoint']['provider_name'], 'Unexpected provider')
        require(call['usage'].get('cost') is not None, 'Missing cost')
        payload = json.loads(request['messages'][1]['content'])
        require(set(payload).issubset({'question','report_page','evidence','previous_validation_feedback','proposed_answer'}), 'Unexpected inference field')
    class ReplayIndex:
        def __init__(self):
            self.position = 0
        def retrieve(self, query, top_k, filename=None):
            require(self.position < len(expected_retrieval), 'Extra retrieval')
            expected_query, expected_filename, evidence = expected_retrieval[self.position]
            self.position += 1
            require((query, filename, top_k) == (expected_query, expected_filename, current['config']['top_k']), 'Changed retrieval query')
            return evidence
    with tempfile.TemporaryDirectory(prefix='finrag-v3-replay-') as tmp:
        target = Path(tmp)
        for name in ('calls.jsonl', 'budget.jsonl'):
            shutil.copyfile(folder / name, target / name)
        client = OpenRouterClient(current['config'], target, 'OFFLINE_NO_KEY', manifest['preflight'])
        replay_index = ReplayIndex()
        with patch('finrag.client.fetch_json', side_effect=AssertionError('Unrecorded request; network forbidden')):
            run_experiment(client, replay_index, data, split, target, current['config'], manifest['preflight'], workers=1)
        replay = {r['id']:r for r in jsonl(target / 'results.jsonl')}
        require(replay == by_id, 'Raw-record replay differs from saved outcomes')
        require(replay_index.position == len(expected_retrieval), 'Missing retrieval step')
    receipt = {'status':'passed', 'questions':len(rows), 'unique_calls':len(calls),
               'reported_cost_usd':sum(float(c['usage']['cost']) for c in calls),
               'all_requests_and_decisions_replayed_offline':True, 'labels_used_during_replay':False,
               'shared_first_answers':True, 'pending_reservations':0, **current}
    write_json(folder / 'INTEGRITY.json', receipt)
    return receipt


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    result = verify(args.run)
    print({k:result[k] for k in ('status','questions','unique_calls','reported_cost_usd')})
