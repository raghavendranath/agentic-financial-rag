"""No-key replay of every inference request, history, calculation and loop decision."""
import hashlib
import json
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from finrag.core import read_json,write_json,digest,jsonl
from finrag.client import OpenRouterClient
from run_benchmark import freeze,answer_one,answer_conversation

def require(value,message):
    if not value:raise ValueError(message)

INFERENCE_FIELDS={'question','report_page','evidence','previous_validation_feedback',
                  'proposed_answer','conversation_history','feedback','history','source_audit'}
CALL_FIELDS={'answer_call','validator_call','source_audit_call'}
OUTPUT_STATUSES={'invalid','abstained','unvalidated','validated'}

def attempt_calls(attempt):
    """Retain every stage receipt, including unsuccessful source audits."""
    require({k for k in attempt if k.endswith('_call')}.issubset(CALL_FIELDS),
            'Unrecognised paid inference stage')
    referenced={attempt[k] for k in CALL_FIELDS if k in attempt and attempt[k] is not None}
    require(all(isinstance(k,str) and k for k in referenced),'Invalid stage request hash')
    return referenced

def output_status(arm,required=False):
    status=arm.get('output_status')
    require(not required or status in OUTPUT_STATUSES,'Missing/invalid FinanceBench output status')
    require(status is None or status in OUTPUT_STATUSES,'Invalid output status')
    if status=='invalid':require(arm.get('final') is None,'Invalid draft exposed as final answer')
    if status=='validated':require(arm.get('accepted') is True,'Validated output was not accepted')
    if status in ('invalid','abstained','unvalidated'):
        require(arm.get('accepted') is not True,'Unvalidated output marked accepted')
    return status

def verify(folder):
    folder=Path(folder);manifest=read_json(folder/'manifest.json');frozen=manifest['frozen']
    require(manifest['status']=='inference_complete','Incomplete inference')
    benchmark=frozen['benchmark'];config=frozen['config'];data=ROOT/'data'/benchmark
    current=freeze(benchmark,config)
    if manifest['split']=='development' and frozen['protocol_hash']!=current['protocol_hash']:
        old_protocol=read_json(ROOT/'provenance/development-protocol.json')
        require(digest(old_protocol)==frozen['protocol_hash'],'Unexplained development protocol change')
        current={**current,'protocol_hash':frozen['protocol_hash']}
    require(frozen==current,'Source/config/data/index/runner changed')
    dataset=read_json(data/'manifest.json');split=manifest['split']
    inputs=read_json(data/(split+'.inputs.json'))
    require(digest(inputs)==dataset['splits'][split]['inputs_hash'],'Inputs changed')
    require(digest(read_json(data/'corpus.json'))==dataset['corpus_hash'],'Corpus changed')
    im=read_json(data/'index/manifest.json')
    require(hashlib.sha256((data/'index/index.faiss').read_bytes()).hexdigest()==im['index_sha256'],'FAISS vectors changed')
    chunks=read_json(data/'index/chunks.json');require(digest(chunks)==im['chunks_hash'],'Chunk metadata changed')
    chunks={c['id']:c for c in chunks}
    rows=jsonl(folder/'results.jsonl');byid={r['id']:r for r in rows}
    require(len(rows)==len(byid) and set(byid)=={i['id'] for i in inputs},'Missing/duplicate/selected results')
    calls=jsonl(folder/'calls.jsonl');bycall={c['request_hash']:c for c in calls}
    require(len(bycall)==len(calls),'Duplicate paid calls')
    require({r['request_hash'] for r in jsonl(folder/'budget.jsonl')}==set(bycall),'Unresolved/missing reservation')
    for c in calls:
        r=c['request'];require(digest(r)==c['request_hash'],'Request changed')
        require(r['model']==c['returned_model']==config['model'],'Model changed')
        require(c['returned_provider']==manifest['preflight']['endpoint']['provider_name'],'Unexpected provider')
        require(r['provider']=={'only':[config['provider']],'allow_fallbacks':False,'require_parameters':True},'Routing changed')
        require(c['usage'].get('cost') is not None,'Missing cost')
        payload=json.loads(r['messages'][1]['content'])
        require(set(payload).issubset(INFERENCE_FIELDS),'Unexpected inference field')
    retrievals=jsonl(folder/'retrievals.jsonl');byquery={r['query_hash']:r for r in retrievals}
    require(len(retrievals)==len(byquery),'Duplicate retrieval query')
    class OfflineIndex:
        def __init__(self):self.used=set()
        def retrieve(self,query,top_k=10,filename=None):
            key=digest([query,top_k,filename]);require(key in byquery,'Unrecorded retrieval')
            record=byquery[key];self.used.add(key)
            require((record['query'],record['top_k'],record['filename'])==(query,top_k,filename),'Query changed')
            evidence=[{**chunks[h['id']],'score':h['score']} for h in record['hits']]
            require(digest(evidence)==record['evidence_hash'],'Evidence changed')
            require(all(e['filename'].rsplit('/',1)[0]==filename.rsplit('/',1)[0] for e in evidence),'Wrong filing retrieved')
            return evidence
    with tempfile.TemporaryDirectory(prefix='three-benchmark-replay-') as tmp:
        for name in ('calls.jsonl','budget.jsonl'):shutil.copyfile(folder/name,Path(tmp)/name)
        client=OpenRouterClient(config,tmp,'OFFLINE_NO_KEY',manifest['preflight'])
        index=OfflineIndex();lock=threading.RLock();replayed={}
        with patch('finrag.client.fetch_json',side_effect=AssertionError('Unrecorded request; network forbidden')):
            for item in inputs:
                row=(answer_conversation if benchmark=='convfinqa' else answer_one)(client,index,lock,item,config)
                replayed[row['id']]=row
        require(replayed==byid,'Replay differs from recorded answers, histories or decisions')
        require(index.used==set(byquery),'Unreferenced retrieval records')
    # Verify all paid responses belong to saved final trajectories, including unsuccessful attempts.
    referenced=set();attempts=0
    for row in rows:
        turns=row['turns'] if benchmark=='convfinqa' else [row]
        for turn in turns:
            for arm in turn['arms'].values():
                output_status(arm,required=benchmark=='financebench')
                require(1<=len(arm['attempts'])<=3,'Exceeded attempt limit')
                for a in arm['attempts']:
                    attempts+=1
                    if benchmark=='financebench':
                        require(a.get('output_status') in OUTPUT_STATUSES,'Missing/invalid attempt output status')
                        if a['output_status']=='invalid':require(a.get('answer') is None,'Invalid attempt exposed as answer')
                    referenced.update(attempt_calls(a))
                if benchmark=='financebench':
                    require(arm['output_status']==arm['attempts'][-1]['output_status'],'Arm/last-attempt output status differs')
    require(referenced==set(bycall),'Unreferenced or missing paid replies')
    result={'status':'passed','units':len(rows),'attempt_records':attempts,'unique_calls':len(calls),
            'reported_cost_usd':sum(float(c['usage']['cost']) for c in calls),'pending_reservations':0,
            'offline_replayed_all_requests_histories_and_decisions':True,'labels_used_in_replay':False,
            'source_audit_receipts_included':True,'output_statuses_checked':True,
            'index_vectors_recomputed':False,'retrieval_records_checked_against_frozen_corpus':True,'frozen':frozen}
    write_json(folder/'INTEGRITY.json',result)
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();r=verify(ROOT/a.run)
    print({k:r[k] for k in ('status','units','unique_calls','reported_cost_usd')})
