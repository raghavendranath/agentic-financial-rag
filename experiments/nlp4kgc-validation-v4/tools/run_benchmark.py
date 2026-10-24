"""Bounded, immutable paired benchmark run; inference never loads references."""
import argparse
import fcntl
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from finrag.core import read_json,write_json,digest,jsonl,append_jsonl
from finrag.client import OpenRouterClient,read_key,preflight,API,fetch_json
from finrag.dense import DenseIndex
from finrag.experiment import source_hash
from finrag import benchmarks as task

def account(key):
    data=fetch_json(API+'/key',key=key,timeout=25)['data']
    result={k:data.get(k) for k in ('usage','limit','limit_remaining')}
    result['utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    if result['limit']!=20 or not result['limit_remaining'] or result['limit_remaining']<=0:raise ValueError('Dedicated $20 account cap check failed')
    return result

def freeze(benchmark,config):
    data=ROOT/'data'/benchmark
    return {'source_hash':source_hash(),'config':config,'data_hash':digest(read_json(data/'manifest.json')),
            'index_hash':digest(read_json(data/'index/manifest.json')),'benchmark':benchmark,
            'runner_hash':digest((ROOT/'tools/run_benchmark.py').read_text()),
            'protocol_hash':digest(read_json(ROOT/'docs/PROTOCOL.json'))}

class RecordingIndex:
    def __init__(self,index,folder):
        self.index=index;self.path=Path(folder)/'retrievals.jsonl'
        self.chunks={c['id']:c for c in index.chunks}
        self.cache={r['query_hash']:r for r in jsonl(self.path)}
    def retrieve(self,query,top_k=10,filename=None):
        key=digest([query,top_k,filename])
        if key not in self.cache:
            evidence=self.index.retrieve(query,top_k,filename=filename)
            record={'query_hash':key,'query':query,'top_k':top_k,'filename':filename,
                    'hits':[{'id':e['id'],'score':e['score']} for e in evidence],'evidence_hash':digest(evidence)}
            append_jsonl(self.path,record);self.cache[key]=record
        record=self.cache[key]
        evidence=[{**self.chunks[h['id']],'score':h['score']} for h in record['hits']]
        if digest(evidence)!=record['evidence_hash']:raise ValueError('Cached retrieval evidence changed')
        return evidence

class ItemIndex:
    def __init__(self,index,lock,item):self.index,self.lock,self.item=index,lock,item
    def retrieve(self,query,top_k=10,filename=None):
        if self.item.get('history'):
            query+='\nPrior questions: '+' | '.join(h['question'] for h in self.item['history'][-3:])
        with self.lock:return self.index.retrieve(query,top_k,filename=filename)

def answer_one(client,index,index_lock,item,config):
    local=ItemIndex(index,index_lock,item)
    evidence=local.retrieve(item['question'],config['top_k'],filename=item['filename'])
    initial=task.draft(client,item,evidence,item['id']+':initial')
    first=task.validate(client,item,initial,item['id']+':validate')
    arms={'single_shot':{'final':initial['answer'],'attempts':[initial],
                               **({'output_status':initial['output_status']} if 'output_status' in initial else {})},
          'agentic':task.trajectory(client,local,item,first,config,True)}
    if item['benchmark']=='financebench' and config['fixed_retrieval_ablation']:
        arms['fixed_retrieval']=task.trajectory(client,local,item,first,config,False)
    return {'id':item['id'],'filename':item['filename'],'question':item['question'],'benchmark':item['benchmark'],'arms':arms}

def answer_conversation(client,index,index_lock,conversation,config):
    histories={'single_shot':[],'agentic':[]};turns=[]
    for turn in range(len(conversation['questions'])):
        arms={};items={}
        for name in ('single_shot','agentic'):
            item=task.conversational_item(conversation,turn,histories[name]);items[name]=item
            local=ItemIndex(index,index_lock,item)
            evidence=local.retrieve(item['question'],config['top_k'],filename=item['filename'])
            initial=task.draft(client,item,evidence,item['id']+':'+name+':initial')
            if name=='agentic':
                first=task.validate(client,item,initial,item['id']+':'+name+':validate')
                arms[name]=task.trajectory(client,local,item,first,config,True)
            else:arms[name]={'final':initial['answer'],'attempts':[initial]}
            histories[name].append(task.history_entry(item['question'],arms[name]['final']))
        turns.append({'turn':turn,'question':conversation['questions'][turn],'items':items,'arms':arms})
    return {'id':conversation['id'],'filename':conversation['filename'],'benchmark':'convfinqa','turns':turns}

def main():
    p=argparse.ArgumentParser();p.add_argument('--benchmark',choices=['financebench','convfinqa'],required=True)
    p.add_argument('--split',choices=['development','evaluation'],required=True);p.add_argument('--out',required=True)
    p.add_argument('--workers',type=int,choices=[1,2,3],default=3);p.add_argument('--device',default='mps')
    a=p.parse_args();folder=ROOT/a.out;folder.mkdir(parents=True,exist_ok=True)
    data=ROOT/'data'/a.benchmark;config=read_json(ROOT/'configs/deepseek.json');frozen=freeze(a.benchmark,config)
    if a.split=='evaluation' and frozen!=read_json(ROOT/'docs'/(a.benchmark+'-FREEZE.json')):raise ValueError('Freeze benchmark before evaluation')
    inputs=read_json(data/(a.split+'.inputs.json'));dataset=read_json(data/'manifest.json')
    if digest(inputs)!=dataset['splits'][a.split]['inputs_hash']:raise ValueError('Frozen inputs changed')
    if not 1<=config['max_attempts']<=3:raise ValueError('Attempt limit exceeded')
    with (folder/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        key=read_key(ROOT.parent/'nlp4kgc-reconstruction/.secrets/openrouter.key')
        receipt=account(key);metadata=preflight(config,key)
        if (folder/'manifest.json').exists():
            manifest=read_json(folder/'manifest.json')
            if manifest['frozen']!=frozen or manifest['split']!=a.split:raise ValueError('Run source/settings changed')
        else:
            write_json(folder/'account-before.json',receipt)
            manifest={'frozen':frozen,'split':a.split,'workers':a.workers,'device':a.device,'preflight':metadata,'status':'in_progress','created_utc':receipt['utc']}
            write_json(folder/'manifest.json',manifest)
        existing={r['id'] for r in jsonl(folder/'results.jsonl')}
        client=OpenRouterClient(config,folder,key,metadata)
        index=DenseIndex(revision='5617a9f61b028005a4858fdac845db406aefb181',device=a.device);index.load(data/'index')
        index=RecordingIndex(index,folder)
        guard=threading.RLock();stop=threading.Event()
        def one(i,item):
            if stop.is_set():return None
            print(a.benchmark,a.split,str(i)+'/'+str(len(inputs)),item['id'],flush=True)
            return (answer_conversation if a.benchmark=='convfinqa' else answer_one)(client,index,guard,item,config)
        errors=[]
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            jobs=[pool.submit(one,i,item) for i,item in enumerate(inputs,1) if item['id'] not in existing]
            for job in as_completed(jobs):
                try:
                    row=job.result()
                    if row:append_jsonl(folder/'results.jsonl',row)
                except Exception as e:stop.set();errors.append(type(e).__name__+': '+str(e))
        write_json(folder/'account-after.json',account(key))
        if errors:raise RuntimeError('Run interrupted; saved replies retained. '+errors[0])
        manifest['status']='inference_complete';write_json(folder/'manifest.json',manifest)
        print('Complete',a.benchmark,a.split,flush=True)

if __name__=='__main__':main()
