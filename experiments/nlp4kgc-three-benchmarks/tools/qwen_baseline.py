"""Paper-named single-shot baseline, using the same recorded retrieval evidence."""
import argparse
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from finrag import benchmarks as task
from finrag.client import OpenRouterClient,preflight,read_key
from finrag.core import read_json,write_json,jsonl,append_jsonl,digest
from finrag.experiment import source_hash
from run_benchmark import account,ItemIndex
import threading

def sources(benchmark):
    if benchmark=='finqa':
        base=ROOT.parent/'nlp4kgc-validation-v3'
        return base/'data/fresh50',base/'runs/holdout50','evaluation'
    return ROOT/'data'/benchmark,ROOT/'runs'/(benchmark+'-evaluation'),'evaluation'

def frozen_for(benchmark):
    data,source,split=sources(benchmark)
    manifest=read_json(source/'manifest.json')
    if manifest['status']!='inference_complete':raise ValueError('Finish the corresponding DeepSeek run first')
    return {'benchmark':benchmark,'source_hash':source_hash(),'config':read_json(ROOT/'configs/qwen-baseline.json'),
        'data_hash':digest(read_json(data/'manifest.json')),'source_manifest_hash':digest(manifest),
        'source_results_hash':digest(jsonl(source/'results.jsonl')),'script_hash':digest((ROOT/'tools/qwen_baseline.py').read_text())}

class RecordedIndex:
    def __init__(self,data,source,folder):
        self.chunks={c['id']:c for c in read_json(data/'index/chunks.json')}
        self.records={r['query_hash']:r for r in jsonl(source/'retrievals.jsonl')}
        self.path=folder/'retrievals.jsonl';self.used={r['query_hash'] for r in jsonl(self.path)}
    def retrieve(self,query,top_k=10,filename=None):
        key=digest([query,top_k,filename])
        if key not in self.records:raise ValueError('Baseline query is absent from the shared retrieval record')
        r=self.records[key]
        evidence=[{**self.chunks[h['id']],'score':h['score']} for h in r['hits']]
        if digest(evidence)!=r['evidence_hash']:raise ValueError('Source retrieval changed')
        if key not in self.used:append_jsonl(self.path,r);self.used.add(key)
        return evidence

def one(client,benchmark,item,source_row,index,config):
    if benchmark=='convfinqa':
        history=[];turns=[]
        for t in range(len(item['questions'])):
            current=task.conversational_item(item,t,history)
            local=ItemIndex(index,threading.RLock(),current)
            evidence=local.retrieve(current['question'],config['top_k'],filename=current['filename'])
            initial=task.draft(client,current,evidence,current['id']+':qwen:initial')
            arm={'final':initial['answer'],'attempts':[initial]}
            turns.append({'turn':t,'question':current['question'],'items':{'qwen_single_shot':current},'arms':{'qwen_single_shot':arm}})
            history.append(task.history_entry(current['question'],initial['answer']))
        return {'id':item['id'],'filename':item['filename'],'benchmark':benchmark,'turns':turns}
    current={**item,'benchmark':benchmark}
    evidence=source_row['arms']['single_shot']['attempts'][0]['evidence']
    initial=task.draft(client,current,evidence,item['id']+':qwen:initial')
    return {'id':item['id'],'filename':item['filename'],'question':item['question'],'benchmark':benchmark,
            'arms':{'qwen_single_shot':{'final':initial['answer'],'attempts':[initial]}}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--benchmark',choices=['finqa','financebench','convfinqa'],required=True);a=p.parse_args()
    benchmark=a.benchmark;data,source,split=sources(benchmark);folder=ROOT/'runs'/('qwen-'+benchmark);folder.mkdir(exist_ok=True)
    frozen=frozen_for(benchmark);config=frozen['config'];dataset=read_json(data/'manifest.json');inputs=read_json(data/(split+'.inputs.json'))
    if digest(inputs)!=dataset['splits'][split]['inputs_hash']:raise ValueError('Inputs changed')
    key=read_key(ROOT.parent/'nlp4kgc-reconstruction/.secrets/openrouter.key');metadata=preflight(config,key)
    if (folder/'manifest.json').exists():
        if read_json(folder/'manifest.json')['frozen']!=frozen:raise ValueError('Baseline source/settings changed')
    else:
        write_json(folder/'account-before.json',account(key));write_json(folder/'manifest.json',{'frozen':frozen,'split':split,'preflight':metadata,'status':'in_progress'})
    client=OpenRouterClient(config,folder,key,metadata);existing={r['id'] for r in jsonl(folder/'results.jsonl')}
    source_rows={r['id']:r for r in jsonl(source/'results.jsonl')}
    index=RecordedIndex(data,source,folder) if benchmark=='convfinqa' else None
    for i,item in enumerate(inputs,1):
        if item['id'] in existing:continue
        row=one(client,benchmark,item,source_rows[item['id']],index,config)
        append_jsonl(folder/'results.jsonl',row);print('Qwen',benchmark,i,'/',len(inputs),flush=True)
    manifest=read_json(folder/'manifest.json');manifest['status']='inference_complete';write_json(folder/'manifest.json',manifest)
    write_json(folder/'account-after.json',account(key))

if __name__=='__main__':main()
