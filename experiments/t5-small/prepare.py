"""Prepare a documented new T5-small baseline, never the lost historical T5."""
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
sys.path.insert(0,str(PARENT/'nlp4kgc-three-benchmarks/src'))
from finrag.core import public_item,digest,write_json,read_json,jsonl
COMMIT='0f16e2867befa6840783e58be38c9efb9229d742'

def filing(row):return '/'.join(row['filename'].split('/')[:2])

def input_text(item):
    """Question-only lexical rank of public rows; no annotation/model_input fields."""
    q=set(re.findall(r'[a-z0-9]+',item['question'].lower()))
    ranked=[]
    for i,c in enumerate(item['chunks']):
        terms=set(re.findall(r'[a-z0-9]+',c['text'].lower()))
        score=len(q&terms)/(max(1,len(terms))**0.5)
        ranked.append((-score,i,c['text']))
    context='\n'.join(x[2] for x in sorted(ranked)[:10])
    return 'generate financial program: question: '+item['question']+' context: '+context

def main():
    raw=ROOT/'data/raw';raw.mkdir(parents=True,exist_ok=True)
    sources={}
    for split in ('train','dev','test'):
        path=raw/(split+'.json');url=f'https://raw.githubusercontent.com/czyssrs/FinQA/{COMMIT}/dataset/{split}.json'
        if not path.exists():path.write_bytes(urllib.request.urlopen(url,timeout=60).read())
        sources[split]={'url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    train=read_json(raw/'train.json');dev=read_json(raw/'dev.json');test=read_json(raw/'test.json')
    evaluation=read_json(PARENT/'nlp4kgc-validation-v3/data/fresh50/evaluation.inputs.json')
    ids={r['id'] for r in evaluation};evaluation_raw=[r for r in test if r['id'] in ids]
    if len(evaluation_raw)!=50:raise ValueError('Missing frozen test questions')
    test_filings={filing(r) for r in evaluation_raw}
    development=sorted([r for r in dev if filing(r) not in test_filings],key=lambda r:digest(['t5-dev-18',r['id']]))[:100]
    dev_filings={filing(r) for r in development}
    # Disallow held-out filings, including differently worded questions on another page.
    training=[r for r in train if filing(r) not in test_filings|dev_filings]
    training_filings={filing(r) for r in training}
    if not training or len(development)!=100:raise ValueError('Insufficient disjoint training/development data')
    def supervised(raw):
        item=public_item(raw)
        return {'id':raw['id'],'filename':raw['filename'],'text':input_text(item),'program':raw['qa']['program']}
    tr=[supervised(r) for r in training];dv=[supervised(r) for r in development]
    # Evaluation uses the *same serialized public rows and ranking* as training.
    # Unlike the DeepSeek/Qwen pipeline, this is a small supervised lexical-reader
    # baseline. It is not a matched BGE-RAG ablation.
    ev=[{'id':r['id'],'filename':r['filename'],'text':input_text(public_item(r)),
         'table':r.get('table_ori') or r['table']} for r in evaluation_raw]
    byid={r['id']:r for r in ev};ev=[byid[r['id']] for r in evaluation]
    labels={r['id']:{'gold':r['qa']['exe_ans'],'program':r['qa']['program']} for r in evaluation_raw}
    write_json(ROOT/'data/train.json',tr);write_json(ROOT/'data/development.json',dv)
    write_json(ROOT/'data/evaluation.inputs.json',ev);write_json(ROOT/'data/evaluation.labels.json',labels)
    info=json.load(urllib.request.urlopen('https://huggingface.co/api/models/google-t5/t5-small',timeout=30))
    config={'model':'google-t5/t5-small','model_revision':info['sha'],'seed':18,'epochs':3,'batch_size':8,
            'learning_rate':0.0003,'weight_decay':0.01,'warmup_fraction':0.1,'gradient_clip':1.0,
            'max_input_tokens':512,'max_target_tokens':128,'num_beams':1,'do_sample':False,
            'selection':'final checkpoint after exactly three epochs; no test-driven checkpoint or hyperparameter selection',
            'reader':'top ten public page rows by deterministic query-token overlap; same at training/development/evaluation',
            'scope':'New T5-small supervised program generator; not the historical 0.76 model and not a matched BGE-RAG ablation'}
    manifest={'config':config,'sources':sources,'source_commit':COMMIT,'train':len(tr),'official_train_total':len(train),
        'excluded_train_filing_overlap':len(train)-len(tr),'development':len(dv),'evaluation':len(ev),
        'data_hashes':{n:digest(read_json(ROOT/'data'/n)) for n in ('train.json','development.json','evaluation.inputs.json','evaluation.labels.json')},
        'disjoint_train_dev_test_filings':not(training_filings&{filing(r) for r in development} or training_filings&test_filings or {filing(r) for r in development}&test_filings),
        'evaluation_ids_equal_frozen_finqa':set(ids)=={r['id'] for r in ev},'additional_api_cost_usd':0}
    write_json(ROOT/'PROTOCOL.json',manifest)
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
