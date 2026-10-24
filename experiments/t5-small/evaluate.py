"""Generate all 50 predictions without labels, then score in a separate process."""
import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'nlp4kgc-three-benchmarks'
sys.path.insert(0,str(SOURCE/'src'))
from finrag.core import read_json,write_json,digest,correct,number
from finrag.report import cluster_interval

def upstream():
    spec=importlib.util.spec_from_file_location('official',SOURCE/'provenance/finqa-official-evaluate.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def generate(device):
    import torch
    from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
    protocol=read_json(ROOT/'PROTOCOL.json');cfg=protocol['config'];receipt=read_json(ROOT/'TRAINING-COMPLETE.json')
    if receipt['status']!='training_complete':raise ValueError('Training incomplete')
    for name,sha in receipt['checkpoint_hashes'].items():
        if hashlib.sha256((ROOT/'checkpoint'/name).read_bytes()).hexdigest()!=sha:raise ValueError('Checkpoint changed')
    if hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest()!=receipt['protocol_sha256']:raise ValueError('Protocol changed')
    inputs=read_json(ROOT/'data/evaluation.inputs.json')
    if digest(inputs)!=protocol['data_hashes']['evaluation.inputs.json']:raise ValueError('Evaluation inputs changed')
    if (ROOT/'PREDICTIONS.json').exists():raise ValueError('Predictions already frozen; do not regenerate based on test results')
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'checkpoint',local_files_only=True,trust_remote_code=False)
    model=AutoModelForSeq2SeqLM.from_pretrained(ROOT/'checkpoint',local_files_only=True,trust_remote_code=False,attn_implementation='eager').to(device)
    model.eval();model.config.use_cache=True;predictions=[];torch.manual_seed(cfg['seed'])
    with torch.inference_mode():
        for i,row in enumerate(inputs,1):
            tokens=tokenizer(row['text'],return_tensors='pt',truncation=True,max_length=cfg['max_input_tokens']).to(device)
            result=model.generate(**tokens,max_new_tokens=cfg['max_target_tokens'],num_beams=cfg['num_beams'],do_sample=False)
            raw=tokenizer.decode(result[0],skip_special_tokens=True)
            predictions.append({'id':row['id'],'filename':row['filename'],'raw_program':raw,
                'input_hash':digest(row),'truncated_input':len(tokenizer(row['text'])['input_ids'])>cfg['max_input_tokens']})
            print(i,row['id'],raw,flush=True)
    write_json(ROOT/'PREDICTIONS.json',predictions)
    write_json(ROOT/'GENERATION-RECEIPT.json',{'checkpoint_hashes':receipt['checkpoint_hashes'],'evaluation_labels_loaded':False,
        'prediction_hash':digest(predictions),'items':len(predictions),'device':device,'sampling':False})

def score():
    protocol=read_json(ROOT/'PROTOCOL.json');inputs=read_json(ROOT/'data/evaluation.inputs.json')
    predictions=read_json(ROOT/'PREDICTIONS.json');labels=read_json(ROOT/'data/evaluation.labels.json')
    if digest(labels)!=protocol['data_hashes']['evaluation.labels.json']:raise ValueError('Labels changed')
    if digest(predictions)!=read_json(ROOT/'GENERATION-RECEIPT.json')['prediction_hash']:raise ValueError('Predictions changed')
    if [p['id'] for p in predictions]!=[r['id'] for r in inputs]:raise ValueError('Incomplete predictions')
    up=upstream();items=[];numeric=[];program_matches=0;invalids=0;nh=[];allhits=[]
    for row,pred in zip(inputs,predictions):
        if pred['input_hash']!=digest(row):raise ValueError('Input changed')
        raw=re.sub(r',\s*',', ',pred['raw_program'].strip())
        tokens=up.program_tokenization(raw)
        invalid,result=up.eval_program(tokens,row['table'])
        value=None if invalid else result
        hit=int(correct(value,labels[row['id']]['gold']));allhits.append(hit)
        pm=False
        if not invalid:
            try:pm=bool(up.equal_program(tokens,up.program_tokenization(labels[row['id']]['program'])))
            except (ValueError,KeyError,AssertionError,IndexError,TypeError):pm=False
        program_matches+=pm;invalids+=bool(invalid)
        item={'id':row['id'],'filename':row['filename'],'raw_program':pred['raw_program'],'value':value,
            'gold':labels[row['id']]['gold'],'gold_program':labels[row['id']]['program'],
            'execution_correct':bool(hit),'program_correct':pm,'invalid_program':bool(invalid),'truncated_input':pred['truncated_input']}
        items.append(item)
        if isinstance(number(item['gold']),(int,float)):numeric.append(item);nh.append(hit)
    result={'status':'complete','model':'New supervised T5-small program generator','items':len(items),
        'correct':sum(allhits),'execution_accuracy':sum(allhits)/len(allhits),'execution_accuracy_95ci':cluster_interval(items,allhits),
        'numerical_items':len(nh),'numerical_correct':sum(nh),'final_scalar_numerical_f1':sum(nh)/len(nh),
        'numerical_f1_95ci':cluster_interval(numeric,nh),'program_accuracy':program_matches/len(items),
        'invalid_programs':invalids,'truncated_evaluation_inputs':sum(p['truncated_input'] for p in predictions),
        'training_examples':protocol['train'],'epochs':protocol['config']['epochs'],'additional_api_cost_usd':0,
        'same_50_evaluation_ids_as_deepseek_and_qwen':protocol['evaluation_ids_equal_frozen_finqa'],
        'training_development_evaluation_filing_disjoint':protocol['disjoint_train_dev_test_filings'],
        'not_a_reproduction_of_historical_t5':True,'retrieval_matched_to_bge_rag':False,
        'limitations':['Small T5-small model with one fixed three-epoch training run.',
            'Uses top ten lexical-ranked public page rows and a 512-token input limit; DeepSeek/Qwen use BGE and longer evidence.',
            'Cannot attribute differences solely to model size or training: retrieval, context length and output format also differ.',
            'No checkpoint or hyperparameter selected using test labels. No gold supporting facts or legacy model_input enter reader input.']}
    write_json(ROOT/'ITEM-SCORES.json',items);write_json(ROOT/'SCORES.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['generate','score']);p.add_argument('--device',default='mps');a=p.parse_args()
    generate(a.device) if a.stage=='generate' else score()
