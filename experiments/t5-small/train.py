"""One fixed local T5-small training run; reads training/development only."""
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parent

def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def main():
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text());cfg=protocol['config']
    if (ROOT/'TRAINING-COMPLETE.json').exists():raise ValueError('Fixed training is already complete')
    for name in ('train.json','development.json'):
        data=json.loads((ROOT/'data'/name).read_text())
        canonical=json.dumps(data,sort_keys=True,ensure_ascii=False,separators=(',',':'))
        if hashlib.sha256(canonical.encode()).hexdigest()!=protocol['data_hashes'][name]:raise ValueError('Training data changed')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.set_num_threads(4)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    directory=snapshot_download(cfg['model'],revision=cfg['model_revision'],cache_dir=str(ROOT/'.cache/huggingface'),
        allow_patterns=['config.json','generation_config.json','model.safetensors','tokenizer.json','tokenizer_config.json','spiece.model','special_tokens_map.json'])
    tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,use_fast=True,trust_remote_code=False)
    model=AutoModelForSeq2SeqLM.from_pretrained(directory,local_files_only=True,trust_remote_code=False,attn_implementation='eager').to(device)
    model.config.use_cache=False
    stats={};encoded={}
    for split,name in (('train','train.json'),('development','development.json')):
        raw=json.loads((ROOT/'data'/name).read_text());enc=[];truncated=targets_truncated=0
        for row in raw:
            x=tokenizer(row['text'],truncation=False)['input_ids'];y=tokenizer(text_target=row['program'],truncation=False)['input_ids']
            truncated+=len(x)>cfg['max_input_tokens'];targets_truncated+=len(y)>cfg['max_target_tokens']
            # Ensure EOS is retained if a long context is truncated.
            x=x[:cfg['max_input_tokens']];x[-1]=tokenizer.eos_token_id
            y=y[:cfg['max_target_tokens']];y[-1]=tokenizer.eos_token_id
            enc.append({'input_ids':x,'labels':y})
        encoded[split]=enc;stats[split]={'examples':len(raw),'truncated_inputs':truncated,'truncated_targets':targets_truncated}
    if any(s['truncated_targets'] for s in stats.values()):raise ValueError('Training target truncation must be addressed before fixed run')
    def collate(rows):
        padded=tokenizer.pad([{'input_ids':r['input_ids']} for r in rows],padding=True,return_tensors='pt')
        length=max(len(r['labels']) for r in rows)
        padded['labels']=torch.tensor([r['labels']+[-100]*(length-len(r['labels'])) for r in rows])
        return {k:v.to(device) for k,v in padded.items()}
    generator=torch.Generator().manual_seed(cfg['seed'])
    train=DataLoader(encoded['train'],batch_size=cfg['batch_size'],shuffle=True,generator=generator,collate_fn=collate)
    dev=DataLoader(encoded['development'],batch_size=cfg['batch_size'],shuffle=False,collate_fn=collate)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    total=len(train)*cfg['epochs'];warmup=max(1,int(total*cfg['warmup_fraction']))
    step=0;start=time.monotonic();history=[]
    state={'status':'training','device':device,'model':cfg,'data':stats,'total_steps':total,'torch':torch.__version__,
           'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),
           'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'evaluation_labels_loaded':False}
    write(ROOT/'TRAINING-STATUS.json',state)
    for epoch in range(cfg['epochs']):
        model.train();loss_sum=0;examples=0
        for batch in train:
            step+=1;scale=min(step/warmup,max(0.0,(total-step)/max(1,total-warmup)))
            for group in optimizer.param_groups:group['lr']=cfg['learning_rate']*scale
            optimizer.zero_grad(set_to_none=True);loss=model(**batch).loss
            if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);optimizer.step()
            count=batch['input_ids'].shape[0];examples+=count;loss_sum+=float(loss.detach().cpu())*count
            if step%25==0 or step==1:
                progress={'epoch':epoch+1,'step':step,'total_steps':total,'training_loss':loss_sum/examples,'elapsed_seconds':time.monotonic()-start}
                print(json.dumps(progress),flush=True);write(ROOT/'TRAINING-STATUS.json',{**state,**progress})
        model.eval();dev_loss=0;dev_n=0
        with torch.inference_mode():
            for batch in dev:
                n=batch['input_ids'].shape[0];dev_loss+=float(model(**batch).loss.cpu())*n;dev_n+=n
        record={'epoch':epoch+1,'training_loss':loss_sum/examples,'development_loss':dev_loss/dev_n,'elapsed_seconds':time.monotonic()-start}
        history.append(record);print(json.dumps(record),flush=True)
        model.save_pretrained(ROOT/'checkpoint',safe_serialization=True);tokenizer.save_pretrained(ROOT/'checkpoint')
        write(ROOT/'EPOCHS.json',history)
    receipt={**state,'status':'training_complete','steps':step,'epochs':history,'elapsed_seconds':time.monotonic()-start,
             'checkpoint_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'checkpoint').iterdir() if p.is_file()}}
    write(ROOT/'TRAINING-COMPLETE.json',receipt);write(ROOT/'TRAINING-STATUS.json',receipt)
    print('TRAINING COMPLETE',flush=True)

if __name__=='__main__':main()
