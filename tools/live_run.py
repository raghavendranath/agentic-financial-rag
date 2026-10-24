#!/usr/bin/env python3
"""Run fresh benchmark inference with the documented adapter; never load labels."""
from __future__ import annotations
import argparse,importlib.util,json,os,sys,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT={'finqa':'nlp4kgc-validation-v3','convfinqa':'nlp4kgc-three-benchmarks','financebench':'nlp4kgc-validation-v4'}

def output_path(value):
    p=Path(value).expanduser().resolve()
    if p==ROOT or ROOT/'experiments'==p or (ROOT/'experiments') in p.parents:
        raise ValueError('Use a separate output directory outside the published scientific snapshots')
    if p.exists():raise ValueError('Output already exists; use a new directory. Inspect ambiguous failures before any continuation.')
    return p

def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',choices=PROJECT,required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--split',choices=['development','evaluation'],default='development')
    p.add_argument('--device',default='cpu')
    p.add_argument('--budget-usd',type=float,default=20)
    p.add_argument('--key-file',type=Path)
    p.add_argument('--config',type=Path,help='Optional new config; published snapshots are never edited')
    a=p.parse_args(argv)
    if not 0<a.budget_usd<10000:p.error('Provide a finite positive budget below 10000 USD')
    return a

def main():
    a=arguments();out=output_path(a.out)
    project=ROOT/'experiments'/PROJECT[a.benchmark]
    sys.path[:0]=[str(project/'src'),str(project/'tools')]
    from finrag.client import OpenRouterClient,read_key,preflight
    from finrag.core import read_json,write_json,jsonl,append_jsonl,digest
    from finrag.dense import DenseIndex
    from finrag.experiment import run_experiment,source_hash
    config=read_json(a.config or project/'configs/deepseek.json')
    config={**config,'budget_usd':a.budget_usd,'paired_v2_baseline':False}
    if not 1<=config['max_attempts']<=3:raise ValueError('Use one to three answer attempts')
    data=project/('data/fresh50' if a.benchmark=='finqa' else 'data/'+a.benchmark)
    split='pilot' if a.benchmark=='finqa' and a.split=='development' else a.split
    inputs=read_json(data/(split+'.inputs.json'));dataset=read_json(data/'manifest.json')
    if digest(inputs)!=dataset['splits'][split]['inputs_hash']:raise ValueError('Input hash mismatch')
    # References and gold programs are deliberately never opened here.
    key=read_key(a.key_file)
    metadata=preflight(config,key)
    index_manifest=read_json(data/'index/manifest.json')
    index=DenseIndex(revision=index_manifest['parameters']['revision'],device=a.device)
    index.load(data/'index')
    out.mkdir(parents=True)
    write_json(out/'config.json',config)
    write_json(out/'preflight.json',metadata)
    write_json(out/'data-manifest.json',dataset)
    client=OpenRouterClient(config,out,key,metadata)
    if a.benchmark=='finqa':
        run_experiment(client,index,data,split,out,config,metadata,workers=1)
    else:
        import run_benchmark as runner
        # Use the original answer/evidence functions without its session-specific account supervisor.
        frozen={'config':config,'source_hash':source_hash(),'data_hash':digest(dataset),
                'index_hash':digest(index_manifest),'benchmark':a.benchmark,
                'adapter_runner_sha256':__import__('hashlib').sha256(Path(runner.__file__).read_bytes()).hexdigest()}
        manifest={'status':'in_progress','frozen':frozen,'split':split,'preflight':metadata,
                  'live_wrapper_version':1,'inference_reads_references':False}
        write_json(out/'manifest.json',manifest)
        recording=runner.RecordingIndex(index,out);lock=threading.RLock()
        for item in inputs:
            row=(runner.answer_conversation if a.benchmark=='convfinqa' else runner.answer_one)(client,recording,lock,item,config)
            append_jsonl(out/'results.jsonl',row)
        manifest['status']='inference_complete';write_json(out/'manifest.json',manifest)
    print('Saved fresh inference:',out)
    print('This is a new run; score references separately. Do not replace published results with these outputs.')
if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,FileNotFoundError) as exc:
        print(str(exc),file=sys.stderr);sys.exit(2)
