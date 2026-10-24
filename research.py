#!/usr/bin/env python3
"""Entry point for reproducibility checks; offline actions never need a key."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EXPERIMENTS=ROOT/'experiments'
PROJECTS=('nlp4kgc-validation-v3','nlp4kgc-three-benchmarks','nlp4kgc-validation-v4','nlp4kgc-regeneration-ablation')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def check():
    manifest=read(ROOT/'SNAPSHOT-MANIFEST.json')
    for rel,expected in manifest.items():
        p=ROOT/rel
        if p.is_symlink() or not p.is_file() or p.stat().st_size!=expected['bytes'] or sha(p)!=expected['sha256']:
            raise ValueError('Missing or changed frozen scientific file: '+rel)
    print(f'PASS: {len(manifest)} snapshot files match SHA-256 checksums')

def execute(project,args,offline=True):
    folder=EXPERIMENTS/project;env=os.environ.copy()
    env['PYTHONPATH']=str(folder/'src');env['PYTHONDONTWRITEBYTECODE']='1'
    if offline:
        env.pop('OPENROUTER_API_KEY',None)
        env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    subprocess.run([sys.executable,*args],cwd=folder,env=env,check=True)

def tests():
    for p in PROJECTS:
        print('\nTests:',p,flush=True)
        execute(p,['-m','unittest','discover','-s','tests','-q'])

def verify():
    check()
    jobs=[
      (PROJECTS[0],['tools/verify_run.py','--run','runs/holdout50']),
      (PROJECTS[0],['tools/score_official.py','--run','runs/holdout50']),
      (PROJECTS[1],['tools/score_benchmark.py','--run','runs/convfinqa-evaluation']),
      (PROJECTS[1],['tools/score_qwen.py','--benchmark','finqa']),
      (PROJECTS[1],['tools/score_qwen.py','--benchmark','convfinqa']),
      (PROJECTS[2],['tools/score_benchmark.py','--run','runs/financebench-evaluation']),
      (PROJECTS[2],['tools/score_qwen.py','--benchmark','financebench']),
      (PROJECTS[3],['tools/score_ablation.py']),
      ('t5-small',['evaluate.py','score'])]
    for p,args in jobs:
        print('\nOffline replay/scoring:',p,' '.join(args),flush=True)
        execute(p,args)
    print('\nPASS: all selected requests/trajectories replayed and saved predictions rescored. No human labels created.')

def results():
    print((ROOT/'results/RESULTS.txt').read_text())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','tests','verify','results'])
    a=p.parse_args();globals()[a.command]()
if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError,subprocess.CalledProcessError) as exc:
        print(str(exc),file=sys.stderr);sys.exit(1)
