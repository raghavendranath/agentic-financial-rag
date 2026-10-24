#!/usr/bin/env python3
"""Check candidate Git files for release hygiene and published result consistency."""
import hashlib,json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BLOCKED={'.secrets','.cache','.venv','__pycache__','checkpoint'}
PRIVATE={'PAPER-BOUNDARIES.txt','REVIEW-COVERAGE.txt','cl-manuscript.txt','workshop-paper.txt','HUMAN-EVALUATION-LIMITATION.txt','SCOPE.md'}
PATTERNS=[re.compile(rb'sk-or-v1-[A-Za-z0-9]{20,}'),re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----')]

def validate_files(paths):
    total=0
    for p in paths:
        rel=p.relative_to(ROOT)
        if set(rel.parts)&BLOCKED or p.name in PRIVATE or p.suffix in ('.key','.pem','.pyc'):
            raise ValueError('Excluded release path: '+str(rel))
        if p.is_symlink():raise ValueError('Release files must not depend on local symlinks: '+str(rel))
        if p.stat().st_size>=50_000_000:raise ValueError('Move large file to release assets: '+str(rel))
        data=p.read_bytes()
        if any(x.search(data) for x in PATTERNS):raise ValueError('Potential secret in '+str(rel))
        total+=len(data)
    return total

def check_results():
    def load(n):return json.loads((ROOT/'results'/n).read_text())
    fb=load('financebench.json')['arms'];cv=load('convfinqa.json')['arms'];f=load('finqa.json')['arms'];ab=load('financebench-ablation.json')['arms']
    if [fb[a]['correct'] for a in ['single_shot','fixed_retrieval','agentic']]!=[6,9,10]:raise ValueError('FinanceBench summary mismatch')
    if [cv[a]['correct'] for a in ['single_shot','agentic']]!=[132,134]:raise ValueError('ConvFinQA summary mismatch')
    if [f[a]['numerical_correct'] for a in ['single_shot','agentic']]!=[28,35]:raise ValueError('FinQA summary mismatch')
    if ab['unguided_fixed_three_pass']['literal_matches']!=9:raise ValueError('Ablation summary mismatch')
    if load('t5-small.json')['numerical_correct']!=12:raise ValueError('T5 summary mismatch')
    for n in ['financebench.json','convfinqa.json','finqa.json']:
        if not (ROOT/'results'/n).exists():raise ValueError('Missing result file')

def main():
    result=subprocess.run(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT,capture_output=True,check=True)
    paths=[ROOT/n for n in result.stdout.decode().split('\0') if n]
    total=validate_files(paths);check_results()
    print(f'PASS: {len(paths)} candidate Git files; {total/1e6:.2f} MB; no excluded paths or recognized secrets; result counts checked')
if __name__=='__main__':
    try:main()
    except (ValueError,subprocess.CalledProcessError) as e:print(e,file=sys.stderr);sys.exit(1)
