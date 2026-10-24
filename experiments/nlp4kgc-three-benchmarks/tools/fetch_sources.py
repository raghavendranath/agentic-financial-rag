"""Re-download pinned official files, verifying every checksum before use."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--include-pdfs',action='store_true');a=p.parse_args()
    for record in json.loads((ROOT/'provenance/SOURCE-FILES.json').read_text()):
        path=ROOT/record['relative_path']
        if path.suffix=='.pdf' and not a.include_pdfs:continue
        if not path.exists():
            with urllib.request.urlopen(record['url'],timeout=90) as response:content=response.read()
            if hashlib.sha256(content).hexdigest()!=record['sha256']:raise ValueError('Source checksum mismatch: '+record['relative_path'])
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Local source changed: '+record['relative_path'])
        print('Verified',record['relative_path'])

if __name__=='__main__':main()
