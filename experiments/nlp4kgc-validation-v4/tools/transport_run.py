"""HTTP transport without connection reuse; exact payloads, no paid auto-retries."""
import hashlib
import sys
import time
from pathlib import Path
import httpx2 as httpx
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import finrag.client as api
from finrag.core import canonical,append_jsonl,write_json
import run_benchmark

def main():
    client=httpx.Client(timeout=httpx.Timeout(180,connect=30),http2=False,limits=httpx.Limits(max_keepalive_connections=0),follow_redirects=True)
    def fetch(url,key=None,payload=None,timeout=180):
        headers={'User-Agent':'finrag-reconstruction/0.1'}
        if key:headers['Authorization']='Bearer '+key
        if payload is not None:headers['Content-Type']='application/json'
        start=time.monotonic()
        try:
            response=client.request('POST' if payload is not None else 'GET',url,headers=headers,
                content=canonical(payload).encode() if payload is not None else None,timeout=httpx.Timeout(timeout,connect=30))
            if response.status_code==429:raise api.RateLimited('Explicit HTTP429 rate-limit rejection')
            if response.status_code>=400:raise RuntimeError(f'HTTP {response.status_code}; no automatic paid retry')
            return response.json()
        except httpx.HTTPError as exc:
            append_jsonl(ROOT/'provenance/TRANSPORT-ERRORS.jsonl',{'exception_type':type(exc).__name__,
                'paid_request':payload is not None,'elapsed_seconds':time.monotonic()-start,'automatic_retry':False})
            raise RuntimeError('Network transport failure; saved replies retained and no paid automatic retry') from None
    write_json(ROOT/'provenance/TRANSPORT-AMENDMENT.json',{'reason':'Two inspected short ReadErrors in v4; disable suspected stale connection reuse. Cause is an inference, not proven.',
        'transport':'httpx2 2.16.0 HTTP/1.1 without connection reuse, zero automatic paid retries','request_payload_unchanged':True,
        'source_prompts_model_provider_and_scorer_unchanged':True,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    api.fetch_json=fetch;run_benchmark.fetch_json=fetch
    try:run_benchmark.main()
    finally:client.close()

if __name__=='__main__':main()
