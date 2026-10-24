"""Run only the frozen, label-free three-pass control; no automatic paid retries."""
import argparse
import fcntl
import threading
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
import httpx2
from common import ROOT,assert_frozen
from finrag.core import read_json,write_json,jsonl,append_jsonl,canonical
from finrag.client import OpenRouterClient,read_key,preflight,API,RateLimited
from finrag.unguided import run_item
import finrag.client as api

def account(fetch,key):
    data=fetch(API+'/key',key=key)['data']
    return {**{k:data.get(k) for k in ('usage','limit','limit_remaining')},
            'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}

def make_fetch(http,folder):
    def fetch(url,key=None,payload=None,timeout=180):
        headers={'User-Agent':'finrag-reconstruction/ablation'}
        if key:headers['Authorization']='Bearer '+key
        if payload is not None:headers['Content-Type']='application/json'
        started=time.monotonic()
        try:
            response=http.request('POST' if payload is not None else 'GET',url,headers=headers,
                content=canonical(payload).encode() if payload is not None else None,
                timeout=httpx2.Timeout(timeout,connect=30))
            if response.status_code==429:raise RateLimited('Explicit HTTP429 rejection')
            if response.status_code>=400:raise RuntimeError('HTTP '+str(response.status_code)+'; no automatic paid retry')
            return response.json()
        except httpx2.HTTPError as exc:
            append_jsonl(folder/'transport-errors.jsonl',{'exception_type':type(exc).__name__,
                'paid_request':payload is not None,'elapsed_seconds':time.monotonic()-started,'automatic_retry':False})
            raise RuntimeError('Network failure; responses retained, no automatic paid retry') from None
    return fetch

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,choices=[1,2,3],default=3)
    args=p.parse_args();frozen=assert_frozen();folder=ROOT/'runs/evaluation';folder.mkdir(exist_ok=True)
    inputs=read_json(ROOT/'data/inputs.json');config=frozen['config']
    if len(inputs)!=50:raise ValueError('Fixed sample must contain all fifty inputs')
    with (folder/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        key=read_key(ROOT.parent/'nlp4kgc-reconstruction/.secrets/openrouter.key')
        with httpx2.Client(timeout=180,http2=False,limits=httpx2.Limits(max_keepalive_connections=0)) as http:
            fetch=make_fetch(http,folder);api.fetch_json=fetch
            receipt=account(fetch,key)
            if receipt['limit_remaining'] is not None and receipt['limit_remaining']<=0:
                raise ValueError('Existing API key has no available credit; no paid call made')
            # User lifted the former $20 personal ceiling. Preserve actual key's
            # server limit; this bounded experiment does not edit key settings.
            check_config={**config,'budget_usd':max(float(config['budget_usd']),float(receipt['limit'] or 0))}
            metadata=preflight(check_config,key if receipt['limit'] is not None else None)
            if metadata['endpoint']['provider_name']!=frozen['source_receipt']['provider_name']:
                raise ValueError('Pinned source provider changed')
            if (folder/'manifest.json').exists():
                manifest=read_json(folder/'manifest.json')
                if manifest['frozen']!=frozen:raise ValueError('Frozen experiment changed')
            else:
                write_json(folder/'account-before.json',receipt)
                manifest={'status':'in_progress','frozen':frozen,'preflight':metadata,
                    'workers':args.workers,'created_utc':receipt['utc'],
                    'spending_authorization':'User explicitly allowed exceeding the former $20 overall ceiling; experiment remains bounded to 100 new answer requests.',
                    'existing_server_key_limit_usd':receipt['limit']}
                write_json(folder/'manifest.json',manifest)
            client=OpenRouterClient(check_config,folder,key,metadata)
            seen={r['id'] for r in jsonl(folder/'results.jsonl')};stop=threading.Event();errors=[]
            def work(i,record):
                if stop.is_set():return None
                print('Unguided revision',i,'/50',record['item']['id'],flush=True)
                return run_item(client,record)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                jobs=[pool.submit(work,i,record) for i,record in enumerate(inputs,1) if record['item']['id'] not in seen]
                for job in as_completed(jobs):
                    try:
                        row=job.result()
                        if row:append_jsonl(folder/'results.jsonl',row)
                    except Exception as exc:stop.set();errors.append(type(exc).__name__+': '+str(exc))
            write_json(folder/'account-after.json',account(fetch,key))
            if errors:raise RuntimeError('Run stopped; preserve all saved responses. '+errors[0])
            rows=jsonl(folder/'results.jsonl')
            if len(rows)!=50 or len({r['id'] for r in rows})!=50:raise ValueError('Incomplete or duplicate results')
            manifest['status']='inference_complete';write_json(folder/'manifest.json',manifest)
            print('Complete: all fifty fixed three-pass controls.',flush=True)

if __name__=='__main__':main()
