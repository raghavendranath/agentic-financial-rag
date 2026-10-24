"""Transport-only continuation using system curl; no key in argv or files."""
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path
from common import ROOT
from finrag.core import canonical,append_jsonl,write_json
from finrag.client import RateLimited
import run_ablation


def make_fetch(unused_http,folder):
    def fetch(url,key=None,payload=None,timeout=180):
        # Curl configuration enters through stdin; credentials never enter argv,
        # body files, response logs, or error messages printed to the user.
        lines=['silent','show-error','http1.1','retry = 0','connect-timeout = 30',
               'max-time = '+str(int(timeout)),'url = '+json.dumps(url),
               'header = '+json.dumps('User-Agent: finrag-reconstruction/ablation'),
               'write-out = "\\n%{http_code}"']
        if key:lines.append('header = '+json.dumps('Authorization: Bearer '+key))
        body_path=None
        if payload is not None:
            lines.append('header = "Content-Type: application/json"')
            with tempfile.NamedTemporaryFile(prefix='public-request-',suffix='.json',dir=folder,delete=False) as body:
                body.write(canonical(payload).encode());body_path=Path(body.name)
            lines.append('data-binary = '+json.dumps('@'+str(body_path)))
        started=time.monotonic()
        try:
            result=subprocess.run(['curl','--disable','--config','-'],input='\n'.join(lines)+'\n',
                capture_output=True,text=True,timeout=timeout+10)
            if result.returncode:
                append_jsonl(folder/'curl-transport-errors.jsonl',{'curl_exit_code':result.returncode,
                    'paid_request':payload is not None,'elapsed_seconds':time.monotonic()-started,'automatic_retry':False})
                raise RuntimeError('Curl network failure; uncertain request preserved, no automatic paid retry')
            content,code=result.stdout.rsplit('\n',1)
            if code=='429':raise RateLimited('Explicit HTTP429 rejection')
            if code!='200':raise RuntimeError('HTTP '+code+'; no automatic paid retry')
            return json.loads(content)
        except subprocess.TimeoutExpired:
            append_jsonl(folder/'curl-transport-errors.jsonl',{'timeout':True,'paid_request':payload is not None,'automatic_retry':False})
            raise RuntimeError('Curl deadline reached; no automatic paid retry') from None
        finally:
            if body_path:body_path.unlink(missing_ok=True)
    return fetch

if __name__=='__main__':
    write_json(ROOT/'provenance/TRANSPORT-CONTINUATION.json',{
        'reason':'Two inspected short Python HTTP read failures; use system curl for transport only.',
        'model_request_body_unchanged':True,'automatic_ambiguous_paid_retries':0,
        'credential_in_process_argv_or_files':False,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    run_ablation.make_fetch=make_fetch
    run_ablation.main()
