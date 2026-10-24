"""Run the Qwen baseline with the same pooled transport amendment."""
import hashlib
from pathlib import Path
import transport_run
import parallel_qwen
from finrag.core import write_json

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    write_json(root/'provenance/QWEN-TRANSPORT-WRAPPER.json',{'transport_only':True,
        'model_request_payload_unchanged':True,'automatic_paid_retries':0,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    transport_run.run_benchmark=parallel_qwen
    transport_run.main()
