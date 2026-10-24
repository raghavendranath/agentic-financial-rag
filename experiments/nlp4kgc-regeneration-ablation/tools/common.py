from pathlib import Path
import hashlib
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from finrag.core import read_json,write_json,digest,jsonl
from finrag.experiment import source_hash
FROZEN_TOOLS=('common.py','prepare.py','run_ablation.py','score_ablation.py')

def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def frozen_state():
    return {'source_hash':source_hash(),
        'tools':{p:file_hash(ROOT/'tools'/p) for p in FROZEN_TOOLS},
        'tests':{p.name:file_hash(p) for p in sorted((ROOT/'tests').glob('*.py'))},
        'config':read_json(ROOT/'configs/deepseek.json'),
        'inputs_hash':digest(read_json(ROOT/'data/inputs.json')),
        'protocol_hash':digest(read_json(ROOT/'docs/PROTOCOL.json')),
        'assessment_files':read_json(ROOT/'provenance/ASSESSMENT-HASHES.json'),
        'source_receipt':read_json(ROOT/'provenance/SOURCE.json')}

def assert_frozen():
    expected=read_json(ROOT/'docs/FREEZE.json')
    if expected!=frozen_state():raise ValueError('Frozen code, protocol, assessment or inputs changed')
    return expected
