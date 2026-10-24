#!/usr/bin/env python3
"""Install a checksum-verified scientific artifact without unsafe archive paths."""
import argparse,hashlib,json,shutil,stat,sys,tempfile,zipfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def member_path(name,root=ROOT):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or len(p.parts)<4 or p.parts[0]!='experiments' or p.parts[2] not in ('data','runs','assessment'):
        raise ValueError('Unexpected or unsafe artifact member: '+name)
    target=root.joinpath(*p.parts)
    if root.resolve() not in target.resolve().parents:raise ValueError('Escaping artifact path')
    return target

def install(path):
    expected=json.loads((ROOT/'ARTIFACTS.json').read_text())['benchmark-artifacts.zip']
    if path.stat().st_size!=expected['bytes'] or digest(path)!=expected['sha256']:raise ValueError('Artifact checksum mismatch')
    snapshot=expected.get('files') or json.loads((ROOT/'SNAPSHOT-MANIFEST.json').read_text())
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate artifact paths')
        for info in z.infolist():
            target=member_path(info.filename)
            if stat.S_ISLNK(info.external_attr>>16) or info.filename not in snapshot:raise ValueError('Unapproved member')
            if info.file_size!=snapshot[info.filename]['bytes']:raise ValueError('Member size mismatch')
            if target.exists() and digest(target)!=snapshot[info.filename]['sha256']:raise ValueError('Would overwrite a changed scientific file: '+info.filename)
        if sum(i.file_size for i in z.infolist())>1024**3:raise ValueError('Unexpected expansion size')
        for info in z.infolist():
            target=member_path(info.filename)
            if target.exists():continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as src,tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as dst:
                temp=Path(dst.name);shutil.copyfileobj(src,dst)
            try:
                if digest(temp)!=snapshot[info.filename]['sha256']:raise ValueError('Member checksum mismatch')
                temp.replace(target)
            finally:temp.unlink(missing_ok=True)
    print('Installed verified artifacts. Run: python research.py check')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path)
    try:install(p.parse_args().archive)
    except (ValueError,FileNotFoundError,zipfile.BadZipFile) as e:print(e,file=sys.stderr);sys.exit(1)
