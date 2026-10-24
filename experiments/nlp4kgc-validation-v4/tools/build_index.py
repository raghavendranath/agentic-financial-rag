"""Checkpointed BGE-M3/FAISS full-report index, with local Metal acceleration."""
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from finrag.core import read_json,write_json,digest
from finrag.dense import DenseIndex

def main():
    p=argparse.ArgumentParser();p.add_argument('--benchmark',choices=['financebench','convfinqa'],required=True)
    p.add_argument('--device',default='mps');p.add_argument('--batch-size',type=int,default=4);a=p.parse_args()
    data=ROOT/'data'/a.benchmark;corpus=read_json(data/'corpus.json')
    if digest(corpus)!=read_json(data/'manifest.json')['corpus_hash']:raise ValueError('Source corpus changed')
    index=DenseIndex(revision='5617a9f61b028005a4858fdac845db406aefb181',device=a.device)
    folder=data/'index';folder.mkdir(exist_ok=True)
    params={'source_hash':digest(corpus),'model':index.model_id,'revision':index.resolved_revision,'chunk_tokens':1500,'overlap_tokens':200,'normalized':True,'index':'IndexFlatIP'}
    cache=ROOT/'.cache'/a.benchmark;cache.mkdir(parents=True,exist_ok=True)
    chunks=[c for d in corpus for c in index.chunk(d['filename'],d['text'],1500,200)]
    stamp={'parameters':params,'chunks_hash':digest(chunks),'device':a.device,'batch_size':a.batch_size}
    if (cache/'manifest.json').exists() and read_json(cache/'manifest.json')!=stamp:raise ValueError('Embedding checkpoint changed')
    write_json(cache/'manifest.json',stamp)
    print('Indexing',len(chunks),'chunks on',a.device,flush=True)
    import numpy as np
    started=time.monotonic();batches=[]
    for start in range(0,len(chunks),a.batch_size):
        path=cache/(str(start)+'.npy');batch=chunks[start:start+a.batch_size]
        if not path.exists():
            vectors=index.encoder.encode([c['text'] for c in batch],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False,batch_size=a.batch_size)
            np.save(path,np.ascontiguousarray(vectors,dtype='float32'))
        vectors=np.load(path,allow_pickle=False)
        if vectors.shape!=(len(batch),1024) or not np.isfinite(vectors).all():raise ValueError('Invalid saved embeddings')
        batches.append(vectors)
        if start%(10*a.batch_size)==0:print('Embedded',min(start+a.batch_size,len(chunks)),'/',len(chunks),'elapsed',round(time.monotonic()-started),'s',flush=True)
    vectors=cache/'vectors.npy';np.save(vectors,np.concatenate(batches))
    index.worker('build',vectors,folder/'index.faiss')
    write_json(folder/'chunks.json',chunks)
    manifest={'parameters':params,'dimension':1024,'chunks':len(chunks),'chunks_hash':digest(chunks),'index_sha256':hashlib.sha256((folder/'index.faiss').read_bytes()).hexdigest(),'embedding_runtime':{'device':a.device,'batch_size':a.batch_size,'seconds':time.monotonic()-started}}
    write_json(folder/'manifest.json',manifest)
    print('Complete:',a.benchmark,len(chunks),'chunks',flush=True)

if __name__=='__main__':main()
