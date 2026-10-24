"""Prepare fixed public inputs and separate references for two added benchmarks."""
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from finrag.core import public_item,read_json,write_json,digest
from finrag.data import document

def save(name,inputs,labels,corpus,metadata):
    folder=ROOT/'data'/name
    for split in inputs:
        write_json(folder/(split+'.inputs.json'),inputs[split])
        write_json(folder/(split+'.labels.json'),labels[split])
    write_json(folder/'corpus.json',corpus)
    manifest={'dataset':name,**metadata,'corpus_hash':digest(corpus),'splits':{
        s:{'n':len(v),'ids':[x['id'] for x in v],'inputs_hash':digest(v),'labels_hash':digest(labels[s])} for s,v in inputs.items()}}
    write_json(folder/'manifest.json',manifest)
    return manifest

def main():
    protocol=read_json(ROOT/'docs/PROTOCOL.json')
    rows={x['financebench_id']:x for x in [json.loads(l) for l in (ROOT/'data/raw/financebench/financebench_open_source.jsonl').read_text().splitlines()]}
    inputs,labels={},{}
    for split in ('development','evaluation'):
        selected=[rows[i] for i in protocol['financebench'][split]]
        inputs[split]=[{'id':x['financebench_id'],'filename':x['doc_name']+'/report','question':x['question'],'benchmark':'financebench'} for x in selected]
        labels[split]={x['financebench_id']:{'answer':x['answer'],'justification':x['justification'],'evidence':x['evidence'],'question_type':x['question_type']} for x in selected}
    import pymupdf
    corpus=[];pages=0;empty=[]
    extracted=(ROOT/'data/financebench/manifest.json').exists()
    for doc in ([] if extracted else protocol['financebench']['documents']):
        pdf=pymupdf.open(ROOT/'data/raw/financebench-pdfs'/(doc+'.pdf'))
        texts=[]
        for n,page in enumerate(pdf):
            text=page.get_text('text',sort=True)
            texts.append('[page_'+str(n)+']\n'+text)
            pages+=1
            if not text.strip():empty.append([doc,n])
        corpus.append({'filename':doc+'/report','text':'\n\n'.join(texts)})
        print('Extracted',doc,len(pdf),'pages',flush=True)
    if not extracted:
        save('financebench',inputs,labels,corpus,{'source':read_json(ROOT/'data/raw/financebench-source.json'),'retrieval':'BGE-M3/FAISS over full selected report PDFs; known-document filter','pdf_pages':pages,'pdf_extractor':'PyMuPDF '+pymupdf.__version__,'empty_pages':empty,'source_evidence_used_in_inference':False})
    previous=set()
    for folder in (ROOT.parent/'nlp4kgc-reconstruction/data/finqa50-v1',ROOT.parent/'nlp4kgc-validation-v2/data/fresh100',ROOT.parent/'nlp4kgc-validation-v3/data/fresh50'):
        for split in ('pilot','evaluation'):
            previous.update(x['filename'].rsplit('/',1)[0] for x in read_json(folder/(split+'.inputs.json')))
    def ordered(split):
        return sorted(read_json(ROOT/'data/raw/convfinqa'/(split+'.json')),key=lambda x:hashlib.sha256(('three-benchmarks-1821|'+x['id']).encode()).hexdigest())
    chosen={};seen=set(previous)
    for split,source,count in [('evaluation','dev',50),('development','train',5)]:
        chosen[split]=[]
        for x in ordered(source):
            filing=x['filename'].rsplit('/',1)[0]
            if filing in seen:continue
            # Structural eligibility only; do not drop mistakes, unknowns, or difficult programs.
            qs=x['annotation']['dialogue_break']
            if not qs:continue
            chosen[split].append(x);seen.add(filing)
            if len(chosen[split])==count:break
        if len(chosen[split])!=count:raise ValueError('Not enough disjoint ConvFinQA filings')
    inputs,labels={},{}
    for split,selected in chosen.items():
        inputs[split]=[{'id':x['id'],'filename':x['filename'],'questions':x['annotation']['dialogue_break'],'benchmark':'convfinqa'} for x in selected]
        labels[split]={x['id']:{'answers':x['annotation']['exe_ans_list'],'programs':x['annotation']['turn_program']} for x in selected}
    filenames={x['filename'].rsplit('/',1)[0] for v in chosen.values() for x in v}
    bypage={}
    for source in ('train','dev'):
        for raw in read_json(ROOT/'data/raw/convfinqa'/(source+'.json')):
            if raw['filename'].rsplit('/',1)[0] in filenames:
                page={k:raw[k] for k in ('id','filename','pre_text','post_text','table','table_ori') if k in raw}
                page['qa']={'question':''}
                doc=document(page)
                if doc['filename'] in bypage and bypage[doc['filename']]!=doc:raise ValueError('Inconsistent public source page')
                bypage[doc['filename']]=doc
    manifest=save('convfinqa',inputs,labels,[bypage[k] for k in sorted(bypage)],{'source':read_json(ROOT/'data/raw/convfinqa-source.json'),'selection':'SHA256 three-benchmarks-1821|id; one conversation per filing; exclude prior observed FinQA filings; development from train and evaluation from dev','retrieval':'Public processed report-page corpus; known-filing filter','history':'Own previous model-generated answers per arm; never gold history','turn_counts':{s:sum(len(x['questions']) for x in v) for s,v in inputs.items()},'previous_finqa_filings_excluded':sorted(previous)})
    print('Prepared FinanceBench',pages,'pages; ConvFinQA turns',manifest['turn_counts'])

if __name__=='__main__':main()
