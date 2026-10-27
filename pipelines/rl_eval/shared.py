"""Pinned provenance and lossless submission helpers for the reviewed GRPO run."""
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
EXPECTED={
    'baseline':'2385069c6061e2cd78631a8f5f59c3936355af60ed2ba3323d86a88a492662c3',
    'test':'132adfbd00b394a293b5eb6fc1f8195471fbfcf0e9a796a2bdbcb905cb860bcb',
    'sample':'7415d29a4ad892ea75a6c56a2bca3dab3740d03ff4e4cdd166e6e690ff90d751'}

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def fingerprint(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def paths():
    base=Path.home()/'RAG';course=base/'author_alignrag_course'
    return dict(pilot=base/'clm_rl_grpo_pilot_v3',
        sft=base/'clm_step6_2wiki/models/Llama-3.1-8B-Instruct-CLM-original10k',
        qwen=base/'models_verified/Qwen/Qwen2.5-7B-Instruct',
        adapter=base/'clm_rl_grpo_pilot_v3/artifacts/checkpoints/step016',
        inference_python=Path.home()/'miniconda3/envs/rag-author/bin/python',
        baseline=course/'course/artifacts/full/qwen/2WikiMultiHopQA/result_rationale_cot.json',
        test=course/'course_data/2WikiMultiHopQA/test.json',
        sample=course/'course_data/2WikiMultiHopQA/sample_submission.csv')

def verify_files(folder,expected):
    folder=Path(folder).resolve()
    for name,digest in expected.items():
        p=(folder/name).resolve()
        if not p.is_relative_to(folder):raise ValueError('Manifest path escapes folder')
        if sha(p)!=digest:raise ValueError('File hash mismatch: '+str(p))

def verify_package():verify_files(ROOT,load(ROOT/'source_manifest.json')['files_sha256'])

def rows_valid(p,selected):
    rows=load(p)
    if len(rows)!=len(selected):raise ValueError('Wrong number of outputs: '+str(p))
    for r,s in zip(rows,selected):
        if str(r['id'])!=str(s['id']) or r['question']!=s['question']:raise ValueError('ID/question mismatch')
        if not isinstance(r.get('rationale'),str) or not r['rationale'].strip():raise ValueError('Empty rationale')
    return rows

def sample_ids(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['id','answer']:raise ValueError('Unexpected CSV columns')
        ids=[r['id'] for r in reader]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate sample IDs')
    return ids

def export_csv(rows,sample,output,preview=False):
    ids=sample_ids(sample);wanted=ids[:len(rows)] if preview else ids
    if [str(r['id']) for r in rows]!=wanted:raise ValueError('CSV IDs differ from template')
    if any(not isinstance(r.get('rationale'),str) or not r['rationale'].strip() for r in rows):raise ValueError('Empty answer')
    output=Path(output);temp=output.with_name(output.name+'.tmp')
    with temp.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['id','answer']);w.writerows((str(r['id']),r['rationale']) for r in rows)
    with temp.open(encoding='utf-8',newline='') as f:reread=list(csv.DictReader(f))
    if reread!=[dict(id=str(r['id']),answer=r['rationale']) for r in rows]:raise ValueError('CSV round-trip failed')
    temp.replace(output)

def verify_inputs(p):
    verify_package();anchor=load(ROOT/'reference/reviewed_training.json')
    print('Verifying reviewed GRPO step016 and training provenance...',flush=True)
    verify_files(p['pilot'],anchor['training_files_sha256'])
    verify_files(p['pilot'],anchor['checkpoint_files_sha256'])
    spec=load(p['pilot']/'artifacts/spec.json');status=load(p['pilot']/'artifacts/status.json')
    if fingerprint(spec['spec'])!=anchor['training_run_fingerprint'] or status['status']!='complete':
        raise ValueError('Training run differs from reviewed complete run')
    cfg=load(p['adapter']/'adapter_config.json')
    if cfg['r']!=8 or cfg['lora_alpha']!=16 or cfg['bias']!='none' or cfg.get('use_dora') or cfg.get('use_rslora'):
        raise ValueError('Unsupported or wrong adapter configuration')
    if Path(cfg['base_model_name_or_path']).resolve()!=p['sft'].resolve():raise ValueError('Adapter base must be trained SFT CLM')
    if load(p['sft']/'merge_manifest.json')!=anchor['sft']:raise ValueError('Wrong SFT manifest')
    print('Verifying SFT CLM and Qwen model files (may take a few minutes)...',flush=True)
    verify_files(p['sft'],anchor['sft']['files_sha256']);verify_files(p['qwen'],anchor['qwen'])
    for name,h in EXPECTED.items():
        if sha(p[name])!=h:raise ValueError('Original experiment input changed: '+name)
    test=load(p['test'])
    if len(test)!=12576 or any(r.get('answers') is not None for r in test):raise ValueError('Expected 12576 unlabeled test rows')
    if sample_ids(p['sample'])!=[str(r['id']) for r in test]:raise ValueError('Template/test order differs')
    rows_valid(p['baseline'],test)
    return anchor,test

def check_environment(p):
    import sys,importlib.metadata as md
    # Check the prefix, never dereference a venv bin/python symlink.
    expected=p['inference_python'].parent.parent.resolve()
    if Path(sys.prefix).resolve()!=expected:raise RuntimeError('Use ~/miniconda3/envs/rag-author/bin/python run.py ...')
    versions={n:md.version(n) for n in ['torch','vllm','transformers','safetensors']}
    if versions['vllm']!='0.10.2':raise RuntimeError('This package was prepared for installed vLLM 0.10.2')
    import torch
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    from safetensors import safe_open
    if not torch.cuda.is_available() or torch.cuda.device_count()!=4:raise RuntimeError('Expose four GPUs: CUDA_VISIBLE_DEVICES=0,1,2,3')
    tensors=0;nonzero_b=0
    with safe_open(p['adapter']/'adapter_model.safetensors',framework='pt',device='cpu') as f:
        for key in f.keys():
            x=f.get_tensor(key);tensors+=1
            if not torch.isfinite(x).all():raise ValueError('Nonfinite adapter tensor '+key)
            if '.lora_B.' in key:nonzero_b+=int(torch.count_nonzero(x))
    if tensors!=448 or nonzero_b==0:raise ValueError('Wrong or zero RL adapter')
    return dict(versions=versions,python_executable=sys.executable,python_prefix=sys.prefix,
                gpu_names=[torch.cuda.get_device_name(i) for i in range(4)],adapter_tensors=tensors,nonzero_lora_b=nonzero_b)
