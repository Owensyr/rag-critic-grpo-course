"""Prepare, smoke test and train the ORIGINAL step4 CFT corpus with LoRA."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent
SOURCE_HASH='c500875fc40020ea3fe302e8797deb4512a46c92e54ab5150d52e16f9928de72'
MODEL_HASH='29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e'
PINS={'llamafactory':'0.9.3','transformers':'4.51.3','tokenizers':'0.21.1','datasets':'3.6.0','accelerate':'1.7.0','peft':'0.15.2','trl':'0.9.6','numpy':'1.26.4','huggingface-hub':'0.34.4'}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,p)
def utc():return datetime.now(timezone.utc).isoformat()
def tree_hashes(p):return {str(x.relative_to(p)):sha(x) for x in sorted(p.rglob('*')) if x.is_file()}
def package_check():
    for n,h in load(ROOT/'package_manifest.json')['files_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('Package changed: '+n)

def environment():
    import torch
    versions={n:importlib.metadata.version(n) for n in PINS}
    for n,v in PINS.items():
        if versions[n]!=v:raise ValueError(f'{n}: expected {v}, got {versions[n]}')
    if not torch.__version__.startswith('2.8.0') or torch.version.cuda!='12.8':raise ValueError('Expected existing torch 2.8.0+cu128')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=4:raise ValueError('Need four visible GPUs')
    info={'versions':versions,'python':sys.version,'python_executable':sys.executable,'torch':torch.__version__,'torch_path':torch.__file__,'cuda':torch.version.cuda,'gpu_names':[torch.cuda.get_device_name(i) for i in range(4)],'bf16':torch.cuda.is_bf16_supported()}
    if not info['bf16']:raise ValueError('BF16 not supported')
    save(ROOT/'reports/environment.json',info);print('TRAINING_ENVIRONMENT_OK',flush=True)
    return info

def config(model,phase):
    c={'model_name_or_path':str(model),'trust_remote_code':False,'stage':'sft','do_train':True,'do_eval':True,'finetuning_type':'lora','lora_target':'all','lora_rank':16,'lora_alpha':64,'lora_dropout':0.0,'template':'llama3','dataset':'clm_original','eval_dataset':'clm_validation','dataset_dir':str(ROOT/'data'),'tokenized_path':str(ROOT/'prepared'/('smoke' if phase=='smoke' else 'full')),'cutoff_len':6144,'train_on_prompt':False,'packing':False,'mask_history':False,'output_dir':str(ROOT/'artifacts'/phase),'overwrite_output_dir':False,'report_to':'none','logging_steps':1 if phase=='smoke' else 5,'logging_nan_inf_filter':False,'save_strategy':'steps','save_steps':3 if phase=='smoke' else 100,'save_total_limit':2,'save_safetensors':True,'plot_loss':True,'per_device_train_batch_size':1,'per_device_eval_batch_size':1,'gradient_accumulation_steps':8,'learning_rate':1e-5,'num_train_epochs':2.0,'lr_scheduler_type':'cosine','warmup_ratio':0.1,'optim':'adamw_torch','weight_decay':0.0,'bf16':True,'fp16':False,'flash_attn':'sdpa','disable_gradient_checkpointing':False,'gradient_checkpointing':True,'gradient_checkpointing_kwargs':{'use_reentrant':False},'ddp_find_unused_parameters':False,'ddp_timeout':1800,'eval_strategy':'steps','eval_steps':3 if phase=='smoke' else 100,'prediction_loss_only':True,'dataloader_num_workers':0,'seed':42,'data_seed':42,'preprocessing_num_workers':1,'val_size':0.0}
    if phase=='smoke':c.update(max_steps=3)
    return c

def prepare(args):
    package_check();env=environment();source=args.source/'artifacts/full/cft_train_10k.json'
    if sha(source)!=SOURCE_HASH:raise ValueError('Use original step4 CFT file, not repaired file')
    if sha(args.model/'config.json')!=MODEL_HASH:raise ValueError('LLaMA base config differs')
    fingerprint={'source_sha256':SOURCE_HASH,'model':str(args.model),'model_config_sha256':MODEL_HASH,'package_sha256':sha(ROOT/'package_manifest.json'),'versions':env['versions'],'torch':env['torch']}
    manifest=ROOT/'reports/prepared.json'
    if manifest.exists():
        old=load(manifest)
        if old['fingerprint']!=fingerprint or old['cache_sha256']!=tree_hashes(ROOT/'prepared'):raise ValueError('Prepared data differs')
        print('PREPARED_DATA_REUSED',flush=True);return
    if (ROOT/'prepared').exists():raise RuntimeError('Partial prepared directory exists; preserve it and return logs for review')
    from datasets import Dataset,DatasetDict
    from transformers import AutoTokenizer
    from llamafactory.hparams import DataArguments
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.data.parser import DatasetAttr
    from llamafactory.data.converter import AlpacaDatasetConverter
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    entries=load(source)
    if len(entries)!=10000:raise ValueError('Expected 10000 entries')
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    da=DataArguments(template='llama3',cutoff_len=6144,train_on_prompt=False,packing=False)
    template=get_template_and_fix_tokenizer(tok,da)
    converter=AlpacaDatasetConverter(DatasetAttr(load_from='file',dataset_name='original'),da)
    processor=SupervisedDatasetProcessor(template,tok,None,da)
    lengths=[];targets=[];preview=[]
    def encoded():
        for i,entry in enumerate(entries):
            a=converter(entry)
            src,tgt=template.encode_oneturn(tok,a['_prompt']+a['_response'],a['_system'],a['_tools'])
            ids,labels=processor._encode_data_example(a['_prompt'],a['_response'],a['_system'],a['_tools'],[],[],[])
            if ids!=src+tgt or labels!=[-100]*len(src)+tgt:raise ValueError(f'Truncation or mask mismatch at {i}')
            if not tgt or len(ids)>6144:raise ValueError(f'Invalid training length at {i}')
            lengths.append(len(ids));targets.append(len(tgt))
            if i<2:preview.append({'index':i,'tokens':len(ids),'target_tokens':len(tgt),'encoded_input':tok.decode(ids),'supervised_output':tok.decode(tgt)})
            yield {'input_ids':ids,'attention_mask':[1]*len(ids),'labels':labels,'images':None,'videos':None,'audios':None,'source_index':i,'length':len(ids)}
            if (i+1)%500==0:print('TOKENIZE',i+1,10000,flush=True)
    ds=Dataset.from_generator(encoded,cache_dir=str(ROOT/'tokenization_work'))
    # A fresh work cache is required so the validation side effects above cannot be skipped.
    if len(lengths)!=10000:raise ValueError('Tokenization cache reused unexpectedly; return logs')
    split=ds.train_test_split(test_size=.1,seed=42,shuffle=True)
    train_indices=list(split['train']['source_index']);val_indices=list(split['test']['source_index'])
    if len(train_indices)!=9000 or len(val_indices)!=1000 or set(train_indices)&set(val_indices):raise ValueError('Split mismatch')
    save(ROOT/'reports/split_indices.json',{'train':train_indices,'validation':val_indices,'seed':42})
    longest=split['train'].sort('length',reverse=True).select(range(64))
    full=DatasetDict({'train':split['train'].remove_columns(['source_index','length']),'validation':split['test'].remove_columns(['source_index','length'])})
    smoke=DatasetDict({'train':longest.remove_columns(['source_index','length']),'validation':split['test'].select(range(8)).remove_columns(['source_index','length'])})
    full.save_to_disk(str(ROOT/'prepared/full'));smoke.save_to_disk(str(ROOT/'prepared/smoke'))
    save(ROOT/'reports/tokenization_preview.json',preview)
    save(ROOT/'data/dataset_info.json',{'clm_original':{'file_name':str(source)},'clm_validation':{'file_name':str(source)}})
    report={'fingerprint':fingerprint,'rows':10000,'train_rows':9000,'validation_rows':1000,'val_size':.1,'split_seed':42,'max_exact_training_tokens':max(lengths),'min_target_tokens':min(targets),'truncated_during_training_encoding':0,'source_generation_truncations_retained':256,'target_generation_truncations_retained':137,'smoke_rows':64,'smoke_selection':'longest 64 within training split','mapping':'Author default Alpaca: instruction + newline + input as USER; output as ASSISTANT; no custom SYSTEM column','cache_sha256':tree_hashes(ROOT/'prepared')}
    save(manifest,report);print('CLM_TRAINING_DATA_READY train=9000 validation=1000',flush=True)

def validate_phase(phase):
    out=ROOT/'artifacts'/phase
    state=load(out/'trainer_state.json');metrics=load(out/'train_results.json')
    if state['global_step']<1:raise ValueError('No optimizer steps')
    if phase=='smoke' and state['global_step']!=3:raise ValueError('Smoke steps differ')
    if phase=='full' and state.get('epoch',0)<1.99:raise ValueError('Training did not finish two epochs')
    for row in state['log_history']:
        for k,v in row.items():
            if isinstance(v,(int,float)) and not math.isfinite(v):raise ValueError('Nonfinite training metric '+k)
    if not math.isfinite(metrics['train_loss']):raise ValueError('Invalid loss')
    adapter=load(out/'adapter_config.json')
    if adapter['r']!=16 or adapter['lora_alpha']!=64:raise ValueError('Adapter parameters differ')
    from safetensors import safe_open
    import torch
    weightfile=out/'adapter_model.safetensors'
    nonzero_b=False;numel=0
    with safe_open(weightfile,framework='pt',device='cpu') as f:
        for k in f.keys():
            value=f.get_tensor(k);numel+=value.numel()
            if not torch.isfinite(value).all():raise ValueError('Nonfinite adapter tensor')
            if 'lora_B' in k and torch.count_nonzero(value).item()>0:nonzero_b=True
    if not nonzero_b:raise ValueError('No learned nonzero LoRA B tensors')
    gpus=[load(out/f'gpu_rank{i}.json') for i in range(4)]
    report={'status':'complete','phase':phase,'global_step':state['global_step'],'epoch':state['epoch'],'train_loss':metrics['train_loss'],'adapter_sha256':sha(weightfile),'adapter_numel':numel,'gpu_peaks_gib':[r['peak_allocated_gib'] for r in gpus],'adapter_lora_b_updated':True}
    save(ROOT/'reports'/f'{phase}_validation.json',report);return report

def launch(phase,model,resume):
    out=ROOT/'artifacts'/phase;done=ROOT/'reports'/f'{phase}_validation.json'
    if done.exists():
        validate_phase(phase);print('SKIP_COMPLETED',phase,flush=True);return
    cfg=config(model,phase)
    if out.exists() and any(out.iterdir()):
        checkpoints=[]
        for cp in out.glob('checkpoint-*'):
            if (cp/'trainer_state.json').exists() and (cp/'optimizer.pt').exists() and (cp/'adapter_model.safetensors').exists():checkpoints.append(cp)
        if not resume or not checkpoints:raise RuntimeError(f'{phase} output exists; use --resume if a complete checkpoint is available')
        cfg['resume_from_checkpoint']=str(max(checkpoints,key=lambda p:int(p.name.split('-')[-1])))
    path=ROOT/'configs'/f'{phase}.json';save(path,cfg)
    logfile=ROOT/'logs'/f'{phase}.log';logfile.parent.mkdir(exist_ok=True)
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=4',str(ROOT/'worker.py'),str(path)]
    print('START_TRAINING',phase,'LOG',logfile,flush=True)
    with logfile.open('a',encoding='utf-8') as f:
        subprocess.run(cmd,cwd=ROOT,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,check=True)
    validate_phase(phase)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['environment','prepare','all','status'])
    p.add_argument('--source',type=Path,default=Path.home()/'RAG/clm_training_step4_data')
    p.add_argument('--model',type=Path,default=Path.home()/'RAG/models_verified/meta-llama/Llama-3.1-8B-Instruct')
    p.add_argument('--resume',action='store_true');a=p.parse_args();a.source=a.source.expanduser().resolve();a.model=a.model.expanduser().resolve()
    if a.command=='status':
        result={'pipeline':load(ROOT/'reports/pipeline.json') if (ROOT/'reports/pipeline.json').exists() else None}
        for phase in ('smoke','full'):
            for name in ('trainer_log.jsonl','trainer_state.json'):
                path=ROOT/'artifacts'/phase/name
                if path.exists():
                    if name.endswith('jsonl'):
                        lines=path.read_text().splitlines();result[phase+'_last_log']=json.loads(lines[-1]) if lines else None
                    else:
                        s=load(path);result[phase+'_saved_state']={k:s[k] for k in ('global_step','epoch','max_steps')}
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    os.environ.update(CUDA_VISIBLE_DEVICES='0,1,2,3',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',WANDB_DISABLED='true',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    os.environ.setdefault('OMP_NUM_THREADS','4')
    package_check()
    if a.command=='environment':environment();return
    (ROOT/'reports').mkdir(exist_ok=True)
    import fcntl
    with (ROOT/'reports/pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state={'status':'running','stage':'prepare','started_utc':utc(),'pid':os.getpid(),'source_sha256':SOURCE_HASH,'source':'original_step4_unfiltered'};mp=ROOT/'reports/pipeline.json';save(mp,state)
        try:
            prepare(a)
            if a.command=='prepare':state.update(status='prepared');return
            for phase in ('smoke','full'):
                state['stage']=phase;save(mp,state);launch(phase,a.model,a.resume)
            state.update(status='complete',stage='full_complete');print('CLM_LORA_TRAINING_COMPLETE epochs=2',flush=True)
        except BaseException as e:state.update(status='failed',error=repr(e));raise
        finally:state['updated_utc']=utc();save(mp,state)

if __name__=='__main__':main()
