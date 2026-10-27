"""Four-3090 deployment of reviewed SFT CLM + step016 RL LoRA, full-answer export."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
from shared import (ROOT,EXPECTED,load,save,sha,fingerprint,paths,verify_inputs,
                    check_environment,rows_valid,export_csv,verify_files,verify_package)

ACTIVE=None

def report(mode,variant):
    out=ROOT/f'artifacts/{variant}_{mode}'
    target=ROOT/f'clm_rl_step7_v3_{variant}_{mode}_reports.tar.gz'
    files=[]
    for d in [out,ROOT/'logs',ROOT/'reports',ROOT/'reference']:
        if d.exists():files.extend(p for p in d.rglob('*') if p.is_file() and p.suffix in {'.json','.jsonl','.csv','.log','.txt'})
    files.extend(p for p in ROOT.iterdir() if p.is_file() and p.suffix in {'.py','.json','.txt'})
    for d in [ROOT/'eval',ROOT/'configs']:
        files.extend(p for p in d.rglob('*') if p.is_file() and p.suffix in {'.py','.yaml'})
    tmp=target.with_suffix('.tmp')
    with tarfile.open(tmp,'w:gz') as t:
        for p in sorted(set(files)):t.add(p,arcname=p.relative_to(ROOT).as_posix())
    tmp.replace(target);print('REPORT_READY',target,flush=True)

def stop(signum,frame):
    if ACTIVE is not None and ACTIVE.poll() is None:os.killpg(ACTIVE.pid,signal.SIGTERM)
    raise KeyboardInterrupt('Evaluation interrupted')

def invoke(command,log):
    global ACTIVE
    env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    with log.open('a',encoding='utf-8') as f:
        f.write(f'\nATTEMPT {time.time()}\n');f.flush()
        ACTIVE=subprocess.Popen([str(x) for x in command],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=ACTIVE.wait()
        finally:
            if ACTIVE.poll() is None:
                os.killpg(ACTIVE.pid,signal.SIGTERM)
                try:ACTIVE.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(ACTIVE.pid,signal.SIGKILL);ACTIVE.wait()
            ACTIVE=None
    if rc:raise RuntimeError(f'Command exited {rc}; inspect {log}')

def build_command(p,out,phase,mode,variant,probe_only=False):
    cmd=[p['inference_python'],'-u','eval/inference_vllm.py','--data_path',p['test'],
         '--dataset_name','2WikiMultiHopQA','--model_name_or_path',p['qwen'],'--critique_model',p['sft'],
         '--config_file','configs/eval.yaml','--type','cot','--iters','1','--n_docs','5','--seed','42',
         '--max_tokens','5120','--max_model_len','32768','--gpu_memory_utilization','0.85',
         '--output_dir',p['baseline'].parent.parent,'--critique_model_path',out,'--phase',str(phase)]
    if mode=='smoke' or probe_only:cmd+=['--max_instances','5']
    if variant=='rl' and phase==2:cmd+=['--rl_adapter',p['adapter']]
    if probe_only:cmd+=['--probe_only']
    return [str(x) for x in cmd]

def status(mode,variant):
    out=ROOT/f'artifacts/{variant}_{mode}';mp=out/'run_manifest.json'
    if not mp.exists():print('NOT_STARTED',out);return
    s=load(mp);alive=False
    try:
        if not isinstance(s.get('pid'),int) or s['pid']<=0:raise ValueError('No PID')
        os.kill(s['pid'],0);alive=True
    except (OSError,TypeError,ValueError):pass
    info={k:s.get(k) for k in ['status','stage','pid','rows','completed_phases','output','error','started','finished']}
    info['process_alive']=alive
    if s.get('stage_log'):
        log=ROOT/s['stage_log']
        if log.exists():info['current_log_tail']=log.read_text(encoding='utf-8',errors='replace').splitlines()[-8:]
    print(__import__('json').dumps(info,ensure_ascii=False,indent=2))

def evaluate(mode,variant):
    import fcntl
    p=paths();out=ROOT/f'artifacts/{variant}_{mode}'
    (ROOT/'artifacts').mkdir(exist_ok=True)
    with (ROOT/'artifacts/evaluation.lock').open('w') as lock:
        # Prevent simultaneous phase runs across smoke/full/SFT/RL on the same four GPUs.
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out.mkdir(exist_ok=True);mp=out/'run_manifest.json'
        previous=load(mp) if mp.exists() else None
        can_update=previous is None
        s=dict(previous) if previous else dict(status='checking',started=time.time(),completed_phases=[],files_sha256={})
        s.update(pid=os.getpid(),stage='preflight')
        try:
            anchor,test=verify_inputs(p);env=check_environment(p)
            identity=dict(source_sha256=sha(ROOT/'source_manifest.json'),training=anchor['training_run_fingerprint'],
                          inputs_sha256=EXPECTED,variant=variant,environment=env,
                          model_loading='SFT plus separate RL LoRA' if variant=='rl' else 'SFT only, matched engine settings',
                          paths={k:str(v) for k,v in p.items()},engine=dict(tensor_parallel_size=4,max_num_seqs=32,enforce_eager=True,
                          enable_prefix_caching=False,lora_dtype='bfloat16',max_model_len=32768,max_tokens=5120))
            if previous and previous.get('identity')!=identity:raise ValueError('Different inputs/configuration; use an unchanged run for resume')
            can_update=True
            s['identity']=identity;s['mode']=mode;s['variant']=variant;s['rows']=5 if mode=='smoke' else 12576
            selected=test[:5] if mode=='smoke' else test
            verify_files(out,s['files_sha256'])
            if previous and previous['status']=='complete':print('ALREADY_COMPLETE',previous['output']);return
            save(mp,s)
            for phase,name in [(2,'result_critic_rationale_cot.json'),(3,'result_refine_cot.json')]:
                result=out/'2WikiMultiHopQA'/name
                if phase in s['completed_phases']:
                    rows_valid(result,selected);print('SKIP_PHASE',phase,flush=True);continue
                s.update(status='running',stage=f'phase{phase}',stage_log=str((out/f'phase{phase}.log').relative_to(ROOT)))
                save(mp,s);print('START_PHASE',phase,flush=True)
                invoke(build_command(p,out,phase,mode,variant),out/f'phase{phase}.log')
                rows_valid(result,selected)
                runtime=load(out/f'phase{phase}_runtime.json')
                expected_adapter=phase==2 and variant=='rl'
                if runtime['rows']!=len(selected) or runtime['submitted_lora_request_count']!=(len(selected) if expected_adapter else 0):
                    raise ValueError('Wrong adapter request coverage')
                if runtime['adapter_path']!=(str(p['adapter'].resolve()) if expected_adapter else None):raise ValueError('Wrong adapter used')
                s['completed_phases'].append(phase)
                for f in [result,out/f'phase{phase}_runtime.json']:s['files_sha256'][f.relative_to(out).as_posix()]=sha(f)
                save(mp,s);print(f'PHASE_{phase}_COMPLETE rows={len(selected)}',flush=True)
            rows=rows_valid(out/'2WikiMultiHopQA/result_refine_cot.json',selected)
            csv=out/(('preview' if mode=='smoke' else 'submission')+f'_qwen_{variant}_clm_full_rationale.csv')
            export_csv(rows,p['sample'],csv,preview=mode=='smoke')
            if sha(p['baseline'])!=EXPECTED['baseline']:raise ValueError('Original baseline changed')
            s['files_sha256'][csv.name]=sha(csv)
            s.update(status='complete',stage='evaluation_complete',output=str(csv),csv_sha256=sha(csv),
                     answer_format='full_rationale_verbatim',preview_only=mode=='smoke',finished=time.time(),
                     diagnostic_policy='No probability-difference probe or prior smoke required; provenance, adapter registry and explicit request routing checked.',
                     sampling=dict(qwen_temperature=.7,qwen_top_p=.8,critic_temperature=.6,critic_top_p=.9,seed=42),
                     limitations=['This is the original full-answer course protocol, not the pilot final_answer-tag metric.',
                                  'Runtime LoRA in vLLM BF16 may differ numerically from training-time PEFT.',
                                  'Eager mode, LoRA kernels and max_num_seqs=32 can change sampled outputs relative to the old SFT run. Matched SFT variant is available.'])
            save(mp,s);print('RL_2WIKI_COMPLETE' if variant=='rl' else 'SFT_2WIKI_COMPLETE',f'rows={len(rows)} csv={csv}',flush=True)
        except BaseException as exc:
            # Do not overwrite a valid previous manifest when an identity check fails.
            if not can_update:
                save(out/f'failed_attempt_{time.time_ns()}.json',dict(error=repr(exc)))
            else:
                s.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc));save(mp,s)
            raise
        finally:report(mode,variant)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['check','smoke','full','status','report'])
    ap.add_argument('--variant',choices=['rl','sft'],default='rl');ap.add_argument('--mode',choices=['smoke','full'],default='smoke');a=ap.parse_args()
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,stop)
    (ROOT/'logs').mkdir(exist_ok=True);(ROOT/'reports').mkdir(exist_ok=True)
    if a.command=='status':status(a.mode,a.variant)
    elif a.command=='report':report(a.mode,a.variant)
    elif a.command=='check':
        try:
            p=paths();anchor,_=verify_inputs(p);env=check_environment(p)
            save(ROOT/'reports/preflight.json',dict(status='passed',environment=env,training_fingerprint=anchor['training_run_fingerprint']))
            print('RL_2WIKI_INPUTS_OK',flush=True)
        except BaseException as exc:
            save(ROOT/'reports/preflight.json',dict(status='failed',error=repr(exc)));raise
        finally:report('smoke',a.variant)
    else:evaluate(a.command,a.variant)

if __name__=='__main__':main()
