"""Staged four-GPU online GRPO pilot; model processes run sequentially."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
from common import ROOT, CONFIG as C, path, load, save, readl, jsonl, fingerprint, sha, round_rows
from model_provenance import verify_package, verify_merge

ART=ROOT/'artifacts'; LOG=ROOT/'logs'; REPORT=ROOT/'reports'
ACTIVE=None


def report():
    target=ROOT/'clm_rl_grpo_pilot_v3_reports.tar.gz';allowed=[]
    for folder in [ART,LOG,REPORT,ROOT/'data']:
        if folder.exists():
            allowed += [p for p in folder.rglob('*') if p.is_file() and p.suffix in {'.json','.jsonl','.log','.txt'}]
    allowed += [p for p in ROOT.iterdir() if p.is_file() and p.suffix in {'.py','.json','.txt'}]
    temp=target.with_suffix('.tmp')
    with tarfile.open(temp,'w:gz') as t:
        for p in sorted(set(allowed)): t.add(p,arcname=p.relative_to(ROOT).as_posix())
    temp.replace(target)
    print('REPORT_READY',target,flush=True)


def interrupt(signum,frame):
    if ACTIVE is not None and ACTIVE.poll() is None: os.killpg(ACTIVE.pid,signal.SIGTERM)
    raise KeyboardInterrupt(f'Received signal {signum}')


def invoke(command,log):
    global ACTIVE
    env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                                    PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1')
    # Preserve diagnostics from unsuccessful attempts, rather than overwriting logs.
    if log.exists(): shutil.copy2(log,log.with_name(log.stem+f'.previous_{time.time_ns()}.log'))
    with log.open('w',encoding='utf-8') as f:
        ACTIVE=subprocess.Popen([str(x) for x in command],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try: rc=ACTIVE.wait()
        finally:
            if ACTIVE.poll() is None:
                os.killpg(ACTIVE.pid,signal.SIGTERM)
                try: ACTIVE.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(ACTIVE.pid,signal.SIGKILL);ACTIVE.wait()
            ACTIVE=None
    if rc: raise RuntimeError(f'Command exited {rc}; inspect {log.relative_to(ROOT)}')


def check():
    verify_package()
    for d in [ART,LOG,REPORT]:d.mkdir(exist_ok=True)
    if C['rounds']*C['groups_per_round']!=C['train_questions'] or C['candidates']!=4:
        raise ValueError('Expected one pass over train data with four candidates')
    for kind in ['training','inference']:
        if not path(kind+'_python').is_file(): raise FileNotFoundError(path(kind+'_python'))
        invoke([path(kind+'_python'),'check_environment.py',kind],LOG/f'check_{kind}.log')
    invoke([path('training_python'),'test_local.py'],LOG/'check_protocol.log')
    invoke([path('training_python'),'test_math.py'],LOG/'check_grpo_math.log')
    if shutil.disk_usage(ROOT).free<12*2**30: raise ValueError('Need at least 12 GiB free for 17 adapters, Adam states and reports')
    provenance=verify_merge(path('clm_model'))
    qwen={p.name:sha(p) for p in sorted(path('qwen_model').iterdir()) if p.is_file()}
    if not any(n.endswith('.safetensors') for n in qwen):raise ValueError('Qwen weights missing')
    spec=dict(config=C,source=load(ROOT/'source_manifest.json'),clm=provenance,qwen=qwen,
        paths={k:str(path(k)) for k in ['training_python','inference_python','qwen_model','clm_model']})
    ident=fingerprint(spec)
    if (ART/'spec.json').exists() and load(ART/'spec.json')['fingerprint']!=ident:
        raise ValueError('Configuration/source/models changed; refusing mixed-run resume')
    save(ART/'spec.json',dict(fingerprint=ident,spec=spec))
    print('GRPO_PILOT_ENVIRONMENT_OK',flush=True)
    return ident


def verify_marker(marker, ident):
    done=load(marker)
    if done['run_fingerprint']!=ident: raise ValueError('Stage specification changed')
    for p,h in done['files'].items():
        if sha(ROOT/p)!=h: raise ValueError('Completed stage output changed: '+p)


def stage(name,command,outputs,state):
    marker=ART/'stages'/f'{name}.json'
    if marker.exists():
        verify_marker(marker,state['run_fingerprint'])
        print('SKIP_COMPLETE',name,flush=True);return
    state.update(status='running',stage=name,stage_log=str(LOG/f'{name}.log'),updated=time.time())
    save(ART/'status.json',state);print('STAGE_START',name,flush=True)
    invoke(command,LOG/f'{name}.log')
    files={}
    for p in outputs:
        if not p.exists():raise FileNotFoundError(p)
        for f in ([p] if p.is_file() else sorted(p.rglob('*'))):
            if f.is_file() and f.suffix!='.tmp':files[f.relative_to(ROOT).as_posix()]=sha(f)
    if not files:raise ValueError('Stage produced no files')
    save(marker,dict(status='complete',run_fingerprint=state['run_fingerprint'],files=files))
    print('STAGE_COMPLETE',name,flush=True)


def combine(directory,number,control=False):
    records=[]
    for i in range(4):records+=readl(directory/f'rank{i}.jsonl')
    expected={(r['id'],c) for r in round_rows(number) for c in (['control'] if control else [f'sample{i}' for i in range(4)])}
    if len(records)!=len(expected) or {(r['id'],r['candidate']) for r in records}!=expected:
        raise ValueError('Wrong/duplicate rollout IDs')
    records.sort(key=lambda r:(r['id'],r['candidate']))
    jsonl(directory/'critique.jsonl',records)


def status():
    if not (ART/'status.json').exists():print('NOT_STARTED');return
    state=load(ART/'status.json');pid=state.get('pid')
    try: os.kill(pid,0);alive=True
    except (OSError,TypeError):alive=False
    updates=[load(p) for p in sorted(ART.glob('round*/update_rank0.json'))]
    state.update(process_alive=alive,planned_rounds=C['rounds'],completed_rounds=len(updates),
        optimizer_steps=sum(r['optimizer_steps'] for r in updates),
        effective_training_groups=sum(r.get('effective_groups',0) for r in updates))
    log=Path(state.get('stage_log',''))
    if log.is_file():state['current_log_tail']=log.read_text(encoding='utf-8',errors='replace').splitlines()[-6:]
    print(__import__('json').dumps(state,ensure_ascii=False,indent=2))


def run_pipeline(until='all'):
    import fcntl
    ART.mkdir(exist_ok=True)
    with (ART/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior=load(ART/'status.json') if (ART/'status.json').exists() else {}
        state=dict(status='checking',stage='environment',pid=os.getpid(),started=prior.get('started',time.time()))
        try:
            ident=check();state['run_fingerprint']=ident
            if prior.get('status')=='complete':
                for marker in (ART/'stages').glob('*.json'):verify_marker(marker,ident)
                print('ALREADY_COMPLETE',flush=True);return
            train=path('training_python');infer=path('inference_python')
            torchrun=[train,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=4','critic_worker.py']
            stage('reward_calibration',[infer,'score_stage.py','calibrate'],[REPORT/'reward_calibration.json'],state)
            if until=='reward-check':
                state.update(status='reward_checked',stage='reward_calibration_complete',updated=time.time());save(ART/'status.json',state);return
            stage('qwen_initial',[infer,'qwen_worker.py','initial','--output',ART/'initial.jsonl'],[ART/'initial.jsonl'],state)
            initial_adapter=ART/'checkpoints/step000'
            stage('initialize_adapter',torchrun+['init','--output',initial_adapter],[initial_adapter],state)
            def evaluate(name,number,adapter,greedy=False):
                out=ART/name
                cmd=torchrun+['rollout','--round',number,'--adapter',adapter,'--output',out]
                if greedy:cmd+=['--greedy']
                files=[out/f'rank{i}.jsonl' for i in range(4)]+[out/f'rollout_resource_rank{i}.json' for i in range(4)]
                stage(name+'_critic',cmd,files,state)
                combine(out,number,greedy)
                stage(name+'_qwen',[infer,'qwen_worker.py','refine','--input',out/'critique.jsonl','--output',out/'refine.jsonl'],[out/'refine.jsonl'],state)
                cmd=[infer,'score_stage.py','score','--directory',out,'--round',number]
                if greedy:cmd+=['--control']
                stage(name+'_score',cmd,[out/'scores.json',out/'scored.jsonl'],state)
                return out
            evaluate('dev_before',0,initial_adapter,True)
            if until=='baseline':
                state.update(status='baseline_complete',stage='dev_before_complete',updated=time.time());save(ART/'status.json',state);return
            adapter=initial_adapter;zero_streak=0;actual_steps=0
            for i in range(1,C['rounds']+1):
                out=evaluate(f'round{i}',i,adapter)
                new_adapter=ART/f'checkpoints/step{i:03d}'
                stage(f'round{i}_update',torchrun+['update','--round',i,'--adapter',adapter,'--output',out],
                    [new_adapter]+[out/f'update_rank{r}.json' for r in range(4)],state)
                update=load(out/'update_rank0.json');actual_steps+=update['optimizer_steps']
                zero_streak=zero_streak+1 if update['optimizer_steps']==0 else 0
                adapter=new_adapter
                state.update(completed_rounds=i,optimizer_steps=actual_steps,updated=time.time());save(ART/'status.json',state)
                if zero_streak>=C['max_zero_signal_rounds']:
                    raise RuntimeError('Three consecutive batches have no correctness variation; stop to inspect rewards/rollouts')
            evaluate('dev_after',0,adapter,True)
            stage('paired_summary',[train,'summarize.py'],[REPORT/'summary.json',REPORT/'paired_dev.jsonl'],state)
            state.update(status='complete',stage='pilot_complete',optimizer_steps=actual_steps,updated=time.time())
            save(ART/'status.json',state)
            print(f'GRPO_PILOT_COMPLETE optimizer_steps={actual_steps} dev_questions={C["dev_questions"]}',flush=True)
        except BaseException as exc:
            state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed_or_reward_gate_stopped',
                error=repr(exc),updated=time.time())
            save(ART/'status.json',state);traceback.print_exc();raise
        finally:report()


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['check','reward-check','baseline','all','status','report']);a=p.parse_args()
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,interrupt)
    if a.command=='status':status()
    elif a.command=='report':report()
    elif a.command=='check':
        try:check()
        finally:report()
    else:run_pipeline(a.command)


if __name__=='__main__':main()
