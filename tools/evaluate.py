"""Portable orchestration of unchanged, historically used evaluation workers."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time
from common import ROOT,configuration,load,save,sha,abspath,verify_model
from export_submission import export,template_ids

def plan(c,group,mode,out,baseline):
    worker=ROOT/'pipelines'/('rl_eval' if group=='rl' else 'qwen')
    initial_root=out if group=='qwen' else out/'input_baseline'
    critic=c['llama'] if group=='frozen' else c['sft']
    commands=[]
    for phase in ([1] if group=='qwen' else [2,3]):
        cmd=[c['inference_python'],'-u','eval/inference_vllm.py','--data_path',c['test'],
             '--dataset_name','2WikiMultiHopQA','--model_name_or_path',c['qwen'],
             '--config_file','configs/eval.yaml','--type','cot','--phase',str(phase),'--iters','1',
             '--n_docs','5','--seed','42','--max_tokens','5120','--max_model_len','32768',
             '--output_dir',str(initial_root)]
        if mode=='smoke':cmd+=['--max_instances','5']
        if group!='qwen':cmd+=['--critique_model',critic,'--critique_model_path',str(out)]
        if group=='rl':
            cmd+=['--gpu_memory_utilization','0.85']
            if phase==2:cmd+=['--rl_adapter',c['rl_adapter']]
        commands.append({'phase':phase,'cwd':str(worker),'command':cmd})
    return commands
def validate_rows(file,selected):
    rows=load(file)
    if len(rows)!=len(selected):raise ValueError('Incomplete phase output')
    for r,s in zip(rows,selected):
        if str(r['id'])!=str(s['id']) or r['question']!=s['question']:raise ValueError('Question/ID mismatch')
        if not isinstance(r.get('rationale'),str) or not r['rationale'].strip():raise ValueError('Empty rationale')
    return rows
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default=ROOT/'configs/paths.json')
    p.add_argument('--group',choices=['qwen','frozen','sft','rl'],required=True)
    p.add_argument('--mode',choices=['smoke','full'],default='smoke');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline',type=Path);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    c=configuration(a.config);out=abspath(a.output);baseline=abspath(a.baseline or c['baseline'])
    commands=plan(c,a.group,a.mode,out,baseline)
    if a.dry_run:
        import json;print(json.dumps(commands,indent=2));return
    if out.exists():raise FileExistsError('Use a NEW output directory; partial stages are preserved: '+str(out))
    gpu_ids=c['gpu_ids'].split(',')
    if len(gpu_ids)!=4 or len(set(gpu_ids))!=4 or any(not x.isdigit() for x in gpu_ids):raise ValueError('This historical profile requires four distinct numeric GPU IDs')
    data=load(c['test']);ids=template_ids(c['sample'])
    if len(data)!=12576 or [str(r['id']) for r in data]!=ids or len(set(ids))!=12576:raise ValueError('Expected 12576 aligned test questions')
    if any(r.get('answers') is not None for r in data):raise ValueError('Use unlabeled course test data')
    if a.group!='qwen':validate_rows(baseline,data)
    models={k:verify_model(c[k]) for k in ['qwen']+([] if a.group=='qwen' else ['llama' if a.group=='frozen' else 'sft'])}
    adapter=None
    if a.group=='rl':
        ac=load(Path(c['rl_adapter'])/'adapter_config.json')
        if ac['r']!=8 or ac['lora_alpha']!=16:raise ValueError('Expected RL rank8 alpha16 adapter')
        adapter={n:sha(Path(c['rl_adapter'])/n) for n in ['adapter_config.json','adapter_model.safetensors']}
        spec_path=Path(c['rl_adapter']).parent.parent/'spec.json'
        if not spec_path.exists():spec_path=Path(c['private_root'])/'evidence/rl_training_spec.json'
        spec=load(spec_path)['spec'];parent=spec['clm']['files_sha256']
        expected={n:h for n,h in parent.items() if n.endswith('.safetensors')}
        actual={n:h for n,h in models['sft'].items() if n.endswith('.safetensors')}
        if not expected or actual!=expected:
            raise ValueError('RL adapter requires the exact SFT base used for training. Restore the original merged SFT, or use the checkpoint and spec from your new GRPO run.')
    out.mkdir(parents=True)
    if a.group!='qwen':
        target=out/'input_baseline/2WikiMultiHopQA/result_rationale_cot.json';target.parent.mkdir(parents=True)
        shutil.copyfile(baseline,target)
    state={'status':'running','group':a.group,'mode':a.mode,'started':time.time(),'completed_phases':[],
           'test_sha256':sha(c['test']),'sample_sha256':sha(c['sample']),
           'baseline_sha256':sha(baseline) if a.group!='qwen' else None,'models':models,'adapter':adapter,'commands':commands,
           'implementation':'Portable launcher; numerical workers copied from historical runs; not a bitwise reproduction guarantee'}
    worker=Path(commands[0]['cwd'])
    state['worker_source_sha256']={p.relative_to(worker).as_posix():sha(p) for p in sorted(worker.rglob('*'))
        if p.is_file() and p.suffix in {'.py','.yaml'} and '__pycache__' not in p.parts}
    state['launcher_sha256']=sha(Path(__file__))
    selected=data[:5] if a.mode=='smoke' else data
    env={**os.environ,'CUDA_VISIBLE_DEVICES':c['gpu_ids'],'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'}
    try:
        for task in commands:
            phase=task['phase'];state['stage']=f'phase{phase}';save(out/'run_manifest.json',state)
            print('START_PHASE',phase,flush=True)
            with (out/f'phase{phase}.log').open('w',encoding='utf-8') as f:
                subprocess.run(task['command'],cwd=task['cwd'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            filename={1:'result_rationale_cot.json',2:'result_critic_rationale_cot.json',3:'result_refine_cot.json'}[phase]
            result=out/'2WikiMultiHopQA'/filename;validate_rows(result,selected)
            if a.group=='rl':
                rt=load(out/f'phase{phase}_runtime.json');wanted=len(selected) if phase==2 else 0
                if rt['rows']!=len(selected) or rt['submitted_lora_request_count']!=wanted:raise ValueError('RL request coverage mismatch')
                if phase==2 and (rt['registered_adapter_ids']!=[16] or rt['adapter_path']!=str(Path(c['rl_adapter']).resolve())):raise ValueError('RL adapter routing mismatch')
                if phase==3 and rt['adapter_path'] is not None:raise ValueError('Qwen must not use a critic adapter')
            state['completed_phases'].append(phase);print('PHASE_COMPLETE',phase,flush=True)
        csv=out/f'{"preview" if a.mode=="smoke" else "submission"}_{a.group}_full_rationale.csv'
        state['export']=export(result,c['sample'],csv,a.mode=='smoke')
        state.update(status='complete',output=str(csv),finished=time.time());print('EVALUATION_COMPLETE',csv,flush=True)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:save(out/'run_manifest.json',state)
if __name__=='__main__':main()
