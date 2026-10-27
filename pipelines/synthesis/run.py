"""Resume 10K weak/strong/contrastive generation, then export candidate CFT records."""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from build import NAMES, QUOTAS, flags, key, load, sha

ROOT=Path(__file__).resolve().parent
ROLES=('weak','strong','critique')

def save(p,obj):
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(temp,p)

def read_lines(p):
    if not p.exists():return []
    with p.open(encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]

def records():return read_lines(ROOT/'data/train_10k.jsonl')

def verify():
    source=load(ROOT/'source_manifest.json')
    for n,h in source['files_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('Package file differs: '+n)
    rows=records();assert len(rows)==10000 and len({key(r['question']) for r in rows})==10000
    for name in NAMES:
        counts=Counter(r['hierarchy'] for r in rows if r['dataset']==name)
        assert counts=={'H1_random_other_question':200,'H2':400,**{'H34_support_'+str(n):QUOTAS[n] for n in range(1,6)}},(name,counts)
    for r in rows:
        assert len(r['ctxs'])==5
        a,f=flags(r,r['dataset'],r['ctxs']);assert a==r['support_flags'] and f==r['official_support_flags']
    print('CLM_10K_INPUTS_OK rows=10000',flush=True)
    return rows

def cft_templates():
    needed={'critic_rationale_system_prompt','critic_rationale_user_prompt','critic_rationale_output_prompt'};result={}
    for n in ast.parse((ROOT/'author/train_data_critic.py').read_text(encoding='utf-8')).body:
        if isinstance(n,ast.Assign):
            for target in n.targets:
                if isinstance(target,ast.Name) and target.id in needed:result[target.id]=ast.literal_eval(n.value)
    assert set(result)==needed
    return result

def text_context(r):
    docs=r['ctxs'][:5]
    if docs[0]['score']>docs[1]['score']:docs=docs[::-1]
    return '\n\n'.join(f"Document {i+1} (Title: {d['title']}): {d['text']}" for i,d in enumerate(docs))+'\n\n'

def question(q):
    if not q.endswith('?'):q+='?'
    if q.startswith('.'):q=q.lstrip('. ')
    return q[0].lower()+q[1:]

def worker(args):
    import yaml
    from vllm import LLM,SamplingParams
    sys.path.insert(0,str(ROOT/'author'))
    from data_prompt import RationalePromptTemplate,CritiqueRationalePromptTemplate
    role=args.worker;out=args.output;data=records();path=out/(role+'.jsonl')
    model=args.weak if role=='weak' else args.strong
    cfg=yaml.safe_load((ROOT/'author/data_generation.yaml').read_text(encoding='utf-8'));cfg['generator_model_path']=str(model)
    upstream={}
    if role=='critique':
        weak=read_lines(out/'weak.jsonl');strong=read_lines(out/'strong.jsonl')
        assert [r['id'] for r in weak]==[r['id'] for r in strong]==[r['id'] for r in data]
        upstream={n:sha(out/(n+'.jsonl')) for n in ('weak','strong')}
    spec={'role':role,'model':str(model),'model_config_sha256':sha(model/'config.json'),'data_sha256':sha(ROOT/'data/train_10k.jsonl'),'package_sha256':sha(ROOT/'source_manifest.json'),'upstream_sha256':upstream,'generation':cfg['generation_params'],'seed_policy':'42 + fixed row position','tensor_parallel_size':1 if role=='weak' else 4,'batch_size':args.batch_size,'versions':{n:importlib.metadata.version(n) for n in ('torch','vllm','transformers')}}
    statepath=out/(role+'.manifest.json');done=read_lines(path)
    if statepath.exists():
        previous=load(statepath)
        if previous['spec']!=spec:raise ValueError('Resume configuration differs: '+role)
        if previous.get('status')=='complete' and previous.get('sha256')!=sha(path):raise ValueError('Completed checkpoint changed: '+role)
    elif done:raise ValueError('Checkpoint has no manifest: '+role)
    assert [r['id'] for r in done]==[r['id'] for r in data[:len(done)]] and len(done)<=len(data)
    state={'spec':spec,'status':'running','completed':len(done),'target':len(data)}
    if len(done)==len(data):
        state.update(status='complete',sha256=sha(path));save(statepath,state);print('SKIP_COMPLETE',role,flush=True);return
    save(statepath,state)
    cls=CritiqueRationalePromptTemplate if role=='critique' else RationalePromptTemplate
    template=cls(cfg);postfix=load(ROOT/'author/rag.json')
    # Intercept the author's truncation hook: fail clearly rather than silently cut training evidence.
    def no_truncate(prompt):
        n=len(template.tokenizer.encode(prompt,add_special_tokens=False))
        if n>cfg['generator_max_input_len']:raise ValueError(f'Input exceeds author 8192-token limit: {n}; review before proceeding')
        return prompt
    template.truncate_prompt=no_truncate
    engine=LLM(model=str(model),dtype='bfloat16',tensor_parallel_size=spec['tensor_parallel_size'],max_model_len=32768,gpu_memory_utilization=0.9,seed=42)
    gp=cfg['generation_params']
    try:
        with path.open('a',encoding='utf-8',newline='\n') as f:
            for start in range(len(done),len(data),args.batch_size):
                batch=data[start:start+args.batch_size];prompts=[];prompt_texts=[];sampling=[]
                for j,r in enumerate(batch,start):
                    kw={'query':question(r['question']),'text':text_context(r)}
                    if role=='critique':kw.update(rationale=weak[j]['text'],gold=strong[j]['text'])
                    else:kw.update(gold=r['answers'],task_specific_instruction=postfix['rationale_generation_postfix_'+r['dataset']])
                    prompt=template.get_string(**kw);prompt_texts.append(prompt)
                    prompts.append({'prompt_token_ids':template.tokenizer.encode(prompt,add_special_tokens=False)})
                    sampling.append(SamplingParams(temperature=gp['temperature'],top_p=gp['top_p'],max_tokens=gp['max_tokens'],seed=42+j))
                preds=engine.generate(prompts,sampling,use_tqdm=False);assert len(preds)==len(batch)
                for r,p,prompt,tokens in zip(batch,preds,prompt_texts,prompts):
                    c=p.outputs[0]
                    if not c.text.strip():raise ValueError('Empty completion '+r['id'])
                    row={'id':r['id'],'text':c.text,'finish_reason':c.finish_reason,'output_tokens':len(c.token_ids),'input_tokens':len(tokens['prompt_token_ids']),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
                    f.write(json.dumps(row,ensure_ascii=False)+'\n')
                f.flush();os.fsync(f.fileno())
                state.update(completed=start+len(batch),updated_utc=datetime.now(timezone.utc).isoformat());save(statepath,state)
                print(f"PROGRESS {role} {state['completed']}/{len(data)}",flush=True)
        state.update(status='complete',sha256=sha(path));save(statepath,state)
    except BaseException as e:
        state.update(status='failed',error=repr(e));save(statepath,state);raise

def export(out,data):
    result={role:read_lines(out/(role+'.jsonl')) for role in ROLES}
    for rows in result.values():assert [r['id'] for r in rows]==[r['id'] for r in data]
    t=cft_templates();previews=[];target=out/'cft_train_10k.json';training=[];metrics={role:Counter() for role in ROLES};selected=set(r['id'] for r in load(ROOT/'data/selection_preview70.json'))
    for i,r in enumerate(data):
        w,s,c=[result[role][i] for role in ROLES]
        kw={'query':question(r['question']),'text':text_context(r),'rationale':w['text'],'critique':c['text'],'gold_rationale':s['text']}
        training.append({k:t['critic_rationale_'+name+'_prompt'].format(**kw) for k,name in [('instruction','system'),('input','user'),('output','output')]})
        for role in ROLES:
            v=result[role][i];metrics[role]['rows']+=1;metrics[role]['length_limited']+=v['finish_reason']=='length'
            metrics[role]['other_finish_reason']+=v['finish_reason'] not in ('stop','length')
            if role!='critique':
                a,f=flags(r,r['dataset'],[{'title':'','text':v['text']}]);metrics[role]['any_answer_presence']+=a[0];metrics[role]['all_subquestion_presence']+=f[0]
        if r['id'] in selected:previews.append({**r,**{role:result[role][i] for role in ROLES}})
    save(target,training);save(out/'review_preview70.json',previews)
    quality={'rows':len(data),'training_quality_approved':False,'metrics':metrics,'notes':['Answer presence is diagnostic only, not factual verification; ASQA any/all counts differ.','No quality filtering or manual answer correction has been applied.','Length-limited completions remain in candidate CFT export and require review before training.'],'cft_sha256':sha(target)}
    save(out/'quality_summary.json',quality)
    return quality

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','all','status'],nargs='?',default='check')
    p.add_argument('--worker',choices=ROLES,help=argparse.SUPPRESS)
    p.add_argument('--weak',type=Path,default=Path.home()/'RAG/models_verified/Qwen/Qwen2.5-0.5B-Instruct')
    p.add_argument('--strong',type=Path,default=Path.home()/'RAG/models_verified/meta-llama/Llama-3.1-8B-Instruct')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/full')
    p.add_argument('--batch-size',type=int,default=32);args=p.parse_args()
    args.output=args.output.expanduser().resolve();args.weak=args.weak.expanduser().resolve();args.strong=args.strong.expanduser().resolve()
    if args.batch_size<1:raise ValueError('batch-size must be positive')
    if args.worker:worker(args);return
    if args.command=='status':
        print(json.dumps({'manifest':load(args.output/'run_manifest.json') if (args.output/'run_manifest.json').exists() else None,'completed_rows':{role:len(read_lines(args.output/(role+'.jsonl'))) for role in ROLES}},ensure_ascii=False,indent=2));return
    data=verify()
    if args.command=='check':return
    for model in (args.weak,args.strong):
        if not (model/'config.json').exists():raise FileNotFoundError(model)
    args.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (args.output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest={'status':'running','rows':len(data),'completed_roles':[],'pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),'training_quality_approved':False,'source_manifest_sha256':sha(ROOT/'source_manifest.json')}
        mp=args.output/'run_manifest.json';save(mp,manifest)
        try:
            for role in ROLES:
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'0' if role=='weak' else '0,1,2,3','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
                cmd=[sys.executable,'-u',str(ROOT/'run.py'),'--worker',role,'--weak',str(args.weak),'--strong',str(args.strong),'--output',str(args.output),'--batch-size',str(args.batch_size)]
                print('START_ROLE',role,flush=True);subprocess.run(cmd,cwd=ROOT,env=env,check=True)
                manifest['completed_roles'].append(role);save(mp,manifest)
            export(args.output,data)
            manifest.update(status='complete',output_sha256={n:sha(args.output/n) for n in ['cft_train_10k.json','quality_summary.json','review_preview70.json']})
        except BaseException as e:manifest.update(status='failed',error=repr(e));raise
        finally:manifest['updated_utc']=datetime.now(timezone.utc).isoformat();save(mp,manifest)
    print('CLM_CFT_SYNTHESIS_COMPLETE rows=10000 training_started=false',flush=True)

if __name__=='__main__':main()
