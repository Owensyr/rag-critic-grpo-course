"""Generate initial and revised answers with frozen Qwen. Never reads gold labels."""
import argparse
from pathlib import Path
from common import ROOT, CONFIG as C, load, readl, jsonl, prompt
from inference import Engine


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['initial','refine'])
    p.add_argument('--input',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=load(ROOT/'data/questions.json');byid={r['id']:r for r in rows};jobs=[]
    if a.mode=='initial':
        jobs=[dict(id=r['id'],variant='original',messages=prompt(r,'initial',field='original')) for r in rows]
    else:
        initials={r['id']:r for r in readl(ROOT/'artifacts/initial.jsonl')}
        for r in readl(a.input):
            jobs.append(dict(id=r['id'],candidate=r['candidate'],messages=prompt(byid[r['id']],'refine',
                initial=initials[r['id']]['text'],critique=r['text'])))
    engine=Engine();result=[]
    limit=C['qwen_initial_max_tokens'] if a.mode=='initial' else C['qwen_refine_max_tokens']
    for start in range(0,len(jobs),4):
        batch=jobs[start:start+4]
        outputs=engine.generate([j['messages'] for j in batch],limit)
        for j,out in zip(batch,outputs): result.append(dict(**{k:v for k,v in j.items() if k!='messages'},**out))
        jsonl(a.output,result)
        print(f'QWEN_{a.mode.upper()} {len(result)}/{len(jobs)}',flush=True)
    if a.mode=='initial' and any(r['finish_reason']!='stop' for r in result):
        raise ValueError('Initial Qwen output truncated; inspect this stage before training')
    print('QWEN_STAGE_COMPLETE',flush=True)


if __name__=='__main__':main()
