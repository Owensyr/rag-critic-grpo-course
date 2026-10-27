"""Matched SFT-vs-GRPO evaluation; preserve unresolved cases and paired transitions."""
import random
from common import ROOT,CONFIG as C,load,save,readl,jsonl


def paired_summary(before,after):
    b={r['id']:r for r in before};a={r['id']:r for r in after}
    if len(b)!=len(before) or len(a)!=len(after) or set(a)!=set(b):raise ValueError('Paired evaluation IDs differ')
    pairs=[dict(id=qid,before=b[qid]['correct'],after=a[qid]['correct'],
        before_answer=b[qid]['answer'],after_answer=a[qid]['answer'],question=b[qid]['question'],gold=b[qid]['gold']) for qid in b]
    usable=[p for p in pairs if p['before'] is not None and p['after'] is not None]
    improved=sum(p['before'] is False and p['after'] is True for p in usable)
    harmed=sum(p['before'] is True and p['after'] is False for p in usable)
    def metric(rows):
        n=len(rows);correct=sum(r['correct'] is True for r in rows);missing=sum(r['correct'] is None for r in rows)
        return dict(total=n,correct=correct,unresolved=missing,accuracy=correct/n if not missing else None,
                    accuracy_lower=correct/n,accuracy_upper=(correct+missing)/n)
    differences=[int(p['after'])-int(p['before']) for p in usable]
    ci=None
    if differences:
        rng=random.Random(20261026)
        boot=sorted(sum(rng.choice(differences) for _ in differences)/len(differences) for _ in range(5000))
        ci=[boot[125],boot[4874]]
    return dict(before=metric(before),after=metric(after),paired_scorable=len(usable),wrong_to_correct=improved,
        correct_to_wrong=harmed,paired_delta=(improved-harmed)/len(usable) if usable else None,
        paired_bootstrap_95_interval=ci,all_pairs_scored=len(usable)==len(pairs),pairs=pairs)


def main():
    art=ROOT/'artifacts';report=ROOT/'reports'
    before=readl(art/'dev_before/scored.jsonl');after=readl(art/'dev_after/scored.jsonl')
    summary=paired_summary(before,after)
    if len(before)!=C['dev_questions']:raise ValueError('Wrong dev denominator')
    jsonl(report/'paired_dev.jsonl',summary.pop('pairs'))
    rounds=[]
    for i in range(1,C['rounds']+1):
        scores=load(art/f'round{i}/scores.json');updates=[load(art/f'round{i}/update_rank{r}.json') for r in range(4)]
        if len({u['adapter_after'] for u in updates})!=1:raise ValueError('Replica hashes differ')
        rounds.append(dict(round=i,counts=scores['counts'],eligible=len(scores['eligible_ids']),effective=len(scores['effective_ids']),updates=updates))
    save(report/'summary.json',dict(status='pilot_complete',policy=C['policy'],evaluation=summary,
        train_questions=C['train_questions'],dev_questions=C['dev_questions'],planned_rounds=C['rounds'],
        optimizer_steps=sum(r['updates'][0]['optimizer_steps'] for r in rounds),rounds=rounds,
        checkpoint=str(art/f'checkpoints/step{C["rounds"]:03d}'),
        comparison='Fixed final checkpoint versus initial SFT, same 128 dev questions and generation settings; no best-checkpoint selection.',
        limitations=['Single seed and 128 validation questions; small changes do not establish a reliable improvement.',
          'Correctness uses rule checks plus frozen Qwen as a semantic verifier; calibration fixtures cannot guarantee judge accuracy.',
          'Qwen is both the frozen answer model and the isolated verifier; systematic shared errors may remain.',
          'Refinement prompts include a final-answer field for this pilot, unlike the original full-answer Kaggle protocol.',
          'One optimizer update per rollout batch: pre-update ratios are one, so clipping is normally inactive for this on-policy step.',
          'Truncated/unresolved groups are excluded; effective data counts are reported. No format bonus or retrieval penalty.',
          'Weights and optimizer are retained on server and omitted from the report archive.']))
    print('PAIRED_EVALUATION',summary,flush=True)


if __name__=='__main__':main()
