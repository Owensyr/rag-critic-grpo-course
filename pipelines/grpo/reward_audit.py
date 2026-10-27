"""Offline sensitivity audit. Does not modify the original reward package or train models."""
import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import string
import tarfile


def norm(s):
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s.lower().translate(str.maketrans('','',string.punctuation))).split())


def extract(text):
    if text.count('<final_answer>')!=1 or text.count('</final_answer>')!=1:
        raise ValueError('Expected exactly one explicit answer field')
    m=re.search(r'<final_answer>([^<>]+)</final_answer>',text,re.S)
    if not m or not m.group(1).strip():raise ValueError('Empty or malformed answer field')
    return m.group(1).strip(),bool(text[m.end():].strip())


MONTHS={s:i for i,s in enumerate('january february march april may june july august september october november december'.split(),1)}


def canonical_date(s):
    s=s.strip().lower()
    pats=[(r'(\d{1,2})\s+([a-z]+)\s+(\d{4})',(3,2,1)),
          (r'([a-z]+)\s+(\d{1,2}),?\s+(\d{4})',(3,1,2))]
    for pattern,(yi,mi,di) in pats:
        m=re.fullmatch(pattern,s)
        if m and m.group(mi) in MONTHS:
            try:return date(int(m.group(yi)),MONTHS[m.group(mi)],int(m.group(di))).isoformat()
            except ValueError:return None
    # Unambiguous ISO only; never guess MM/DD versus DD/MM.
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}',s):
        try:return date.fromisoformat(s).isoformat()
        except ValueError:pass
    return None


def match(answer,gold,question):
    if norm(answer) in {norm(x) for x in gold}:return True,'normalized_exact_match'
    d=canonical_date(answer)
    if d and any(d==canonical_date(g) for g in gold):return True,'unambiguous_date_equivalence'
    # Explicit, scoped, provisional mapping, not a universal country/demonym resolver.
    us={'american','united states','united states of america','usa','us'}
    if re.search(r'\b(country|nationality|nationalities)\b',question.lower()):
        if norm(answer) in us and any(norm(g) in us for g in gold):return True,'scoped_us_country_demonym_equivalence'
    return False,'no_supported_equivalence'


def measure(record,gold,question):
    if record['finish_reason']!='stop':return dict(correct=None,reason='nonstop_generation')
    try:a,trailing=extract(record['text'])
    except ValueError as e:return dict(correct=None,reason=str(e))
    correct,reason=match(a,gold,question)
    return dict(correct=correct,reason=reason,answer=a,has_text_after_field=trailing)


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with tarfile.open(a.report) as t:
        read=lambda n:t.extractfile(n).read()
        manifest=json.loads(read('artifacts/pilot32/run_manifest.json'))
        for name,h in manifest['output_sha256'].items():
            assert hashlib.sha256(read('artifacts/pilot32/'+name)).hexdigest()==h
        original=json.loads(read('artifacts/pilot32/reward_summary.json'))
        rows=[json.loads(line) for line in read('artifacts/pilot32/review32.jsonl').decode().splitlines()]
    counts=Counter();groups=[];changed=[];reviews=[]
    for r in rows:
        base=measure(r['baseline_record'],r['gold_aliases'],r['question'])
        counts['baseline_total']+=1;counts['baseline_correct']+=base['correct'] is True
        counts['baseline_unparsable']+=base['correct'] is None
        candidates=[];sample_correct=[];sensitivity_rewards=[]
        for c in r['candidates']:
            kind='control' if c['candidate']=='control' else 'sampled'
            new=measure(c['refinement'],r['gold_aliases'],r['question'])
            counts[kind+'_total']+=1;counts[kind+'_correct']+=new['correct'] is True
            counts[kind+'_unparsable']+=new['correct'] is None
            counts[kind+'_equivalence_'+new['reason']]+=1
            if new['correct']!=c['outcome']['correct']:
                changed.append(dict(id=r['id'],candidate=c['candidate'],gold=r['gold_aliases'],old=c['outcome'],new=new))
            if base['correct'] is not None and new['correct'] is not None:
                counts[f'{kind}_transition_{int(base["correct"])}_{int(new["correct"])}']+=1
            # Sensitivity variant only, NOT a silently deployed reward update.
            if c['critique']['finish_reason']!='stop':soft=0.0
            elif new['correct'] is None:soft=None
            else:soft=round(0.9*int(new['correct'])+0.1*int(c['critic_format_ok']),2)
            if kind=='sampled':sample_correct.append(new['correct']);sensitivity_rewards.append(soft)
            candidates.append(dict(candidate=c['candidate'],outcome=new,proposed_outcome_first_reward=soft,
                                   old_reward=c['reward'],critic_format_ok=c['critic_format_ok'],critic_finish=c['critique']['finish_reason']))
        groups.append(dict(id=r['id'],outcomes=sample_correct,
            outcome_variance=None if None in sample_correct else len(set(sample_correct))>1,
            proposed_rewards=sensitivity_rewards))
        reviews.append(dict(id=r['id'],question=r['question'],gold=r['gold_aliases'],baseline=base,candidates=candidates))
    result=dict(status='offline_audit_complete',source_archive_sha256=hashlib.sha256(a.report.read_bytes()).hexdigest(),
        original_summary={k:v for k,v in original.items() if k!='groups'},diagnostic_rescore_counts=dict(counts),
        diagnostic_groups_with_outcome_variance=sum(g['outcome_variance'] is True for g in groups),
        original_format_only_variance_groups=[g['id'] for g in original['groups'] if g['reward_has_variance'] and not g['outcome_has_variance']],
        proposed_changes=['Allow one explicit answer field outside final position; retain trailing-text flag for manual review.',
                          'Normalize unambiguous full dates; never guess ambiguous numeric dates.',
                          'Provisional scoped US country/demonym equivalence for country/nationality questions.',
                          'For sensitivity only: 0.9 correctness + 0.1 strict format on completed critiques; unfinished critique reward 0.'],
        changed_outcomes=changed,groups=groups,review=reviews,
        limitations=['This is a diagnostic rescore on the reward development set, not a model improvement.',
          'US mapping is limited and not a comprehensive entity-alias solution; independent validation required.',
          'Wrong reasoning may still produce a correct final answer; this reward does not prove reasoning correctness.',
          '29/32 initial responses contain explicit unknown; prompt and decoding effects need a matched control.',
          'Original files, prompts and reward.py remain unchanged; no GRPO updates executed.'],
        next_step='Fix reward protocol and verify an independent development sample, then a short GRPO engineering smoke test; do not start full training with v1 rewards.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(counts),indent=2));print('OUTCOME_VARIANCE_GROUPS',result['diagnostic_groups_with_outcome_variance'])


if __name__=='__main__':main()
