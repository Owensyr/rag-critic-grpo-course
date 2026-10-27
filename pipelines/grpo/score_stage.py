"""Isolated gold-aware reward process. Frozen Qwen judges only answer equivalence.

The label-blind extraction call has no accepted answers. Raw prompts/responses are
cached for audit. This is an adapted model-based reward, not an oracle evaluator.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from common import ROOT, CONFIG as C, load, save, readl, jsonl, round_rows, fingerprint
from inference import Engine
from logical_reward import country_relation_decision, unsupported_name_suffix
from reward import (EXTRACT_SYSTEM,JUDGE_SYSTEM,parse_json,parse_field,validate_extraction,
                    fast_decision,parse_decision,format_ok,summarize_groups,norm)


class Verifier:
    def __init__(self, cache_path=None):
        self.engine=Engine();self.path=cache_path or ROOT/'artifacts/reward_cache.json'
        self.cache=load(self.path) if self.path.exists() else {}
        spec=ROOT/'artifacts/spec.json'
        self.identity=load(spec)['fingerprint'] if spec.exists() else fingerprint(C)

    def request(self, system, payload, validator):
        messages=[dict(role='system',content=system),dict(role='user',content=json.dumps(payload,ensure_ascii=False))]
        traces=[]
        for attempt in range(C['judge_retries']+1):
            current=list(messages)
            if attempt: current[0]=dict(role='system',content=system+'\nReturn valid JSON only, following the required schema exactly. Do not output Markdown.')
            key=fingerprint(dict(identity=self.identity,messages=current,max_tokens=C['judge_max_tokens']))
            if key not in self.cache:
                self.cache[key]=dict(messages=current,output=self.engine.generate([current],C['judge_max_tokens'])[0])
                save(self.path,self.cache)
            out=self.cache[key]['output'];trace=dict(cache_key=key,**out);traces.append(trace)
            try:
                if out['finish_reason']!='stop': raise ValueError('Verifier output truncated')
                value=validator(parse_json(out['text']))
                return value,traces
            except (ValueError,TypeError) as exc: trace['error']=str(exc)
        return None,traces

    def extract(self, question, text):
        try: return parse_field(text),dict(method='explicit_final_field')
        except ValueError:
            answer,traces=self.request(EXTRACT_SYSTEM,dict(question=question,student_response=text),
                lambda obj:validate_extraction(obj,text))
            return answer,dict(method='label_blind_extraction',attempts=traces)

    def judge(self, question, answer, gold, force=False):
        if not force:
            if unsupported_name_suffix(question,answer,gold):
                return None,dict(method='unverified_name_suffix',reason='No supplied alias establishes identity with/without a generational suffix')
            logical=country_relation_decision(question,answer,gold)
            if logical is not None:
                return logical,dict(method='explicit_country_relation_truth_table')
            correct,rule=fast_decision(answer,gold,question)
            if correct is not None: return correct,dict(method=rule)
        correct,traces=self.request(JUDGE_SYSTEM,dict(question=question,accepted_answers=gold,student_answer=answer),parse_decision)
        return correct,dict(method='frozen_qwen_equivalence',attempts=traces)


def calibrate():
    verifier=Verifier();cases=load(ROOT/'data/reward_cases.json');results=[]
    for case in cases:
        got,trace=verifier.judge(case['question'],case['answer'],case['gold'])
        # Gate the exact deployed hybrid scorer; still disclose pure model weaknesses.
        if trace.get('method')=='frozen_qwen_equivalence':raw,raw_trace=got,trace
        else:raw,raw_trace=verifier.judge(case['question'],case['answer'],case['gold'],force=True)
        results.append(dict(**case,actual=got,passed=got is case['expected'],trace=trace,
                            raw_model_actual=raw,raw_model_passed=raw is case['expected'],raw_model_trace=raw_trace))
        print('JUDGE_CALIBRATION',len(results),'/',len(cases),case['id'],'pass=',results[-1]['passed'],flush=True)
    passed=sum(r['passed'] for r in results)
    critical_failed=[r['id'] for r in results if r.get('critical',False) and not r['passed']]
    # Evaluate the original suite independently: extra easy cases cannot dilute failures.
    suites={}
    for row in results:
        group=suites.setdefault(row.get('suite','original40'),dict(passed=0,total=0))
        group['total']+=1;group['passed']+=int(row['passed'])
    for group in suites.values():group['accuracy']=group['passed']/group['total']
    ready=all(s['accuracy']>=.95 for s in suites.values()) and not critical_failed
    fallback=[r for r in results if r['trace'].get('method')=='frozen_qwen_equivalence']
    raw_passed=sum(r['raw_model_passed'] for r in results)
    save(ROOT/'reports/reward_calibration.json',dict(status='passed' if ready else 'failed',passed=passed,
        total=len(results),accuracy=passed/len(results),critical_failed=critical_failed,suites=suites,cases=results,
        calibration_target='deployed_hybrid_scorer_v5',
        methods=dict(Counter(r['trace'].get('method','unspecified') for r in results)),
        unresolved=sum(r['actual'] is None for r in results),
        semantic_fallback=dict(total=len(fallback),passed=sum(r['passed'] for r in fallback)),
        raw_model_diagnostic=dict(passed=raw_passed,total=len(results),accuracy=raw_passed/len(results),
                                  failed_ids=[r['id'] for r in results if not r['raw_model_passed']],gate=False),
        limitation='Protocol regression fixtures, not an independent estimate of real 2Wiki judge accuracy.'))
    if not ready: raise RuntimeError('Reward verifier failed calibration; no RL updates run. Inspect reports/reward_calibration.json')
    print('REWARD_CALIBRATION_OK',flush=True)


def score(directory, number, control=False):
    questions={r['id']:r for r in round_rows(number)}
    gold={r['id']:r['answers'] for r in load(ROOT/'data/gold.json')}
    critics=readl(directory/'critique.jsonl');revised=readl(directory/'refine.jsonl')
    lookup={(r['id'],r['candidate']):r for r in revised}
    candidates=['control'] if control else [f'sample{i}' for i in range(C['candidates'])]
    expected={(qid,c) for qid in questions for c in candidates}
    if len(critics)!=len(expected) or len(revised)!=len(expected) or set(lookup)!=expected or {(r['id'],r['candidate']) for r in critics}!=expected:
        raise ValueError('Missing, duplicate, or unexpected records')
    verifier=Verifier();records=[];counts=Counter()
    for critic in critics:
        qid=critic['id'];ref=lookup[(qid,critic['candidate'])];q=questions[qid]['question']
        answer=None;correct=None;extraction={};judgement={}
        if ref['finish_reason']=='stop':
            answer,extraction=verifier.extract(q,ref['text'])
            if answer is not None: correct,judgement=verifier.judge(q,answer,gold[qid])
        r=dict(id=qid,candidate=critic['candidate'],question=q,gold=gold[qid],answer=answer,correct=correct,
            reward=None if correct is None else float(correct),critic_finished=critic['finish_reason']=='stop',
            qwen_finished=ref['finish_reason']=='stop',critic_format_ok=format_ok(critic['text']),
            critic_tokens=critic['completion_tokens'],extraction=extraction,judgement=judgement,
            gold_mentioned_diagnostic=any(norm(g) and norm(g) in norm(ref['text']) for g in gold[qid]))
        records.append(r)
        counts['correct']+=correct is True;counts['incorrect']+=correct is False;counts['unresolved']+=correct is None
        counts['truncated_critic']+=not r['critic_finished'];counts['truncated_qwen']+=not r['qwen_finished']
        counts['format_valid']+=r['critic_format_ok']
        counts['semantic_judged']+=judgement.get('method')=='frozen_qwen_equivalence'
        jsonl(directory/'scored.jsonl',records)
        print(f'SCORED {len(records)}/{len(critics)} correct={counts["correct"]} unresolved={counts["unresolved"]}',flush=True)
    grouped=summarize_groups(records,len(candidates));problems=[]
    if not control:
        if len(grouped['eligible_ids'])/len(questions)<C['min_eligible_group_fraction']:
            problems.append('Too few complete scorable groups; unresolved/truncated groups are excluded, never relabeled wrong')
        if counts['truncated_critic']/len(critics)>C['max_critic_truncation_fraction']:
            problems.append('Critique truncation exceeds configured guard')
    save(directory/'scores.json',dict(status='blocked_before_update' if problems else 'complete',policy=C['policy'],
        rows=len(records),counts=dict(counts),problems=problems,**grouped,
        accuracy_lower=counts['correct']/len(records),accuracy_upper=(counts['correct']+counts['unresolved'])/len(records),
        metric='Final-answer correctness from safe rules plus frozen-Qwen equivalence; unresolved is not incorrect.',
        reward_format_weight=0,retrieval_penalty=0))
    print('REWARD_CHECK',dict(counts),'eligible',len(grouped['eligible_ids']),'effective',len(grouped['effective_ids']),flush=True)
    if problems: raise RuntimeError('; '.join(problems))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['calibrate','score'])
    p.add_argument('--directory',type=Path);p.add_argument('--round',type=int,default=0);p.add_argument('--control',action='store_true');a=p.parse_args()
    if a.mode=='calibrate': calibrate()
    else: score(a.directory,a.round,a.control)


if __name__=='__main__':main()
