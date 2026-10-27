"""CPU/stdlib regression tests for reward isolation, candidate mapping and resume."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from common import ROOT,CONFIG as C,load,save,prompt,group_advantages,readl
from reward import (parse_field,parse_decision,fast_decision,validate_extraction,summarize_groups,
                    EXTRACT_SYSTEM,JUDGE_SYSTEM)


class Tests(unittest.TestCase):
    def test_interpreter_symlink_not_resolved(self):
        import common,os
        with patch.object(Path,'resolve',side_effect=AssertionError('Do not resolve bin/python')):
            self.assertEqual(common.path('training_python'),Path(os.path.abspath(os.path.expanduser(C['training_python']))))

    def test_dataset_is_disjoint_and_label_isolated(self):
        from build_data_base import key
        rows=load(ROOT/'data/questions.json');gold=load(ROOT/'data/gold.json')
        self.assertEqual(len(rows),C['train_questions']+C['dev_questions'])
        self.assertEqual(len({r['id'] for r in rows}),len(rows))
        self.assertEqual(len({key(r['question']) for r in rows}),len(rows))
        self.assertEqual({r['id'] for r in rows},{r['id'] for r in gold})
        self.assertEqual(sum(r['split']=='dev' for r in rows),C['dev_questions'])
        for row in rows:
            self.assertEqual(set(row),{'id','source_index','question','ctxs','split'})
            for d in row['ctxs']:self.assertEqual(set(d),{'title','text','score'})
        audit=load(ROOT/'data/selection_audit.json')
        self.assertTrue(all(audit[k]==0 for k in ['overlap_sft','overlap_previous_rl','overlap_evaluation','overlap_train_dev']))
        sentinel=dict(rows[0],answers=['NEVER_EXPOSE_THIS_GOLD'])
        for stage in ['initial','critique','refine']:
            self.assertNotIn('NEVER_EXPOSE_THIS_GOLD',str(prompt(sentinel,stage,initial='text',critique='critique')))
        for name in ['critic_worker.py','qwen_worker.py']:
            self.assertNotIn('gold.json',(ROOT/name).read_text(encoding='utf-8'))

    def test_no_false_negative_from_unsupported_rule(self):
        self.assertEqual(fast_decision('Nicosia, Cyprus',['Nicosia'],'Where was he born?')[0],None)
        self.assertEqual(fast_decision('He died due to a heart attack.',['heart attack'],'What caused his death?')[0],None)
        self.assertEqual(fast_decision('Both are in the same country.',['yes'],'Are they in the same country?')[0],None)
        self.assertEqual(fast_decision('March 22, 1948',['22 March 1948'],'When?')[0],True)
        self.assertEqual(fast_decision('unknown',['Nicosia'],'Where?')[0],False)

    def test_no_substring_reward(self):
        for text in ['Not Nicosia','Nicosia or Limassol','Nicosia was considered, but I cannot determine the answer.']:
            self.assertIsNone(fast_decision(text,['Nicosia'],'Where?')[0])
        self.assertIsNone(fast_decision('12',['2'],'How many?')[0])

    def test_answer_extraction(self):
        self.assertEqual(parse_field('reason\n<final_answer>Nicosia</final_answer>.'),'Nicosia')
        for text in ['missing','<final_answer></final_answer>','<final_answer>A</final_answer><final_answer>B</final_answer>',
                     '<final_answer>A</final_answer> Actually B.']:
            with self.assertRaises(ValueError):parse_field(text)
        original='He was born in Nicosia, Cyprus.'
        self.assertEqual(validate_extraction(dict(answer='Nicosia, Cyprus',quote=original),original),'Nicosia, Cyprus')
        with self.assertRaises(ValueError):validate_extraction(dict(answer='London',quote=original),original)
        with self.assertRaises(ValueError):validate_extraction(dict(answer=None,quote=original),original)
        self.assertEqual(validate_extraction(dict(answer=None,quote='I cannot determine it.'),'I cannot determine it.'),'unknown')

    def test_uncertain_is_not_zero(self):
        self.assertIsNone(parse_decision(dict(decision='uncertain',reason='ambiguous')))
        with self.assertRaises(ValueError):parse_decision(dict(decision=True,reason='invalid schema'))

    def test_format_differences_do_not_create_advantage(self):
        self.assertEqual(group_advantages([0,0,0,0]),[0]*4)
        self.assertEqual(group_advantages([1,1,1,1]),[0]*4)
        a=group_advantages([1,0,1,0]);self.assertGreater(a[0],0);self.assertLess(a[1],0)
        rows=[dict(id='q',candidate=f'sample{i}',correct=False,reward=0.,critic_finished=True,qwen_finished=True,critic_format_ok=bool(i%2)) for i in range(4)]
        s=summarize_groups(rows)
        self.assertEqual(s['eligible_ids'],['q']);self.assertFalse(s['effective_ids'])
        rows[0].update(correct=None,reward=None)
        self.assertFalse(summarize_groups(rows)['eligible_ids'])
        rows[0].update(correct=True,reward=1.,critic_finished=False)
        self.assertFalse(summarize_groups(rows)['eligible_ids'])

    def test_verifier_fallback_is_label_blind_and_cached(self):
        import score_stage
        calls=[]
        class FakeEngine:
            def generate(self,messages,max_tokens):
                calls.extend(messages)
                if messages[0][0]['content']==EXTRACT_SYSTEM:
                    text=json.dumps(dict(answer='Nicosia, Cyprus',quote='He was born in Nicosia, Cyprus.'))
                else:text=json.dumps(dict(decision='correct',reason='City with country qualifier'))
                return [dict(text=text,finish_reason='stop')]
        with tempfile.TemporaryDirectory() as t,patch.object(score_stage,'Engine',FakeEngine):
            v=score_stage.Verifier(Path(t)/'cache.json')
            answer,_=v.extract('Where?','He was born in Nicosia, Cyprus.')
            self.assertEqual(answer,'Nicosia, Cyprus')
            self.assertNotIn('accepted_answers',calls[0][1]['content'])
            self.assertTrue(v.judge('Where?',answer,['Nicosia'])[0])
            self.assertIn('accepted_answers',calls[1][1]['content'])
            self.assertTrue(v.judge('Where?',answer,['Nicosia'])[0])
            self.assertEqual(len(calls),2)

    def test_bad_verifier_json_remains_unresolved(self):
        import score_stage
        class Bad:
            def generate(self,*args):return [dict(text='not json',finish_reason='stop')]
        with tempfile.TemporaryDirectory() as t,patch.object(score_stage,'Engine',Bad):
            v=score_stage.Verifier(Path(t)/'cache.json')
            self.assertIsNone(v.judge('Where?','answer',['gold'])[0])

    def test_calibration_critical_failure_stops(self):
        import score_stage
        # A high aggregate score must not hide a failure on an explicit negation.
        cases=[dict(id=str(i),question='q',gold=['g'],answer='a',expected=True,critical=False) for i in range(39)]
        cases.append(dict(id='negation',question='q',gold=['g'],answer='not g',expected=False,critical=True))
        class Overaccepting:
            def judge(self,*args,**kwargs):return True,{}
        with tempfile.TemporaryDirectory() as t,patch.object(score_stage,'ROOT',Path(t)),\
             patch.object(score_stage,'load',return_value=cases),patch.object(score_stage,'Verifier',Overaccepting):
            with self.assertRaises(RuntimeError):score_stage.calibrate()
            result=load(Path(t)/'reports/reward_calibration.json')
            self.assertEqual(result['accuracy'],.975)
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['critical_failed'],['negation'])

    def test_extra_calibration_cases_cannot_hide_original_failures(self):
        import score_stage
        cases=[dict(id=str(i),question='q',gold=['g'],answer='a',expected=i>=3,critical=False)
               for i in range(40)]
        cases += [dict(id=f'extra{i}',suite='paraphrase16',question='q',gold=['g'],answer='a',
                       expected=True,critical=False) for i in range(100)]
        class AlwaysTrue:
            def judge(self,*args,**kwargs):return True,{}
        with tempfile.TemporaryDirectory() as t,patch.object(score_stage,'ROOT',Path(t)),\
             patch.object(score_stage,'load',return_value=cases),patch.object(score_stage,'Verifier',AlwaysTrue):
            with self.assertRaises(RuntimeError):score_stage.calibrate()
            r=load(Path(t)/'reports/reward_calibration.json')
            self.assertGreater(r['accuracy'],.95)
            self.assertEqual(r['suites']['original40']['accuracy'],.925)
            self.assertEqual(r['status'],'failed')

    def test_judge_prompt_changes_invalidate_cache(self):
        import score_stage
        calls=[]
        class Fake:
            def generate(self,messages,max_tokens):
                calls.append(messages)
                return [dict(text='{"reason":"equivalent", "decision":"correct"}',finish_reason='stop')]
        with tempfile.TemporaryDirectory() as t,patch.object(score_stage,'Engine',Fake):
            v=score_stage.Verifier(Path(t)/'cache.json')
            v.judge('q','a',['g'],force=True)
            with patch.object(score_stage,'JUDGE_SYSTEM',JUDGE_SYSTEM+'\nUpdated protocol.'):
                v.judge('q','a',['g'],force=True)
                v.judge('q','a',['g'],force=True)
            self.assertEqual(len(calls),2)

    def test_paraphrase_fixtures_cover_polarity_and_hedges(self):
        cases=load(ROOT/'data/reward_cases.json')
        self.assertEqual(len(cases),56)
        self.assertEqual(len({r['id'] for r in cases}),56)
        extra=[r for r in cases if r.get('suite')=='paraphrase16']
        self.assertEqual(len(extra),16)
        self.assertTrue(all(r['critical'] for r in extra))
        self.assertEqual(sum(r['expected'] for r in extra),8)

    def test_country_truth_table_polarities(self):
        from logical_reward import country_relation_decision as rule
        for prefix,negate in [('the same country',False),('different countries',True),('two different countries',True)]:
            for not_word in ['', 'not ']:
                target_same=not negate
                if not_word:target_same=not target_same
                q=f'Are Northport and Southport {not_word}in {prefix}?'
                for a,same in [('They are in the same country.',True),
                               ('Northport and Southport are both in one country.',True),
                               ('Southport and Northport are in different countries.',False),
                               ('They are not in the same country.',False),
                               ('They are not in different countries.',True)]:
                    for gold in ['yes','no']:
                        self.assertIs(rule(q,a,[gold]),(target_same==same)==(gold=='yes'),(q,a,gold))

    def test_country_rule_rejects_complex_or_unrelated_text(self):
        from logical_reward import country_relation_decision as rule
        q='Are Northport and Southport in different countries?'
        for a in ['Maybe they are in the same country.',
                  'They are in the same country, but I cannot determine it.',
                  'They are in the same country. Actually, they are not.',
                  'Otherport and Elseport are in the same country.',
                  'They were in the same country.',
                  'Are they in the same country?',
                  '"They are in the same country."',
                  'They are not both in the same country.',
                  'Both are not in the same country.',
                  'They are in different countries or the same country.']:
            self.assertIsNone(rule(q,a,['no']),a)
        for question in ['Were Northport and Southport in different countries?',
                         'Are Northport and Southport in different countries today?',
                         'Are Northport and Southport not both in the same country?',
                         'Are Northport and Southport and Eastport in different countries?',
                         'Are Northport and Southport in different cities?',
                         'Are Northport and Southport in different countries, or not?']:
            self.assertIsNone(rule(question,'They are in the same country.',['no']),question)
        self.assertIsNone(rule(q,'They are in the same country.',['yes','no']))
        self.assertIsNone(rule(q,'They are in the same country.',['France']))

    def test_suffix_ambiguity_is_not_positive_reward(self):
        from logical_reward import unsupported_name_suffix as ambiguous
        self.assertTrue(ambiguous('Who won?','Albert Brown Jr.',['Albert Brown']))
        self.assertTrue(ambiguous('What is the name?','Albert Brown',['Albert Brown III']))
        self.assertFalse(ambiguous('Who won?','Albert Brown Jr.',['Albert Brown','Albert Brown Jr.']))
        self.assertFalse(ambiguous('Who won?','Albert Brown Jr.',['Albert Brown Sr.']))

    def test_hybrid_calibration_uses_production_and_discloses_raw_failure(self):
        import score_stage
        cases=[dict(id='fixture',question='q',gold=['no'],answer='same',expected=True,critical=True)]
        class Hybrid:
            def judge(self,*args,force=False):
                return (False,dict(method='frozen_qwen_equivalence')) if force else (True,dict(method='explicit_country_relation_truth_table'))
        with tempfile.TemporaryDirectory() as t,patch.object(score_stage,'ROOT',Path(t)),\
             patch.object(score_stage,'load',return_value=cases),patch.object(score_stage,'Verifier',Hybrid):
            score_stage.calibrate()
            r=load(Path(t)/'reports/reward_calibration.json')
            self.assertEqual(r['status'],'passed')
            self.assertEqual(r['raw_model_diagnostic']['accuracy'],0)
            self.assertFalse(r['raw_model_diagnostic']['gate'])

    def test_hybrid_judge_calls_and_uncertain_suffix(self):
        import score_stage
        class NoModel:
            def generate(self,*args):raise AssertionError('Rule should handle this without a model call')
        with tempfile.TemporaryDirectory() as t,patch.object(score_stage,'Engine',NoModel):
            v=score_stage.Verifier(Path(t)/'cache.json')
            self.assertTrue(v.judge('Are Lakeport and Hillport in different countries?',
                                   'They are in the same country.',['no'])[0])
            self.assertIsNone(v.judge('Who won?','Albert Brown Jr.',['Albert Brown'])[0])

    def test_paired_summary_denominators(self):
        from summarize import paired_summary
        b=[dict(id=str(i),correct=c,answer='a',question='q',gold=['g']) for i,c in enumerate([True,False,None])]
        a=[dict(r,correct=c) for r,c in zip(b,[False,True,True])]
        s=paired_summary(b,a)
        self.assertEqual(s['paired_scorable'],2);self.assertEqual(s['wrong_to_correct'],1);self.assertEqual(s['correct_to_wrong'],1)
        self.assertIsNone(s['before']['accuracy']);self.assertEqual(s['before']['accuracy_lower'],1/3)
        self.assertEqual(s['before']['accuracy_upper'],2/3);self.assertFalse(s['all_pairs_scored'])

    def test_resume_hashes_and_report_excludes_weights(self):
        import run,tarfile
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);art=root/'artifacts';log=root/'logs';reports=root/'reports'
            for d in [art,log,reports]:d.mkdir()
            output=art/'result.json';calls=[]
            def fake(command,logfile):calls.append(command);save(output,{'done':True})
            with patch.multiple(run,ROOT=root,ART=art,LOG=log,REPORT=reports),patch.object(run,'invoke',fake):
                state={'run_fingerprint':'a'}
                run.stage('test',['test'],[output],state);run.stage('test',['test'],[output],state)
                self.assertEqual(len(calls),1)
                with self.assertRaises(ValueError):run.stage('test',['test'],[output],{'run_fingerprint':'b'})
                save(output,{'changed':True})
                with self.assertRaises(ValueError):run.stage('test',['test'],[output],state)
                (art/'adapter_model.safetensors').write_bytes(b'weights');(art/'optimizer.pt').write_bytes(b'optimizer')
                run.report()
            with tarfile.open(root/'clm_rl_grpo_pilot_v3_reports.tar.gz') as archive:
                    self.assertIn('artifacts/result.json',archive.getnames())
                    self.assertFalse(any(n.endswith(('.safetensors','.pt')) for n in archive.getnames()))

    def test_python_syntax(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))

    def test_full_pipeline_orchestration_and_second_run(self):
        # Exercise the real stage graph/hash checks with simulated model outputs.
        # No claim of a GPU/model test: this catches path/count/resume wiring errors.
        import run,summarize,sys,types
        from common import jsonl
        fake_lock=types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *a:None)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);art=root/'artifacts';log=root/'logs';reports=root/'reports'
            for d in [art,log,reports]:d.mkdir()
            calls=[]
            def rows(n):return [dict(id=f'{n}:{i}',question='question',split='dev' if not n else 'train') for i in range(2)]
            def fake_check():return 'simulated_spec'
            def fake_invoke(command,logfile):
                calls.append(command);cmd=[str(x) for x in command];logfile.write_text('simulated\n')
                if 'calibrate' in cmd:save(reports/'reward_calibration.json',dict(status='passed'));return
                if 'qwen_worker.py' in cmd:
                    out=Path(cmd[cmd.index('--output')+1])
                    if 'initial' in cmd:jsonl(out,[dict(id='0:0',text='initial',variant='original')])
                    else:jsonl(out,readl(Path(cmd[cmd.index('--input')+1])))
                    return
                if 'critic_worker.py' in cmd:
                    out=Path(cmd[cmd.index('--output')+1]);number=int(cmd[cmd.index('--round')+1]) if '--round' in cmd else 0
                    if 'init' in cmd:
                        save(out/'adapter_config.json',dict(step=0));return
                    if 'rollout' in cmd:
                        greedy='--greedy' in cmd
                        for rank in range(4):
                            rr=[dict(id=r['id'],candidate='control' if greedy else f'sample{rank}',text='answer')
                                for i,r in enumerate(rows(number)) if not greedy or i%4==rank]
                            jsonl(out/f'rank{rank}.jsonl',rr);save(out/f'rollout_resource_rank{rank}.json',{})
                        return
                    ckpt=art/f'checkpoints/step{number:03d}';save(ckpt/'adapter_config.json',dict(step=number))
                    for rank in range(4):save(out/f'update_rank{rank}.json',dict(optimizer_steps=1,effective_groups=2,adapter_after=str(number)))
                    return
                if 'score_stage.py' in cmd:
                    out=Path(cmd[cmd.index('--directory')+1]);number=int(cmd[cmd.index('--round')+1])
                    scored=[dict(r,question='q',gold=['gold'],answer='a',correct=(r['id'].endswith('0') or out.name=='dev_after')) for r in readl(out/'critique.jsonl')]
                    jsonl(out/'scored.jsonl',scored);save(out/'scores.json',dict(counts={},eligible_ids=[r['id'] for r in rows(number)],effective_ids=[r['id'] for r in rows(number)]))
                    return
                if 'summarize.py' in cmd:summarize.main();return
                self.fail(str(cmd))
            with patch.dict(sys.modules,fcntl=fake_lock),patch.dict(C,rounds=2,dev_questions=2,train_questions=4),\
                 patch.multiple(run,ROOT=root,ART=art,LOG=log,REPORT=reports),patch.object(summarize,'ROOT',root),\
                 patch.object(run,'check',fake_check),patch.object(run,'invoke',fake_invoke),patch.object(run,'round_rows',rows):
                run.run_pipeline()
                self.assertEqual(load(art/'status.json')['status'],'complete')
                s=load(reports/'summary.json');self.assertEqual(s['optimizer_steps'],2)
                self.assertEqual(s['evaluation']['wrong_to_correct'],1)
                n=len(calls);run.run_pipeline();self.assertEqual(len(calls),n)


if __name__=='__main__':unittest.main()
