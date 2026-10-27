import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from common import save,load,sha
from export_submission import export
from evaluate import plan,validate_rows,validate_sampling_paths
from prepare_workdir import prepare
from collect_server_assets import collect

class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.d=Path(self.temp.name)
        self.sample=self.d/'sample.csv'
        self.sample.write_text('id,answer\n0,\n1,\n',encoding='utf-8')
        self.r=self.d/'r.json';self.out=self.d/'out.csv'
        self.rows=[{'id':0,'question':'q0','rationale':'I think "A, B".\nSecond line: 中文'},
                   {'id':1,'question':'q1','rationale':'Cannot determine from the supplied documents.'}]
        save(self.r,self.rows)
    def test_full_text_roundtrip(self):
        info=export(self.r,self.sample,self.out)
        with self.out.open(encoding='utf-8',newline='') as f:got=list(csv.DictReader(f))
        self.assertEqual([r['answer'] for r in got],[r['rationale'] for r in self.rows]);self.assertFalse(info['preview_only'])
        self.assertEqual(info['csv_sha256'],sha(self.out))
    def test_no_overwrite(self):
        self.out.write_text('keep')
        with self.assertRaises(FileExistsError):export(self.r,self.sample,self.out)
        self.assertEqual(self.out.read_text(),'keep')
    def test_missing_rows_rejected(self):
        save(self.r,self.rows[:1])
        with self.assertRaises(ValueError):export(self.r,self.sample,self.out)
    def test_preview_explicit(self):
        save(self.r,self.rows[:1]);self.assertTrue(export(self.r,self.sample,self.out,True)['preview_only'])
    def test_reordered_ids_rejected(self):
        save(self.r,list(reversed(self.rows)))
        with self.assertRaises(ValueError):export(self.r,self.sample,self.out)
    def test_blank_answer_rejected(self):
        self.rows[0]['rationale']=' \n';save(self.r,self.rows)
        with self.assertRaises(ValueError):export(self.r,self.sample,self.out)
    def test_duplicate_template_rejected(self):
        self.sample.write_text('id,answer\n0,\n0,\n')
        with self.assertRaises(ValueError):export(self.r,self.sample,self.out)
    def test_question_identity(self):
        selected=[{'id':0,'question':'changed'},{'id':1,'question':'q1'}]
        with self.assertRaises(ValueError):validate_rows(self.r,selected)

class PlanTests(unittest.TestCase):
    def config(self):return {k:'/example/'+k for k in ['inference_python','qwen','llama','sft','rl_adapter','test']}
    def test_baseline_phase1_only(self):
        jobs=plan(self.config(),'qwen','full',Path('/out'),Path('/base'))
        self.assertEqual([j['phase'] for j in jobs],[1]);self.assertNotIn('--rl_adapter',jobs[0]['command'])
    def test_critic_only_rl_binding(self):
        jobs=plan(self.config(),'rl','full',Path('/out'),Path('/base'))
        self.assertEqual([j['phase'] for j in jobs],[2,3])
        self.assertIn('--rl_adapter',jobs[0]['command']);self.assertNotIn('--rl_adapter',jobs[1]['command'])
        self.assertEqual(jobs[0]['command'][jobs[0]['command'].index('--critique_model')+1],'/example/sft')
        self.assertIn('rl_eval',jobs[0]['cwd'])
    def test_frozen_uses_original_llama(self):
        jobs=plan(self.config(),'frozen','smoke',Path('/out'),Path('/base'))
        for j in jobs:
            self.assertEqual(j['command'][j['command'].index('--critique_model')+1],'/example/llama')
            self.assertEqual(j['command'][j['command'].index('--max_instances')+1],'5')
    def test_sft_matches_historical_engine_profile(self):
        jobs=plan(self.config(),'sft','full',Path('/out'),Path('/base'))
        self.assertTrue(all(Path(j['cwd']).name=='qwen' for j in jobs))
        self.assertTrue(all('--gpu_memory_utilization' not in j['command'] for j in jobs))

    def test_renamed_critic_cannot_silently_change_sampling(self):
        c=dict(qwen='/models/Qwen2.5-7B-Instruct',sft='/runs/models/CLM8B-SFT')
        with self.assertRaises(ValueError):validate_sampling_paths(c,'sft')
        c['sft']='/runs/models/Llama-3.1-8B-Instruct-CLM-SFT'
        validate_sampling_paths(c,'sft');validate_sampling_paths(c,'rl')
        c['qwen']='/home/llama/models/Qwen2.5-7B-Instruct'
        with self.assertRaises(ValueError):validate_sampling_paths(c,'qwen')

    def test_default_profiles_execute_historical_sampling_branches(self):
        import ast
        from types import SimpleNamespace
        from common import configuration
        config=configuration(ROOT/'configs/paths.example.json')
        for group in ('qwen','frozen','sft','rl'):validate_sampling_paths(config,group)
        for worker in ('qwen','rl_eval'):
            source=ROOT/'pipelines'/worker/'eval/inference_vllm.py'
            function=next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                          if isinstance(n,ast.FunctionDef) and n.name=='init_generator')
            ns={'torch':SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:4)),
                'LLM':lambda **kw:kw,'SamplingParams':lambda **kw:kw,'engine_options':lambda phase:{}}
            exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),ns)
            args=SimpleNamespace(seed=42,max_model_len=32768,gpu_memory_utilization=.85,max_tokens=5120,phase=2)
            _,params=ns['init_generator'](config['sft'],args)
            self.assertEqual((params['temperature'],params['top_p']),(.6,.9))

class PrepareTests(unittest.TestCase):
    def test_release_check_allows_locally_downloaded_ignored_models(self):
        from unittest.mock import patch
        import check_release
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);save(d/'provenance/source_inventory.json',{'copied_without_modification':[]})
            (d/'README.md').write_text('Local checkout')
            (d/'models').mkdir();(d/'models/model.safetensors').write_bytes(b'local untracked weights')
            with patch.object(check_release,'ROOT',d),patch.object(sys,'argv',['check_release.py']):
                check_release.main()

    def test_new_corpus_requires_explicit_acceptance_and_removes_stale_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);corpus=d/'corpus.json';save(corpus,[dict(instruction='q',input='weak',output='feedback') for _ in range(10000)])
            model=d/'model';model.mkdir();save(model/'config.json',{})
            c=dict(private_root=str(d),llama=str(model),training_python='/venv/bin/python',gpu_ids='0,1,2,3')
            with self.assertRaises(ValueError):prepare('sft',c,d/'blocked',corpus,False)
            self.assertFalse((d/'blocked').exists())
            out=prepare('sft',c,d/'ok',corpus,True)
            text=(out/'run.py').read_text(encoding='utf-8')
            self.assertIn('new_corpus_explicitly_accepted',text)
            self.assertIn("'target_generation_truncations_retained':None",text)
            self.assertEqual(load(out/'launch.json')['command'][0],'/venv/bin/python')
            for name,digest in load(out/'package_manifest.json')['files_sha256'].items():self.assertEqual(sha(out/name),digest)
            with self.assertRaises(FileExistsError):prepare('sft',c,out,corpus,True)
    def test_duplicate_gpu_ids_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):prepare('sft',{'gpu_ids':'0,0,1,2'},Path(tmp)/'new')
    def test_missing_assets_fail_before_archive_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):collect(Path(tmp))

if __name__=='__main__':unittest.main()
