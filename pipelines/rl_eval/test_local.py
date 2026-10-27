"""Local CPU protocol tests; model/kernel integration still requires the server smoke."""
import ast
import csv
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from shared import ROOT,export_csv,rows_valid,save,load,sha,verify_files
import run
sys.path.insert(0,str(ROOT/'eval'))
import rl_runtime as rt

class Registry:
    def __init__(self,ids=()):self.ids=set(ids);self.added=[]
    def list_loras(self):return set(self.ids)
    def add_lora(self,request):self.added.append(request);self.ids.add(request.lora_int_id);return True

class Tests(unittest.TestCase):
    def test_csv_exact_quotes_newlines_order(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);sample=p/'sample.csv';sample.write_text('id,answer\n0,\n1,\n',encoding='utf-8')
            rows=[dict(id='0',rationale='A, "B"\nNext line'),dict(id='1',rationale='unknown')]
            export_csv(rows,sample,p/'out.csv')
            with (p/'out.csv').open(encoding='utf-8',newline='') as f:out=list(csv.DictReader(f))
            self.assertEqual([r['answer'] for r in out],[r['rationale'] for r in rows])
            with self.assertRaises(ValueError):export_csv(rows[::-1],sample,p/'wrong.csv')
            with self.assertRaises(ValueError):export_csv(rows[:1],sample,p/'short.csv')
            export_csv(rows[:1],sample,p/'preview.csv',True)

    def test_rows_validate_identity_not_only_count(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'data.json';save(p,[dict(id='1',question='q',rationale='a')])
            with self.assertRaises(ValueError):rows_valid(p,[dict(id='0',question='q')])
            with self.assertRaises(ValueError):rows_valid(p,[dict(id='1',question='other')])

    def test_manifest_containment_and_tampering(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'f').write_text('old');h=sha(p/'f');verify_files(p,{'f':h})
            (p/'f').write_text('new')
            with self.assertRaises(ValueError):verify_files(p,{'f':h})
            with self.assertRaises(ValueError):verify_files(p,{'../escape':h})

    def test_critic_only_adapter_command(self):
        p={k:Path('/tmp')/k for k in ['inference_python','test','qwen','sft','baseline','adapter']}
        p['inference_python']=Path('/env/bin/python')
        for phase in [2,3]:
            cmd=run.build_command(p,Path('/out'),phase,'full','rl')
            self.assertEqual('--rl_adapter' in cmd,phase==2)
            self.assertNotIn('--max_instances',cmd)
            self.assertEqual(cmd[0],str(Path('/env/bin/python')))
        self.assertNotIn('--rl_adapter',run.build_command(p,Path('/out'),2,'smoke','sft'))
        cmd=run.build_command(p,Path('/out'),2,'smoke','rl',True)
        self.assertIn('--probe_only',cmd);self.assertEqual(cmd[cmd.index('--max_instances')+1],'5')

    def test_v1_missing_metadata_uses_registry_but_rejects_conflict(self):
        req=types.SimpleNamespace(lora_int_id=16,lora_path='/tmp/adapter')
        good=types.SimpleNamespace(lora_request=req);bad=types.SimpleNamespace(lora_request=None)
        model=types.SimpleNamespace(llm_engine=Registry([16]))
        rt.check_binding([good],req,model);rt.check_binding([bad],None)
        evidence=rt.check_binding([bad],req,model)
        self.assertEqual(evidence['returned_metadata_missing_count'],1)
        self.assertEqual(evidence['returned_metadata_matched_count'],0)
        self.assertEqual(evidence['submitted_lora_request_count'],1)
        conflict=types.SimpleNamespace(lora_request=types.SimpleNamespace(lora_int_id=17,lora_path='/tmp/adapter'))
        with self.assertRaises(ValueError):rt.check_binding([conflict],req,model)
        conflict.lora_request.lora_int_id=16;conflict.lora_request.lora_path='/tmp/other'
        with self.assertRaises(ValueError):rt.check_binding([conflict],req,model)
        with self.assertRaises(ValueError):rt.check_binding([bad],req)
        with self.assertRaises(ValueError):rt.check_binding([good],req,types.SimpleNamespace(llm_engine=Registry()))
        with self.assertRaises(ValueError):rt.check_binding([good],None)
        with self.assertRaises(ValueError):rt.request_for(3,'/tmp/adapter')

    def test_explicit_registration_and_failed_registration(self):
        req=types.SimpleNamespace(lora_int_id=16,lora_path='/tmp/adapter')
        reg=Registry();model=types.SimpleNamespace(llm_engine=reg)
        self.assertEqual(rt.ensure_adapter_loaded(model,req),[16])
        self.assertEqual(reg.added,[req])
        rt.ensure_adapter_loaded(model,req);self.assertEqual(len(reg.added),1)
        reg.ids.clear();reg.add_lora=lambda request:False
        with self.assertRaises(ValueError):rt.ensure_adapter_loaded(model,req)

    def test_probe_compares_only_identical_histories(self):
        a=[dict(token_ids=[1,2,3],logprobs=[{'1':-1},{'2':-1},{'3':-1}])]
        b=[dict(token_ids=[1,9,3],logprobs=[{'1':-1.01},{'2':-1.1},{'3':-100}])]
        result=rt.compare_same_prefix(a,b)
        self.assertAlmostEqual(result['max_logprob_delta'],.1)
        self.assertEqual(result['compared_values'],2)

    def test_optional_probe_records_inconclusive_without_blocking(self):
        req=types.SimpleNamespace(lora_int_id=16,lora_path='/tmp/adapter')
        class Fake:
            def __init__(self,effect):self.effect=effect;self.llm_engine=Registry()
            def chat(self,prompts,params,lora_request=None,**kw):
                lp=-1.+(self.effect if lora_request else 0)
                comp=types.SimpleNamespace(text='a',token_ids=[1],finish_reason='length',logprobs=[{1:types.SimpleNamespace(logprob=lp)}])
                # Match V1 0.10.2: returned metadata is None even when the adapter is used.
                return [types.SimpleNamespace(lora_request=None,outputs=[comp]) for _ in prompts]
        vllm=types.SimpleNamespace(SamplingParams=lambda **kw:kw)
        with tempfile.TemporaryDirectory() as t,patch.dict(sys.modules,vllm=vllm):
            out=Path(t)/'probe.json'
            rt.probe(Fake(0),[[]],req,out)
            self.assertEqual(load(out)['status'],'inconclusive')
            rt.probe(Fake(.01),[[]],req,out);self.assertEqual(load(out)['status'],'passed')
            self.assertEqual(load(out)['binding']['adapted']['returned_metadata_missing_count'],1)

    def test_runtime_records_submitted_vs_echoed_counts(self):
        req=types.SimpleNamespace(lora_int_id=16,lora_path='/tmp/adapter')
        model=types.SimpleNamespace(llm_engine=Registry([16]))
        args=types.SimpleNamespace(phase=2,critique_model='sft',model_name_or_path='qwen',gpu_num=4,max_tokens=5120,max_model_len=32768)
        outputs=[types.SimpleNamespace(lora_request=None,outputs=[types.SimpleNamespace(finish_reason='stop',token_ids=[1,2])])]
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'runtime.json';rt.save_runtime(p,outputs,args,req,model);r=load(p)
            self.assertEqual(r['submitted_lora_request_count'],1)
            self.assertEqual(r['returned_metadata_matched_count'],0)
            self.assertEqual(r['registered_adapter_ids'],[16])

    def test_python310_syntax_and_original_prompts_preserved(self):
        for p in ROOT.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,10))
        original=load(ROOT/'reference/author_source.json')['files_sha256']
        for name,digest in original.items():
            if name!='eval/inference_vllm.py':self.assertEqual(sha(ROOT/name),digest,name)
        self.assertNotIn('<final_answer>',(ROOT/'eval/inference_vllm.py').read_text(encoding='utf-8'))

    def test_mocked_direct_full_without_smoke_or_probe_then_resume(self):
        lock=types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *a:None)
        with tempfile.TemporaryDirectory() as t,patch.dict(sys.modules,fcntl=lock),patch.object(run,'ROOT',Path(t)):
            root=Path(t);p={k:root/k for k in ['inference_python','test','qwen','sft','adapter','baseline']}
            test=[dict(id=str(i),question=f'q{i}') for i in range(7)]
            save(p['baseline'],[dict(r,rationale='initial') for r in test]);p['sample']=root/'sample.csv'
            p['sample'].write_text('id,answer\n'+''.join(f'{i},\n' for i in range(7)),encoding='utf-8')
            save(root/'source_manifest.json',{});calls=[]
            def invoke(cmd,log):
                calls.append(cmd);out=Path(cmd[cmd.index('--critique_model_path')+1]);phase=int(cmd[cmd.index('--phase')+1])
                self.assertNotIn('--probe_only',cmd)
                selected=test[:5] if '--max_instances' in cmd else test
                name='result_critic_rationale_cot.json' if phase==2 else 'result_refine_cot.json'
                save(out/'2WikiMultiHopQA'/name,[dict(r,rationale='A, "B"\nFull answer') for r in selected])
                save(out/f'phase{phase}_runtime.json',dict(rows=len(selected),submitted_lora_request_count=len(selected) if phase==2 else 0,adapter_path=str(p['adapter'].resolve()) if phase==2 else None))
            with patch.object(run,'paths',return_value=p),patch.object(run,'verify_inputs',return_value=({'training_run_fingerprint':'f'},test)),\
                 patch.object(run,'check_environment',return_value={}),patch.object(run,'invoke',side_effect=invoke),\
                 patch.object(run,'report'),patch.dict(run.EXPECTED,baseline=sha(p['baseline'])):
                self.assertFalse((root/'artifacts/rl_smoke').exists())
                run.evaluate('full','rl');self.assertEqual(len(calls),2)
                self.assertFalse((root/'artifacts/rl_smoke').exists())
                run.evaluate('full','rl');self.assertEqual(len(calls),2)
                run.evaluate('smoke','rl');self.assertEqual(len(calls),4)
                run.evaluate('smoke','rl');self.assertEqual(len(calls),4)
                m=load(root/'artifacts/rl_full/run_manifest.json');self.assertEqual(m['status'],'complete')
                self.assertEqual(m['completed_phases'],[2,3])
                # The complete run must remain intact after a changed-source attempt.
                before=(root/'artifacts/rl_full/run_manifest.json').read_bytes()
                save(root/'source_manifest.json',{'modified':True})
                with self.assertRaises(ValueError):run.evaluate('full','rl')
                self.assertEqual((root/'artifacts/rl_full/run_manifest.json').read_bytes(),before)

if __name__=='__main__':unittest.main()
