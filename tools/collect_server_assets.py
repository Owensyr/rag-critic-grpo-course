"""Standalone stdlib-only collector. Run on the ORIGINAL server, never upload its tar to GitHub."""
import argparse,hashlib,io,json,os,tarfile,time
from pathlib import Path

REQUIRED={
 'course/test.json':'author_alignrag_course/course_data/2WikiMultiHopQA/test.json',
 'course/sample_submission.csv':'author_alignrag_course/course_data/2WikiMultiHopQA/sample_submission.csv',
 'course/qwen_initial.json':'author_alignrag_course/course/artifacts/full/qwen/2WikiMultiHopQA/result_rationale_cot.json',
 'sft/cft_train_10k.json':'clm_training_step4_data/artifacts/full/cft_train_10k.json',
 'sft/train_10k.jsonl':'clm_training_step4_data/data/train_10k.jsonl',
 'sft/build_report.json':'clm_training_step4_data/data/build_report.json',
 'sft/selection_preview70.json':'clm_training_step4_data/data/selection_preview70.json',
 'grpo/questions.json':'clm_rl_grpo_pilot_v3/data/questions.json',
 'grpo/gold.json':'clm_rl_grpo_pilot_v3/data/gold.json',
 'grpo/selection_audit.json':'clm_rl_grpo_pilot_v3/data/selection_audit.json',
 'grpo/reward_cases.json':'clm_rl_grpo_pilot_v3/data/reward_cases.json',
 'weights/sft_adapter/adapter_model.safetensors':'clm_training_step5/artifacts/full/adapter_model.safetensors',
 'weights/sft_adapter/adapter_config.json':'clm_training_step5/artifacts/full/adapter_config.json',
 'weights/rl_adapter/adapter_model.safetensors':'clm_rl_grpo_pilot_v3/artifacts/checkpoints/step016/adapter_model.safetensors',
 'weights/rl_adapter/adapter_config.json':'clm_rl_grpo_pilot_v3/artifacts/checkpoints/step016/adapter_config.json',
 'evidence/sft_merge_manifest.json':'clm_step6_2wiki/models/Llama-3.1-8B-Instruct-CLM-original10k/merge_manifest.json',
 'evidence/sft_validation.json':'clm_training_step5/reports/full_validation.json',
 'evidence/rl_summary.json':'clm_rl_grpo_pilot_v3/reports/summary.json',
 'evidence/rl_training_spec.json':'clm_rl_grpo_pilot_v3/artifacts/spec.json',
 'evidence/rl_training_status.json':'clm_rl_grpo_pilot_v3/artifacts/status.json',
 'evidence/rl_full_manifest.json':'clm_rl_step7_2wiki_v3/artifacts/rl_full/run_manifest.json'}
PACKAGES=['author_alignrag_course/course','frozen_clm_full_answer','clm_training_step4_data',
          'clm_training_step5','clm_step6_2wiki','clm_rl_grpo_pilot_v3','clm_rl_step7_2wiki_v3']
PINS={'weights/sft_adapter/adapter_model.safetensors':'f0c8979757af7dffd41635eb61387902a7d367f73730c9bc31855cc2b8388343',
      'weights/rl_adapter/adapter_model.safetensors':'89d30d081e628a4c34d571f76a9bcdb38a83366150bda95d554732004d680a18',
      'sft/cft_train_10k.json':'c500875fc40020ea3fe302e8797deb4512a46c92e54ab5150d52e16f9928de72'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def collect(root,include_merged=False,include_evidence=False,include_optimizer=False):
    files={k:root/v for k,v in REQUIRED.items()};missing=[str(p) for p in files.values() if not p.is_file()]
    if missing:raise FileNotFoundError('Required original-run files missing:\n'+'\n'.join(missing))
    for package in PACKAGES:
        directory=root/package
        for p in directory.rglob('*'):
            rel=p.relative_to(directory)
            if not p.is_file() or any(x in rel.parts for x in ['__pycache__','.git','artifacts','logs','reports','data','models','prepared','tokenization_work']):continue
            if p.suffix in ['.py','.sh','.yaml','.json','.txt','.md','.yml']:files['server_code/'+package+'/'+rel.as_posix()]=p
    for src,dest in [('clm_training_step4_data/artifacts/full','evidence/synthesis'),('clm_training_step5/reports','evidence/sft_reports'),('clm_rl_grpo_pilot_v3/reports','evidence/grpo_reports')]:
        for p in (root/src).glob('*.json'):
            if p.stat().st_size<4*1024**2:files[dest+'/'+p.name]=p
    if include_merged:
        directory=root/'clm_step6_2wiki/models/Llama-3.1-8B-Instruct-CLM-original10k'
        for p in directory.iterdir():
            if p.is_file():files['weights/sft_merged/'+p.name]=p
    if include_optimizer:
        p=root/'clm_rl_grpo_pilot_v3/artifacts/checkpoints/step016/optimizer.pt'
        if not p.is_file():raise FileNotFoundError(p)
        files['weights/rl_adapter/optimizer.pt']=p
    if include_evidence:
        for package in PACKAGES:
            for sub in ['artifacts/full','artifacts/rl_full','reports','logs']:
                for p in (root/package/sub).rglob('*'):
                    if p.is_file() and p.suffix in ['.json','.jsonl','.csv','.log','.txt']:
                        files['run_evidence/'+package+'/'+p.relative_to(root/package).as_posix()]=p
    for dest,p in files.items():
        if not p.resolve().is_relative_to(root.resolve()):raise ValueError('Source symlink escapes RAG directory: '+str(p))
    return files
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rag-root',type=Path,default=Path.home()/'RAG')
    p.add_argument('--output',type=Path,default=Path.home()/'RAG/course_private_assets.tar.gz')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--include-merged-sft',action='store_true')
    p.add_argument('--include-evidence',action='store_true');p.add_argument('--include-optimizer',action='store_true');a=p.parse_args()
    root=a.rag_root.expanduser().resolve();out=a.output.expanduser().resolve()
    files=collect(root,a.include_merged_sft,a.include_evidence,a.include_optimizer)
    total=sum(f.stat().st_size for f in files.values())
    print(json.dumps({'files':len(files),'uncompressed_bytes':total,'GiB':round(total/2**30,3),
       'includes_base_Qwen_or_Llama':False,'includes_merged_SFT':a.include_merged_sft,'output':str(out)},indent=2),flush=True)
    if a.dry_run:
        for name in REQUIRED:print('REQUIRED',name,'<-',files[name])
        return
    if out.exists() or out.with_suffix('.tmp').exists():raise FileExistsError('Output exists; use a new --output filename')
    inventory={}
    for i,(name,f) in enumerate(files.items(),1):
        digest=sha(f)
        if name in PINS and digest!=PINS[name]:raise ValueError('Historical asset hash differs: '+name)
        inventory[name]={'sha256':digest,'bytes':f.stat().st_size,'source':f.relative_to(root).as_posix()}
        if i%20==0:print('HASHED',i,'/',len(files),flush=True)
    out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp')
    with tarfile.open(tmp,'w:gz') as t:
        for name,f in files.items():t.add(f,arcname='private/'+name,recursive=False)
        report=json.dumps({'format_version':1,'created_unix':time.time(),'files':inventory},ensure_ascii=False,indent=2).encode()
        m=tarfile.TarInfo('private/assets_manifest.json');m.size=len(report);t.addfile(m,io.BytesIO(report))
    tmp.replace(out);digest=sha(out)
    out.with_name(out.name+'.sha256').write_text(digest+'  '+out.name+'\n',encoding='ascii')
    print('PRIVATE_ASSETS_READY',out,'SHA256',digest,flush=True)
if __name__=='__main__':main()
