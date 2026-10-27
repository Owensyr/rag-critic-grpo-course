"""Initialize NEW portable workdirs; keep historical sources and runs untouched."""
import argparse
from pathlib import Path
import re
import shutil
import sys
from common import ROOT,configuration,load,save,sha,abspath
from data_utils import clean_docs,question_key

def validate_new_selection(kind, data):
    if kind=='synthesis':
        report=load(data/'build_report.json')
        if report.get('status')!='complete' or report.get('rows')!=10000:
            raise ValueError('Expected a completed five-dataset 10K selection')
        if report['train_sha256']!=sha(data/'train_10k.jsonl'):
            raise ValueError('SFT selection changed after its audit')
        load(data/'selection_preview70.json')
    else:
        rows=load(data/'questions.json');gold=load(data/'gold.json');audit=load(data/'selection_audit.json')
        config=load(ROOT/'pipelines/grpo/config.json')
        if len(rows)!=config['train_questions']+config['dev_questions']:
            raise ValueError('Expected 128 train plus 128 dev questions')
        if {str(r['id']) for r in rows}!={str(r['id']) for r in gold} or len(gold)!=len(rows):
            raise ValueError('Question/gold ID mismatch')
        if len({r['id'] for r in rows})!=len(rows) or len({question_key(r['question']) for r in rows})!=len(rows):
            raise ValueError('Duplicate GRPO questions')
        for split in ('train','dev'):
            if sum(r['split']==split for r in rows)!=config[split+'_questions']:
                raise ValueError('GRPO split size differs')
        for r in rows:
            if set(r)!={'id','source_index','question','ctxs','split'}:
                raise ValueError('Unexpected prompt fields; gold must remain separate')
            clean_docs(r)
            if any(set(d)!={'title','text','score'} for d in r['ctxs']):raise ValueError('Unexpected context metadata')
        if any(not isinstance(r.get('answers'),list) or not r['answers'] or
               not all(isinstance(x,str) and x.strip() for x in r['answers']) for r in gold):
            raise ValueError('Invalid gold labels')
        for name in ('questions.json','gold.json'):
            if audit['files_sha256'][name]!=sha(data/name):raise ValueError('GRPO selection changed after audit')
        if any(audit[k]!=0 for k in ('overlap_sft','overlap_previous_rl','overlap_evaluation','overlap_train_dev')):
            raise ValueError('Training/evaluation overlap detected')

def reseal(folder, name):
    p=folder/name;m=load(p)
    m['files_sha256']={n:sha(folder/n) for n in m['files_sha256']}
    m['distribution_note']='New portable run. Configuration/path or input-pin changes are in portability_changes.json; not the original run fingerprint.'
    save(p,m)
def prepare(kind,c,out,corpus=None,accept_new=False,data_dir=None,accept_new_data=False):
    gpu_ids=c['gpu_ids'].split(',')
    if len(gpu_ids)!=4 or len(set(gpu_ids))!=4 or any(not x.isdigit() for x in gpu_ids):
        raise ValueError('Specify four distinct numeric GPU IDs')
    if out.exists():raise FileExistsError('New run directory required: '+str(out))
    src=ROOT/'pipelines'/kind;private=Path(c.get('private_root',ROOT/'private'));changes={}
    data=abspath(data_dir) if data_dir else private/('sft' if kind=='synthesis' else 'grpo')
    if kind=='sft':
        corpus=abspath(corpus or c.get('sft_corpus') or private/'sft/cft_train_10k.json')
        digest=sha(corpus)
        expected='c500875fc40020ea3fe302e8797deb4512a46c92e54ab5150d52e16f9928de72'
        if digest!=expected and not accept_new:raise ValueError('Different corpus: pass --accept-new-corpus for an explicitly new experiment')
        rows=load(corpus)
        if len(rows)!=10000 or any(not all(isinstance(r.get(k),str) and r[k].strip() for k in ['instruction','input','output']) for r in rows):raise ValueError('Expected 10000 Alpaca CFT records')
        sha(Path(c['llama'])/'config.json')
    elif kind=='grpo':
        if accept_new_data:validate_new_selection(kind,data)
        else:
            for n in ['questions.json','gold.json','selection_audit.json']:
                expected=load(src/'source_manifest.json')['files_sha256']['data/'+n]
                if sha(data/n)!=expected:raise ValueError('New data requires --accept-new-data: '+n)
        merge=load(Path(c['sft'])/'merge_manifest.json')
        if merge['status']!='complete' or not merge['changed_weight_elements']:raise ValueError('SFT merge is incomplete')
    elif kind=='synthesis':
        if accept_new_data:validate_new_selection(kind,data)
        else:
            for n in ['train_10k.jsonl','selection_preview70.json','build_report.json']:
                if sha(data/n)!=load(src/'source_manifest.json')['files_sha256']['data/'+n]:raise ValueError('New data requires --accept-new-data: '+n)
    out.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(src,out,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    if kind=='sft':
        target=out/'input_source/artifacts/full/cft_train_10k.json';target.parent.mkdir(parents=True)
        shutil.copyfile(corpus,target)
        text=(out/'run.py').read_text(encoding='utf-8')
        text,n=re.subn(r"SOURCE_HASH='[0-9a-f]{64}'",f"SOURCE_HASH='{digest}'",text,count=1)
        if n!=1:raise ValueError('Unexpected SFT source structure')
        model_hash=sha(Path(c['llama'])/'config.json')
        text,n=re.subn(r"MODEL_HASH='[0-9a-f]{64}'",f"MODEL_HASH='{model_hash}'",text,count=1)
        if n!=1:raise ValueError('Unexpected SFT model hash anchor')
        changes['base_config_sha256']=model_hash
        text=text.replace("CUDA_VISIBLE_DEVICES='0,1,2,3'",f"CUDA_VISIBLE_DEVICES='{c['gpu_ids']}'")
        if digest!=expected:
            text=text.replace("'original_step4_unfiltered'","'new_corpus_explicitly_accepted'")
            text=text.replace("'source_generation_truncations_retained':256","'source_generation_truncations_retained':None")
            text=text.replace("'target_generation_truncations_retained':137","'target_generation_truncations_retained':None")
        (out/'run.py').write_text(text,encoding='utf-8')
        changes['corpus_sha256']=digest;changes['historical_corpus']=digest==expected
        command=[c['training_python'],'-u','run.py','all','--source',str(out/'input_source'),'--model',c['llama']]
        reseal(out,'package_manifest.json')
    elif kind=='synthesis':
        (out/'data').mkdir(exist_ok=True)
        for n in ['train_10k.jsonl','selection_preview70.json','build_report.json']:shutil.copyfile(data/n,out/'data'/n)
        text=(out/'run.py').read_text(encoding='utf-8')
        text=text.replace("'CUDA_VISIBLE_DEVICES':'0' if role=='weak' else '0,1,2,3'",
            f"'CUDA_VISIBLE_DEVICES':'{gpu_ids[0]}' if role=='weak' else '{c['gpu_ids']}'")
        (out/'run.py').write_text(text,encoding='utf-8')
        command=[c['inference_python'],'-u','run.py','all','--weak',c['weak'],'--strong',c['llama']]
        reseal(out,'source_manifest.json')
    else:
        (out/'data').mkdir(exist_ok=True)
        for n in ['questions.json','gold.json','selection_audit.json']:shutil.copyfile(data/n,out/'data'/n)
        config=load(out/'config.json');old=dict(config)
        config.update(inference_python=c['inference_python'],training_python=c['training_python'],qwen_model=c['qwen'],clm_model=c['sft'])
        save(out/'config.json',config);changes['config_paths']={k:{'old':old[k],'new':config[k]} for k in config if config[k]!=old[k]}
        text=(out/'model_provenance.py').read_text(encoding='utf-8')
        for key,value in [('ADAPTER_SHA',merge['adapter_sha256']),('SOURCE_SHA',merge['source_sha256'])]:
            if not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('Invalid provenance hash')
            text,n=re.subn(key+r" = '[0-9a-f]{64}'",f"{key} = '{value}'",text,count=1)
            if n!=1:raise ValueError('Unexpected provenance source structure')
        (out/'model_provenance.py').write_text(text,encoding='utf-8')
        changes['sft_adapter_sha256']=merge['adapter_sha256'];changes['sft_source_sha256']=merge['source_sha256']
        command=[c['training_python'],'-u','run.py','all'];reseal(out,'source_manifest.json')
    changes['gpu_ids']=c['gpu_ids']
    if kind in ('synthesis','grpo'):
        changes['new_selection_explicitly_accepted']=accept_new_data
        changes['input_data_directory']=str(data)
        changes['input_files_sha256']={p.name:sha(p) for p in sorted((out/'data').iterdir()) if p.is_file()}
    save(out/'portability_changes.json',{'historical_source_folder':src.relative_to(ROOT).as_posix(),'changes':changes,'algorithm_modified':False})
    save(out/'launch.json',{'cwd':str(out),'command':command,'gpu_ids':c['gpu_ids']})
    (out/'logs').mkdir(exist_ok=True)
    print('WORKDIR_READY',out);print('LAUNCH: python tools/launch.py --workdir',out)
    return out
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',choices=['synthesis','sft','grpo'])
    p.add_argument('--config',default='configs/paths.json');p.add_argument('--output',required=True,type=Path)
    p.add_argument('--corpus',type=Path);p.add_argument('--accept-new-corpus',action='store_true')
    p.add_argument('--data-dir',type=Path,help='Fresh selection from tools/prepare_data.py')
    p.add_argument('--accept-new-data',action='store_true',help='Explicitly accept new synthesis/GRPO selection and record its hashes')
    a=p.parse_args()
    if a.kind=='sft' and (a.data_dir or a.accept_new_data):p.error('SFT uses --corpus and --accept-new-corpus')
    if a.kind!='sft' and (a.corpus or a.accept_new_corpus):p.error('--corpus applies only to SFT')
    prepare(a.kind,configuration(a.config),abspath(a.output),a.corpus,a.accept_new_corpus,a.data_dir,a.accept_new_data)
if __name__=='__main__':main()
