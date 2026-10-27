"""Restore the SFT CLM using LLaMA-Factory BF16 export, as in the original experiment."""
import argparse,os,shutil
from pathlib import Path
from common import configuration,load,save,sha,abspath,verify_model
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/paths.json')
    p.add_argument('--adapter',type=Path);p.add_argument('--corpus',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();c=configuration(a.config)
    base=Path(c['llama']);adapter=abspath(a.adapter or c.get('sft_adapter') or Path(c['private_root'])/'weights/sft_adapter')
    out=abspath(a.output or c['sft']);work=out.with_name(out.name+'.building')
    corpus=abspath(a.corpus or c.get('sft_corpus') or Path(c['private_root'])/'sft/cft_train_10k.json')
    if out.exists() or work.exists():raise FileExistsError('Use a new output path; preserve previous merge files')
    cfg=load(adapter/'adapter_config.json')
    if cfg['r']!=16 or cfg['lora_alpha']!=64:raise ValueError('Expected SFT adapter rank16, alpha64 (not RL rank8)')
    verify_model(base);source_hash=sha(corpus);adapter_hash=sha(adapter/'adapter_model.safetensors')
    out.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(out.parent).free<20*1024**3:raise ValueError('Need at least 20 GiB free disk')
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    from importlib.metadata import version
    for name,want in [('llamafactory','0.9.3'),('transformers','4.51.3'),('peft','0.15.2')]:
        if version(name)!=want:raise ValueError('Run with the pinned training environment: '+name)
    from llamafactory.train.tuner import export_model
    from safetensors import safe_open
    import torch
    export_model(dict(model_name_or_path=str(base),adapter_name_or_path=str(adapter),template='llama3',
        finetuning_type='lora',trust_remote_code=False,infer_dtype='bfloat16',export_dir=str(work),
        export_size=4,export_device='cpu',export_legacy_format=False))
    idx=load(work/'model.safetensors.index.json')['weight_map'];base_idx=load(base/'model.safetensors.index.json')['weight_map']
    if set(idx)!=set(base_idx) or any('lora_' in k for k in idx):raise ValueError('Merged tensor index mismatch')
    for shard in set(idx.values()):
        with safe_open(work/shard,framework='pt',device='cpu') as f:
            for key in f.keys():
                x=f.get_tensor(key)
                if x.dtype!=torch.bfloat16 or not torch.isfinite(x).all():raise ValueError('Invalid merged tensor '+key)
    changed={}
    for layer in [0,15,31]:
        key=f'model.layers.{layer}.self_attn.q_proj.weight'
        with safe_open(base/base_idx[key],framework='pt',device='cpu') as b,safe_open(work/idx[key],framework='pt',device='cpu') as m:
            changed[key]=int(torch.count_nonzero(b.get_tensor(key).to(torch.bfloat16)!=m.get_tensor(key)))
    if not any(changed.values()):raise ValueError('SFT merge did not change sampled base weights')
    save(work/'merge_manifest.json',{'status':'complete','adapter_sha256':adapter_hash,'source_sha256':source_hash,
        'base_config_sha256':sha(base/'config.json'),'changed_weight_elements':changed,
        'files_sha256':{f.name:sha(f) for f in sorted(work.iterdir()) if f.is_file()},
        'export_method':'LLaMA-Factory 0.9.3 CPU BF16; portable re-export, new artifact manifest'})
    work.rename(out);print('SFT_MERGE_COMPLETE',out)
if __name__=='__main__':main()
