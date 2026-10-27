"""Export the reviewed final adapter with LLaMA-Factory 0.9.3, on CPU."""
import argparse
import importlib.metadata
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from shared import ROOT, ADAPTER_SHA, BASE_CONFIG_SHA, SOURCE_SHA, load, save, sha, verify_package, verify_merge


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=Path.home() / 'RAG/models_verified/meta-llama/Llama-3.1-8B-Instruct')
    p.add_argument('--training', type=Path, default=Path.home() / 'RAG/clm_training_step5')
    p.add_argument('--output', type=Path, default=ROOT / 'models/Llama-3.1-8B-Instruct-CLM-original10k')
    a = p.parse_args()
    base, training, out = (x.expanduser().resolve() for x in (a.base, a.training, a.output))
    verify_package()
    for name, expected in [('llamafactory', '0.9.3'), ('transformers', '4.51.3'), ('peft', '0.15.2')]:
        if importlib.metadata.version(name) != expected:
            raise RuntimeError('Use ~/RAG/clm_training_env/bin/python; wrong version: ' + name)
    adapter = training / 'artifacts/full'
    report = load(training / 'reports/full_validation.json')
    pipeline = load(training / 'reports/pipeline.json')
    if report['status'] != 'complete' or report['global_step'] != 562 or not report['adapter_lora_b_updated']:
        raise ValueError('Training validation differs')
    if pipeline['status'] != 'complete' or pipeline['source_sha256'] != SOURCE_SHA:
        raise ValueError('Not the original step4 training run')
    if sha(adapter / 'adapter_model.safetensors') != ADAPTER_SHA or report['adapter_sha256'] != ADAPTER_SHA:
        raise ValueError('Not the reviewed FINAL adapter')
    if sha(base / 'config.json') != BASE_CONFIG_SHA:
        raise ValueError('Wrong base model config')
    cfg = load(adapter / 'adapter_config.json')
    if cfg['r'] != 16 or cfg['lora_alpha'] != 64 or Path(cfg['base_model_name_or_path']).resolve() != base:
        raise ValueError('Adapter/base mismatch')
    if out.exists():
        verify_merge(out)
        print('CLM_MERGE_ALREADY_COMPLETE', out, flush=True)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < 20 * 1024**3:
        raise RuntimeError('Need at least 20 GiB free disk space for the merged BF16 model')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    export = dict(model_name_or_path=str(base), adapter_name_or_path=str(adapter), template='llama3',
                  finetuning_type='lora', trust_remote_code=False, infer_dtype='bfloat16',
                  export_dir=str(out), export_size=4, export_device='cpu', export_legacy_format=False)
    save(ROOT / 'configs/merge.json', export)
    status = dict(status='running', base=str(base), adapter=str(adapter), output=str(out),
                  adapter_sha256=ADAPTER_SHA, source_sha256=SOURCE_SHA,
                  started_utc=datetime.now(timezone.utc).isoformat(), config=export)
    save(ROOT / 'reports/merge.json', status)
    try:
        from llamafactory.train.tuner import export_model
        print('Exporting BF16 CLM on CPU; allow memory for the 8B base model.', flush=True)
        export_model(export)
        import torch
        from safetensors import safe_open
        from transformers import AutoTokenizer
        idx = load(out / 'model.safetensors.index.json')['weight_map']
        base_idx = load(base / 'model.safetensors.index.json')['weight_map']
        if set(idx) != set(base_idx) or any('lora_' in k for k in idx):
            raise ValueError('Merged tensor keys differ from base model')
        numel = 0
        for shard in sorted(set(idx.values())):
            print('Validating', shard, flush=True)
            with safe_open(out / shard, framework='pt', device='cpu') as f:
                if set(f.keys()) != {k for k, v in idx.items() if v == shard}:
                    raise ValueError('Shard index mismatch')
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    if tensor.dtype != torch.bfloat16 or not torch.isfinite(tensor).all():
                        raise ValueError('Invalid merged tensor: ' + key)
                    numel += tensor.numel()
        # Confirm learned changes survived the BF16 export at multiple layers.
        changed = {}
        for layer in (0, 15, 31):
            key = f'model.layers.{layer}.self_attn.q_proj.weight'
            with safe_open(base / base_idx[key], framework='pt', device='cpu') as b, safe_open(out / idx[key], framework='pt', device='cpu') as m:
                changed[key] = int(torch.count_nonzero(b.get_tensor(key).to(torch.bfloat16) != m.get_tensor(key)).item())
        if not any(changed.values()):
            raise ValueError('Sampled weights unchanged from base')
        tok = AutoTokenizer.from_pretrained(out, local_files_only=True)
        tok.apply_chat_template([{'role': 'user', 'content': 'Check.'}], tokenize=True, add_generation_prompt=True)
        if len(tok) != load(out / 'config.json')['vocab_size']:
            raise ValueError('Tokenizer/model vocabulary mismatch')
        status.update(status='complete', tensor_numel=numel, changed_weight_elements=changed,
                      files_sha256={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
                      completed_utc=datetime.now(timezone.utc).isoformat())
        save(out / 'merge_manifest.json', status)
        print('CLM_MERGE_COMPLETE', out, flush=True)
    except BaseException as exc:
        status.update(status='failed', error=repr(exc))
        raise
    finally:
        save(ROOT / 'reports/merge.json', status)


if __name__ == '__main__':
    main()
