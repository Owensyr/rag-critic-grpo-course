"""Portable paths and integrity helpers; preserve venv executable symlinks."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p, obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def abspath(p):
    p=Path(os.path.expanduser(str(p)))
    return Path(os.path.abspath(p if p.is_absolute() else ROOT/p))
def configuration(p):
    c=load(p)
    for k in ['inference_python','training_python','qwen','llama','weak','sft','rl_adapter','test','sample','baseline','private_root',
              'data_root','sft_corpus','sft_adapter']:
        if k in c:c[k]=str(abspath(c[k]))
    return c
def verify_model(p):
    p=Path(p)
    for f in ['config.json','tokenizer_config.json']:
        if not (p/f).is_file():raise FileNotFoundError(p/f)
    idx=p/'model.safetensors.index.json'
    names=set(load(idx)['weight_map'].values()) if idx.exists() else {'model.safetensors'}
    for f in names:
        if not (p/f).is_file():raise FileNotFoundError(p/f)
    metadata={f.name for f in p.iterdir() if f.is_file() and f.suffix in {'.json','.txt','.model','.tiktoken'}}
    return {f:sha(p/f) for f in sorted(names|metadata)}
