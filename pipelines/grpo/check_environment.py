import argparse
import importlib.metadata as md
import json
import sys
from pathlib import Path
from common import path


def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['training','inference']);a=p.parse_args()
    expected=path(a.kind+'_python')
    print(json.dumps(dict(kind=a.kind,expected_python=str(expected),python=sys.executable,
                          prefix=sys.prefix,base_prefix=sys.base_prefix),indent=2),flush=True)
    if Path(sys.prefix).resolve()!=expected.parent.parent.resolve():
        raise RuntimeError('Wrong Python environment prefix; preserve the configured bin/python entry point')
    import torch
    if not torch.__version__.startswith('2.8.0'): raise ValueError('Expected existing torch 2.8.0')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=4: raise ValueError('Need four visible CUDA GPUs')
    names=['torch','transformers','peft','safetensors','accelerate'] if a.kind=='training' else ['torch','transformers','vllm']
    versions={n:md.version(n) for n in names}
    if a.kind=='training':
        from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
        from peft import LoraConfig,PeftModel,get_peft_model
        if versions['transformers']!='4.51.3' or versions['peft']!='0.15.2':
            raise ValueError('Use the unchanged clm_training_env from step 5; expected transformers 4.51.3 / peft 0.15.2')
    else:
        from vllm import LLM,SamplingParams
        if versions['vllm']!='0.10.2' or versions['transformers']!='4.55.2':
            raise ValueError('Expected rag-author vLLM 0.10.2 / transformers 4.55.2')
    print(json.dumps(dict(status='environment_ok',kind=a.kind,python=sys.executable,versions=versions,
        devices=[dict(name=torch.cuda.get_device_name(i),total_gib=torch.cuda.get_device_properties(i).total_memory/2**30) for i in range(4)]),indent=2))


if __name__=='__main__':main()
