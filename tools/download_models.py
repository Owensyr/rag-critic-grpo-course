"""Download official base models; uses the user's Hugging Face authentication."""
import argparse
from common import configuration
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/paths.json')
    p.add_argument('--include-weak',action='store_true');p.add_argument('--revision',help='Optional revision for one model; use with --only')
    p.add_argument('--only',choices=['qwen','llama','weak']);a=p.parse_args();c=configuration(a.config)
    if a.revision and not a.only:p.error('--revision requires --only')
    from huggingface_hub import snapshot_download
    names={'qwen':'Qwen/Qwen2.5-7B-Instruct','llama':'meta-llama/Llama-3.1-8B-Instruct','weak':'Qwen/Qwen2.5-0.5B-Instruct'}
    for key in ([a.only] if a.only else ['qwen','llama']+(['weak'] if a.include_weak else [])):
        snapshot_download(repo_id=names[key],local_dir=c[key],revision=a.revision or 'main',
            allow_patterns=['*.json','*.safetensors','*.model','*.tiktoken','*.txt','*.md','LICENSE*','USE_POLICY*'])
        print('MODEL_DOWNLOADED',key,c[key])
if __name__=='__main__':main()
