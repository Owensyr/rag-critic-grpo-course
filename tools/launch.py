"""Run the explicit launch plan from prepare_workdir.py (Linux GPU server)."""
import argparse,os,subprocess
from common import abspath,load
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workdir',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    d=abspath(a.workdir);plan=load(d/'launch.json');cmd=plan['command']
    if a.resume:
        if not (d/'worker.py').exists():p.error('--resume applies to SFT; GRPO resumes automatically with the unchanged launch command')
        cmd=cmd+['--resume']
    subprocess.run(cmd,cwd=d,env={**os.environ,'CUDA_VISIBLE_DEVICES':plan['gpu_ids']},check=True)
if __name__=='__main__':main()
