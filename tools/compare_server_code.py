"""Report whether downloaded server source matches the assembled historical snapshot."""
import argparse
from pathlib import Path
from common import ROOT,load,sha,abspath
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--private',default='private');a=p.parse_args();base=abspath(a.private)
    missing=[];changed=[];matched=0
    for row in load(ROOT/'provenance/source_inventory.json')['copied_without_modification']:
        source=base/'server_code'/row['source']
        if row['source']=='clm_rl_grpo_pilot_v3/data/reward_cases.json':source=base/'grpo/reward_cases.json'
        if not source.is_file():missing.append(row['source'])
        elif sha(source)!=row['sha256']:changed.append(row['source'])
        else:matched+=1
    import json;print(json.dumps({'matched':matched,'missing':missing,'changed':changed},ensure_ascii=False,indent=2))
    if missing or changed:raise SystemExit(1)
if __name__=='__main__':main()
