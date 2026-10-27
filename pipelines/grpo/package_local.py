"""Maintainer-only packaging. Does not run models or alter any prior experiment."""
import json
from pathlib import Path
import tarfile
from common import ROOT,sha,save


def main():
    files=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in {'.py','.json','.txt'} and p.name!='source_manifest.json']
    files+=list((ROOT/'data').glob('*.json'))
    manifest=dict(package='clm_rl_grpo_pilot_v3',version=3,
        files_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)},
        scope='128 train / 128 held-out dev, outcome-only semantic reward, independent RL LoRA from SFT')
    save(ROOT/'source_manifest.json',manifest);files.append(ROOT/'source_manifest.json')
    target=ROOT.parent/'clm_rl_grpo_pilot_v3.tar.gz'
    with tarfile.open(target,'w:gz') as archive:
        for p in sorted(files):archive.add(p,arcname='clm_rl_grpo_pilot_v3/'+p.relative_to(ROOT).as_posix())
    with tarfile.open(target) as archive:
        names=archive.getnames()
        if len(names)!=len(files):raise ValueError('Archive file count mismatch')
        import hashlib
        for name,h in manifest['files_sha256'].items():
            if hashlib.sha256(archive.extractfile('clm_rl_grpo_pilot_v3/'+name).read()).hexdigest()!=h:
                raise ValueError('Archive hash mismatch: '+name)
    print(json.dumps(dict(package=str(target),bytes=target.stat().st_size,sha256=sha(target),files=len(files)),indent=2))


if __name__=='__main__':main()
