"""Maintainer package builder; excludes outputs, caches, models and data."""
import tarfile
import hashlib
from shared import ROOT,save,sha

def main():
    files=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in {'.py','.json','.txt'} and p.name!='source_manifest.json']
    for name in ['eval','configs','reference']:
        files.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and p.suffix in {'.py','.yaml','.json'})
    m=dict(package='clm_rl_step7_2wiki_v3',version=3,files_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)})
    save(ROOT/'source_manifest.json',m);files.append(ROOT/'source_manifest.json')
    out=ROOT.parent/'clm_rl_step7_2wiki_v3.tar.gz'
    with tarfile.open(out,'w:gz') as t:
        for p in sorted(files):t.add(p,arcname='clm_rl_step7_2wiki_v3/'+p.relative_to(ROOT).as_posix())
    with tarfile.open(out) as t:
        for n,h in m['files_sha256'].items():assert hashlib.sha256(t.extractfile('clm_rl_step7_2wiki_v3/'+n).read()).hexdigest()==h
    print(out.name,'bytes',out.stat().st_size,'sha256',sha(out))

if __name__=='__main__':main()
