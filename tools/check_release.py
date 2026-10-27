"""Read-only release checks: source identity, syntax, private-file exclusions and links."""
import argparse,ast,json,re
from pathlib import Path
from common import ROOT,load,save,sha
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--write-report',action='store_true');a=ap.parse_args()
    copied=load(ROOT/'provenance/source_inventory.json')['copied_without_modification']
    for row in copied:
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Historical source modified: '+row['path'])
    compiled=0
    for p in ROOT.rglob('*.py'):
        if p.relative_to(ROOT).parts[0]=='data' or any(x in p.relative_to(ROOT).parts for x in ['private','runs','.git','__pycache__']):continue
        ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p));compiled+=1
    forbidden=[];secrets=[]
    for p in ROOT.rglob('*'):
        if not p.is_file() or p.relative_to(ROOT).parts[0]=='data' or any(x in p.relative_to(ROOT).parts for x in ['private','runs','.git','__pycache__']):continue
        if p.suffix in {'.safetensors','.bin','.pt','.arrow','.tar','.gz','.zip','.docx','.pptx'}:forbidden.append(str(p.relative_to(ROOT)))
        if p.suffix in {'.py','.json','.md','.txt','.yaml','.sh'}:
            text=p.read_text(encoding='utf-8-sig')
            if re.search(r'\b(?:hf_|ghp_|github_pat_|sk-proj-)[A-Za-z0-9_\-]{25,}',text):secrets.append(str(p.relative_to(ROOT)))
    if forbidden or secrets:raise ValueError({'forbidden_files':forbidden,'possible_secrets':secrets})
    broken=[]
    for p in [ROOT/'README.md',*(ROOT/'docs').glob('*.md')]:
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if '://' in link or link.startswith('#'):continue
            target=p.parent/link.split('#')[0]
            if target==ROOT/'provenance/release_validation.json' and a.write_report:continue
            if not target.exists():broken.append((p.name,link))
    if broken:raise ValueError({'broken_links':broken})
    result={'status':'passed','historical_files_verified':len(copied),'python_files_syntax_checked':compiled,
            'forbidden_public_files':forbidden,'secret_pattern_hits':secrets,'broken_document_links':broken,
            'scope':'Static CPU checks only. Historical GPU runs are independently recorded; portable GPU rerun has not been performed here.'}
    if a.write_report:save(ROOT/'provenance/release_validation.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
