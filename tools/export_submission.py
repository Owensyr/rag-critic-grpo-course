"""Lossless final-rationale export; never infer or replace a short answer."""
import argparse
import csv
from pathlib import Path
from common import load, save, sha

def template_ids(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f)
        if r.fieldnames!=['id','answer']:raise ValueError('Template must have id,answer columns')
        ids=[row['id'] for row in r]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate template IDs')
    return ids
def export(rationale, sample, output, preview=False):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    rows=load(rationale);all_ids=template_ids(sample)
    wanted=all_ids[:len(rows)] if preview else all_ids
    if not rows or [str(r['id']) for r in rows]!=wanted:raise ValueError('Wrong row count or ID order')
    if any(not isinstance(r.get('rationale'),str) or not r['rationale'].strip() for r in rows):raise ValueError('Empty/non-text final rationale')
    expected=[{'id':str(r['id']),'answer':r['rationale']} for r in rows]
    output.parent.mkdir(parents=True,exist_ok=True);tmp=output.with_name(output.name+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','answer']);w.writeheader();w.writerows(expected)
    with tmp.open(encoding='utf-8',newline='') as f:
        if list(csv.DictReader(f))!=expected:raise ValueError('CSV round-trip mismatch')
    tmp.replace(output)
    report={'rows':len(rows),'preview_only':preview,'answer_format':'full_rationale_verbatim',
            'csv_sha256':sha(output),'rationale_sha256':sha(rationale),'sample_sha256':sha(sample)}
    save(output.with_suffix('.manifest.json'),report)
    return report
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rationale',type=Path,required=True);p.add_argument('--sample',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--preview',action='store_true');a=p.parse_args()
    print(export(a.rationale,a.sample,a.output,a.preview))
if __name__=='__main__':main()
