"""Collect diagnostics and five-case texts, without model weights/full results."""
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
files = [ROOT / 'README.txt', ROOT / 'source_manifest.json', ROOT / 'training_review.json']
for folder in ('logs', 'reports', 'configs', 'artifacts/smoke'):
    files.extend(p for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in ('.json', '.jsonl', '.csv', '.log', '.txt'))
files.extend((ROOT / 'artifacts').glob('*/run_manifest.json'))
target = ROOT / 'clm_step6_reports.tar.gz'
with tarfile.open(target, 'w:gz') as t:
    for p in sorted(set(files)):
        if p.is_file():
            t.add(p, arcname=p.relative_to(ROOT).as_posix())
print('STEP6_REPORTS_PACKED', target)
