from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parent
files=[]
for directory in ('reports','logs','configs'):
    files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt','.png'))
for phase in ('smoke','full'):
    base=ROOT/'artifacts'/phase
    for name in ('trainer_state.json','trainer_log.jsonl','train_results.json','eval_results.json','all_results.json','adapter_config.json','training_loss.png','training_eval_loss.png'):
        if (base/name).exists():files.append(base/name)
    files.extend(base.glob('gpu_rank*.json'))
    files.extend(base.glob('checkpoint-*/trainer_state.json'))
files.extend([ROOT/'package_manifest.json',ROOT/'README.txt'])
target=ROOT/'clm_step5_training_reports.tar.gz'
with tarfile.open(target,'w:gz') as t:
    for p in sorted(set(files)):t.add(p,arcname=p.relative_to(ROOT).as_posix())
print('TRAINING_REPORTS_PACKED',target)
