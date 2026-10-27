"""Extract only the ten author train/test JSON files from the upstream ZIP."""
import argparse
from pathlib import Path, PurePosixPath
import shutil
import zipfile
from common import abspath, save, sha
from data_utils import NAMES


def unpack(archive, output):
    if output.exists():
        raise FileExistsError('Use a new author-data directory: ' + str(output))
    with zipfile.ZipFile(archive) as z:
        chosen = {}
        for member in z.infolist():
            p = PurePosixPath(member.filename.replace('\\', '/'))
            if len(p.parts) < 2 or p.parts[-2] not in NAMES or p.name not in ('train.json', 'test.json'):
                continue
            if p.is_absolute() or '..' in p.parts or member.is_dir():
                raise ValueError('Invalid archive member: ' + member.filename)
            key = '/'.join(p.parts[-2:])
            if key in chosen:
                raise ValueError('Ambiguous duplicate dataset file: ' + key)
            chosen[key] = member
        required = {f'{name}/{split}.json' for name in NAMES for split in ('train', 'test')}
        if set(chosen) != required:
            raise ValueError('Missing author train/test files: ' + str(sorted(required - set(chosen))))
        output.mkdir(parents=True)
        for key, member in chosen.items():
            target = output/key; target.parent.mkdir(exist_ok=True)
            with z.open(member) as source, target.open('xb') as dest:
                shutil.copyfileobj(source, dest, 8 * 1024 * 1024)
    report = dict(archive_sha256=sha(archive), files_sha256={n: sha(output/n) for n in sorted(required)},
                  source='Author preprocessed data linked from RAG-ReasonAlignment README')
    save(output/'download_manifest.json', report)
    print('AUTHOR_DATA_READY', output, flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True); p.add_argument('--output', default='data/raw/author')
    a = p.parse_args(); unpack(abspath(a.archive), abspath(a.output))


if __name__ == '__main__': main()
