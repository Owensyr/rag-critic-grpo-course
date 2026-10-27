import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ADAPTER_SHA = 'f0c8979757af7dffd41635eb61387902a7d367f73730c9bc31855cc2b8388343'
BASE_CONFIG_SHA = '29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e'
SOURCE_SHA = 'c500875fc40020ea3fe302e8797deb4512a46c92e54ab5150d52e16f9928de72'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def verify_package():
    for name, expected in load(ROOT / 'source_manifest.json')['files_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Package file changed: ' + name)


def verify_merge(model):
    model = Path(model)
    report = load(model / 'merge_manifest.json')
    if report['status'] != 'complete' or report['adapter_sha256'] != ADAPTER_SHA:
        raise ValueError('Expected the reviewed final trained adapter')
    if report['source_sha256'] != SOURCE_SHA or not report['changed_weight_elements']:
        raise ValueError('Missing training provenance or merged weight changes')
    actual = {p.name for p in model.iterdir() if p.is_file() and p.name != 'merge_manifest.json'}
    if actual != set(report['files_sha256']):
        raise ValueError('Merged model file set changed')
    print('Verifying merged model hashes...', flush=True)
    for name, expected in report['files_sha256'].items():
        if sha(model / name) != expected:
            raise ValueError('Merged model changed: ' + name)
    return report
