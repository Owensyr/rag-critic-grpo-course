"""Local data preparation only; package the 32 questions, not the large raw dataset."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import unicodedata
from model_provenance import sha, save


def key(text):
    return ' '.join(re.sub(r'[^\w\s]', ' ', unicodedata.normalize('NFKC', text).casefold()).split())


def stream_array(path, chunk_size=1024*1024):
    decoder = json.JSONDecoder()
    with Path(path).open(encoding='utf-8-sig') as f:
        buf = ''; eof = False; started = False; after_item = False
        while True:
            buf = buf.lstrip()
            if not buf:
                if eof:
                    raise ValueError('Incomplete JSON array')
                part = f.read(chunk_size); eof = not part; buf += part; continue
            if not started:
                if buf[0] != '[':
                    raise ValueError('Expected JSON array')
                started = True; buf = buf[1:]; continue
            if buf[0] == ']':
                if (buf[1:] + f.read()).strip():
                    raise ValueError('Trailing data after array')
                return
            if after_item:
                if buf[0] != ',':
                    raise ValueError('Missing comma')
                buf = buf[1:]; after_item = False; continue
            try:
                row, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                if eof: raise
                part = f.read(chunk_size); eof = not part; buf += part; continue
            yield row
            buf = buf[end:]; after_item = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--sft', type=Path, required=True)
    p.add_argument('--course-test', type=Path, required=True)
    a = p.parse_args(); root = Path(__file__).resolve().parent
    sft = [json.loads(line) for line in a.sft.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(sft) != 10000 or sha(a.sft) != '9922932781ea8b623a67a83edb705196d24689bb5354e722d705c503a6c94e72':
        raise ValueError('Unexpected original SFT selection')
    excluded_sft = {key(r['question']) for r in sft}; excluded_eval = set(); inputs = {}
    paths = sorted(a.dataset.glob('*/test.json')) + [a.course_test]
    if len(paths) != 6: raise ValueError('Expected five author test files and course test')
    for i, path in enumerate(paths):
        count = 0
        for r in stream_array(path):
            excluded_eval.add(key(r['question'])); count += 1
        inputs[f'evaluation_{i}'] = dict(path=str(path), sha256=sha(path), rows=count)
    train = a.dataset / '2WikiMultiHopQA/train.json'
    train_sha = sha(train)
    if train_sha != '715dc2e1c9704c4f7cf73b121ddb40bb2dafa455becac40c0c95d3e291491913':
        raise ValueError('Unexpected author train data')
    rng = random.Random(20261024); seen = set(); selected = []; eligible = 0; counts = Counter()
    for index, r in enumerate(stream_array(train)):
        counts['source_rows'] += 1; qkey = key(r['question'])
        if qkey in excluded_sft: counts['excluded_sft'] += 1; continue
        if qkey in excluded_eval: counts['excluded_evaluation'] += 1; continue
        if qkey in seen: counts['duplicate_question'] += 1; continue
        seen.add(qkey)
        try:
            aliases = r['answers']
            if not isinstance(aliases, list) or not aliases or not all(isinstance(x,str) and x.strip() for x in aliases):
                raise ValueError('bad labels')
            docs = r['ctxs'][:5]
            if len(docs) != 5: raise ValueError('bad docs')
            clean = [dict(title=d['title'], text=d['text'], score=None if d.get('score') is None else float(d['score'])) for d in docs]
            if not all(isinstance(d['title'],str) and isinstance(d['text'],str) and d['text'].strip() for d in clean):
                raise ValueError('bad text')
        except (ValueError, KeyError, TypeError):
            counts['invalid'] += 1; continue
        eligible += 1
        item = dict(id=f'2WikiMultiHopQA:train:{index}', source_index=index, question=r['question'], ctxs=clean, answers=aliases)
        if len(selected) < 32: selected.append(item)
        else:
            j = rng.randrange(eligible)
            if j < 32: selected[j] = item
        if index % 20000 == 0: print('SCAN',index,flush=True)
    selected.sort(key=lambda r:r['source_index'])
    if len(selected) != 32: raise ValueError('Insufficient candidates')
    questions = [{k:r[k] for k in ('id','source_index','question','ctxs')} for r in selected]
    gold = [{'id':r['id'],'answers':r['answers']} for r in selected]
    assert all(key(r['question']) not in excluded_sft | excluded_eval for r in selected)
    save(root/'data/questions.json', questions); save(root/'data/gold.json', gold)
    save(root/'data/selection_audit.json',dict(seed=20261024, sampling='uniform reservoir over eligible unique training questions; no selection on model outcomes',
         rows=32, eligible=eligible, counts=dict(counts), sft_questions_excluded=len(excluded_sft), evaluation_questions_excluded=len(excluded_eval),
         train_sha256=train_sha, sft_sha256=sha(a.sft), evaluation_sources=inputs,
         selected_ids=[r['id'] for r in selected], normalized_question_overlap_sft=0, normalized_question_overlap_evaluation=0,
         checks='NFKC/casefold/punctuation/whitespace question equality; paraphrases not detected',
         purpose='reward-protocol development subset of TRAIN, not a held-out accuracy benchmark',
         prompt_fields=['question','ctxs.title','ctxs.text','ctxs.score'], labels_only_in='data/gold.json',
         dropped_source_fields=['answers from prompt data','rationale','qa_pairs','ctxs.has_answer','other metadata']))
    print('DATA_READY rows=32',flush=True)


if __name__ == '__main__': main()
