"""Prepare course inputs, select five-dataset SFT questions and fresh GRPO train/dev."""
import argparse
from collections import Counter
from pathlib import Path
import random
import shutil
import subprocess
import sys
from common import ROOT, abspath, load, save, sha
from data_utils import NAMES, clean_docs, question_key, read_jsonl, stream_array
from export_submission import template_ids


def prepare_course(test, sample, output, expected_rows=12576):
    if output.exists(): raise FileExistsError('Use a new course-data directory: ' + str(output))
    rows = list(stream_array(test)); ids = template_ids(sample)
    if len(rows) != expected_rows or [str(r.get('id')) for r in rows] != ids:
        raise ValueError(f'Expected {expected_rows} course questions in submission-template order')
    clean = []
    for r in rows:
        if not isinstance(r.get('question'), str) or not question_key(r['question']):
            raise ValueError('Empty question')
        if r.get('answers') is not None:
            raise ValueError('Course input must be unlabeled; do not substitute author train/test data')
        clean.append(dict(id=r['id'], question=r['question'], ctxs=clean_docs(r), answers=None))
    output.mkdir(parents=True)
    save(output/'test.json', clean); shutil.copyfile(sample, output/'sample_submission.csv')
    report = dict(rows=len(clean), raw_test_sha256=sha(test), raw_sample_sha256=sha(sample),
                  files_sha256={n:sha(output/n) for n in ('test.json', 'sample_submission.csv')},
                  policy='Keep IDs/order/questions/top5 title,text,score; remove other source metadata; no labels')
    save(output/'input_manifest.json', report)
    print('COURSE_DATA_READY', len(clean), output, flush=True)
    return report


def select_sft(author, course_test, output):
    # Use the unchanged historical hierarchy builder, not privately selected indices.
    for name in NAMES:
        for split in ('train', 'test'):
            if not (author/name/(split+'.json')).is_file():
                raise FileNotFoundError(author/name/(split+'.json'))
    subprocess.run([sys.executable, str(ROOT/'pipelines/synthesis/build.py'),
                    '--data-root', str(author), '--course-test', str(course_test), '--output', str(output)], check=True)


def select_grpo(author, course_test, sft_selection, output, seed=20261026,
                train_count=128, dev_count=128, previous=()):
    if output.exists(): raise FileExistsError('Use a new GRPO-data directory: ' + str(output))
    sft = read_jsonl(sft_selection)
    excluded_sft = {question_key(r['question']) for r in sft}
    excluded_eval = set(); evaluation = {}
    for name, path in [(n, author/n/'test.json') for n in NAMES] + [('course', course_test)]:
        count = 0
        for r in stream_array(path): excluded_eval.add(question_key(r['question'])); count += 1
        evaluation[name] = dict(sha256=sha(path), rows=count)
    excluded_old = set(); old_hashes = {}
    for index, path in enumerate(previous):
        excluded_old.update(question_key(r['question']) for r in stream_array(path))
        old_hashes[str(index)] = sha(path)
    excluded = excluded_sft | excluded_eval | excluded_old
    train = author/'2WikiMultiHopQA/train.json'
    counts = Counter(); seen = set(); sample = []; eligible = 0
    size = train_count + dev_count; rng = random.Random(seed)
    if train_count < 1 or dev_count < 1: raise ValueError('Nonempty train and dev required')
    for index, r in enumerate(stream_array(train)):
        counts['source_rows'] += 1
        q = r.get('question')
        if not isinstance(q, str) or not question_key(q): counts['invalid'] += 1; continue
        k = question_key(q)
        if k in excluded: counts['excluded'] += 1; continue
        if k in seen: counts['duplicate'] += 1; continue
        seen.add(k)
        try:
            answers = r['answers']
            if not isinstance(answers, list) or not answers or not all(isinstance(x,str) and x.strip() for x in answers):
                raise ValueError('Invalid gold aliases')
            docs = clean_docs(r)
        except (ValueError, KeyError, TypeError): counts['invalid'] += 1; continue
        item = dict(id=f'2WikiMultiHopQA:train:{index}', source_index=index, question=q, ctxs=docs, answers=answers)
        eligible += 1
        if len(sample) < size: sample.append(item)
        else:
            j = rng.randrange(eligible)
            if j < size: sample[j] = item
    if len(sample) != size: raise ValueError(f'Only {len(sample)} eligible questions; need {size}')
    rng.shuffle(sample)
    for i, row in enumerate(sample): row['split'] = 'dev' if i < dev_count else 'train'
    questions = [{k:v for k,v in r.items() if k!='answers'} for r in sample]
    gold = [dict(id=r['id'],answers=r['answers']) for r in sample]
    selected_keys = {question_key(r['question']) for r in questions}
    train_keys = {question_key(r['question']) for r in questions if r['split']=='train'}
    dev_keys = {question_key(r['question']) for r in questions if r['split']=='dev'}
    audit = dict(mode='fresh_selection_not_historical_pilot', seed=seed, train_questions=train_count,
                 dev_questions=dev_count, eligible=eligible, counts=dict(counts),
                 sampling='Uniform reservoir over eligible TRAIN questions then seeded shuffle; no outcome filtering',
                 excluded_sft=len(excluded_sft), excluded_previous_rl=len(excluded_old), excluded_evaluation=len(excluded_eval),
                 overlap_sft=len(selected_keys & excluded_sft), overlap_previous_rl=len(selected_keys & excluded_old),
                 overlap_evaluation=len(selected_keys & excluded_eval), overlap_train_dev=len(train_keys & dev_keys),
                 train_sha256=sha(train), sft_sha256=sha(sft_selection), previous_rl_sha256=old_hashes,
                 evaluation_sha256=evaluation, limitation='Normalized exact-question exclusion only, not paraphrase detection. '
                 'Historical reward32/smoke exclusions are absent unless explicitly supplied with --exclude-questions.')
    output.mkdir(parents=True); save(output/'questions.json', questions); save(output/'gold.json', gold)
    audit['files_sha256'] = {n:sha(output/n) for n in ('questions.json','gold.json')}
    save(output/'selection_audit.json', audit)
    print('GRPO_DATA_READY', size, output, flush=True)
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    course = sub.add_parser('course'); course.add_argument('--test', required=True); course.add_argument('--sample', required=True)
    course.add_argument('--output', default='data/prepared/course')
    for kind in ('sft', 'grpo'):
        q = sub.add_parser(kind); q.add_argument('--author-root', default='data/raw/author')
        q.add_argument('--course-test', default='data/prepared/course/test.json'); q.add_argument('--output', default='data/prepared/'+kind)
        if kind=='grpo':
            q.add_argument('--sft-selection', default='data/prepared/sft/train_10k.jsonl')
            q.add_argument('--exclude-questions', action='append', default=[])
    a = p.parse_args()
    if a.command=='course': prepare_course(abspath(a.test), abspath(a.sample), abspath(a.output))
    elif a.command=='sft': select_sft(abspath(a.author_root), abspath(a.course_test), abspath(a.output))
    else: select_grpo(abspath(a.author_root), abspath(a.course_test), abspath(a.sft_selection),
                      abspath(a.output), previous=[abspath(x) for x in a.exclude_questions])


if __name__ == '__main__': main()
