import hashlib
import json
import os
from pathlib import Path
from model_provenance import load, sha

ROOT = Path(__file__).resolve().parent
CONFIG = load(ROOT/'config.json')


def path(key):
    # Keep the venv entry point: resolving bin/python symlinks can select the
    # base interpreter and bypass packages installed in clm_training_env.
    return Path(os.path.abspath(os.path.expanduser(CONFIG[key])))


def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    tmp.replace(p)


def jsonl(p, rows):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.tmp')
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    tmp.replace(p)


def readl(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def adapter_fingerprint(adapter):
    if adapter == 'base': return 'merged_sft_base'
    return fingerprint({p.name: sha(p) for p in sorted(Path(adapter).glob('adapter*')) if p.is_file()})


def contexts(row):
    docs = row['ctxs'][:5]
    if len(docs) != 5: raise ValueError('Expected exactly five documents')
    if docs[0]['score'] is not None and docs[1]['score'] is not None and docs[0]['score'] > docs[1]['score']:
        docs = docs[::-1]
    return '\n\n'.join(f"Document {i+1} (Title: {d['title']}): {d['text']}" for i,d in enumerate(docs))+'\n\n'


OLD_FIELD = ('\nAfter your reasoning, end your response with exactly one final answer field: '
    '<final_answer>your concise answer</final_answer>. '
    'Put only your chosen answer in the field, without explanations or alternative guesses. '
    'If you cannot determine an answer, put unknown in the field. Do not write anything after the field.')
NEW_FIELD = ('\nAfter your reasoning, write your chosen concise answer exactly once in '
    '<final_answer>...</final_answer>. Put only that answer inside the tags, without explanations '
    'or a list of alternative guesses. Do not invent an answer when you cannot determine one.')


def prompt(row, stage, initial=None, critique=None, field='new'):
    name = {'initial':'RationalePromptTemplate','critique':'CritiqueRationalePromptTemplate','refine':'RefinePromptTemplate'}[stage]
    t = load(ROOT/'prompt_templates.json')[name]
    params = dict(query=row['question'], text=contexts(row), rationale=initial, critique=critique)
    user = t['base_user_prompt'].format(**params)
    if stage != 'critique': user += {'original':'','old':OLD_FIELD,'new':NEW_FIELD}[field]
    system = t.get('base_system_prompt_rationale',t.get('base_system_prompt_critique')).format(**params)
    return [{'role':'system','content':system},{'role':'user','content':user}]


def split_rows(split):
    rows = load(ROOT/'data/questions.json')
    return [r for r in rows if r['split'] == split]


def round_rows(round_number):
    if round_number == 0: return split_rows('dev')
    rows = split_rows('train'); n = CONFIG['groups_per_round']
    return rows[(round_number-1)*n:round_number*n]


def group_advantages(rewards):
    import math
    if len(rewards) != CONFIG['candidates'] or not all(math.isfinite(r) for r in rewards):
        raise ValueError('Invalid reward group')
    mean = sum(rewards)/len(rewards)
    # Sample std, matching the original GRPO convention; constant groups get exactly zero.
    std = math.sqrt(sum((r-mean)**2 for r in rewards)/(len(rewards)-1))
    return [(r-mean)/(std+1e-4) for r in rewards]
