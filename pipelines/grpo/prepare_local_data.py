"""Build the small portable train/dev pack from locally verified author TRAIN data."""
import random
from collections import Counter
from build_data_base import stream_array, key
from common import ROOT, CONFIG as C, load, save, sha, readl


def main():
    workspace = ROOT.parents[1]
    sft = workspace/'RAG/clm_training_step4_data/data/train_10k.jsonl'
    if sha(sft) != '9922932781ea8b623a67a83edb705196d24689bb5354e722d705c503a6c94e72':
        raise ValueError('Unexpected SFT selection')
    excluded_sft = {key(r['question']) for r in readl(sft)}
    old_paths = [workspace/'RAG'/n/'data/questions.json' for n in ['clm_rl_reward32','clm_rl_grpo_smoke']]
    excluded_old = {key(r['question']) for p in old_paths for r in load(p)}
    old_audit = load(workspace/'RAG/clm_rl_reward32/data/selection_audit.json')
    excluded_test = set(); hashes = {}
    for item in old_audit['evaluation_sources'].values():
        p = workspace/item['path']
        actual = sha(p)
        if actual != item['sha256']: raise ValueError('Evaluation exclusion source changed')
        hashes[str(p.relative_to(workspace))] = actual
        excluded_test.update(key(r['question']) for r in stream_array(p))
    train = workspace/'dataset/dataset/2WikiMultiHopQA/train.json'
    source_hash = sha(train)
    if source_hash != old_audit['train_sha256']: raise ValueError('Training source changed')
    excluded = excluded_sft | excluded_old | excluded_test
    rng = random.Random(C['seed']); seen = set(); sample = []; counts = Counter(); eligible = 0
    size = C['train_questions'] + C['dev_questions']
    for index, row in enumerate(stream_array(train)):
        counts['source_rows'] += 1; k = key(row['question'])
        if k in excluded: counts['excluded'] += 1; continue
        if k in seen: counts['duplicate'] += 1; continue
        seen.add(k)
        if not row.get('answers') or not all(isinstance(x,str) and x.strip() for x in row['answers']):
            counts['invalid_gold'] += 1; continue
        docs = [dict(title=d['title'],text=d['text'],score=float(d['score']) if d.get('score') is not None else None) for d in row['ctxs'][:5]]
        if len(docs) != 5: counts['invalid_docs'] += 1; continue
        item = dict(id=f'2WikiMultiHopQA:train:{index}',source_index=index,question=row['question'],ctxs=docs,answers=row['answers'])
        eligible += 1
        if len(sample) < size: sample.append(item)
        else:
            j = rng.randrange(eligible)
            if j < size: sample[j] = item
        if index % 30000 == 0: print('SCAN',index,flush=True)
    rng.shuffle(sample)
    if len(sample) != size: raise ValueError('Too few eligible questions')
    for i,row in enumerate(sample): row['split'] = 'dev' if i<C['dev_questions'] else 'train'
    save(ROOT/'data/questions.json',[{k:v for k,v in row.items() if k!='answers'} for row in sample])
    save(ROOT/'data/gold.json',[dict(id=row['id'],answers=row['answers']) for row in sample])
    save(ROOT/'data/selection_audit.json',dict(seed=C['seed'],counts=dict(counts),eligible=eligible,
        sampling='Uniform reservoir then deterministic shuffle; no filtering on model outcomes.',
        train_questions=C['train_questions'],dev_questions=C['dev_questions'],
        excluded_sft=len(excluded_sft),excluded_previous_rl=len(excluded_old),excluded_evaluation=len(excluded_test),
        overlap_sft=0,overlap_previous_rl=0,overlap_evaluation=0,overlap_train_dev=0,
        train_sha256=source_hash,sft_sha256=sha(sft),previous_rl_sha256={str(p.relative_to(workspace)):sha(p) for p in old_paths},
        evaluation_sha256=hashes,limitation='Normalized exact question deduplication; paraphrases are not detected.'))
    print('PILOT_DATA_READY',size,flush=True)


if __name__ == '__main__': main()
