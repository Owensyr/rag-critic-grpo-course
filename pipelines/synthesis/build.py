"""Construct a reproducible 10K hierarchy reconstruction; never overwrite source data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import string
import unicodedata

ROOT = Path(__file__).resolve().parent
NAMES = ('2WikiMultiHopQA', 'ASQA', 'NaturalQuestions', 'PopQA', 'TriviaQA')
QUOTAS = {0: 400, 1: 400, 2: 400, 3: 200, 4: 200, 5: 200}
STOP = set('a an the is are was were be been being of in on at by for from to and or who what which when where why how does do did has have had with as it its this that these those their his her name film movie song book director author born country first last same different number total new come out'.split())
TRANS = str.maketrans('', '', string.punctuation)

def load(p):
    with p.open(encoding='utf-8') as f: return json.load(f)

def save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def norm(s):
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', s.lower().translate(TRANS)).split())

def key(s):
    return ' '.join(re.sub(r'[^\w\s]', ' ', unicodedata.normalize('NFKC',s).casefold()).split())

def terms(s):
    return set(re.findall(r'\w+',s.casefold())) - STOP

def groups(r, name):
    values=[q['answers'] for q in r['qa_pairs']] if name=='ASQA' else [r['answers']]
    result=[[norm(a) for a in g] for g in values]
    if not result or any(not g or any(not a for a in g) for g in result):
        raise ValueError('Empty normalized label')
    return result

def flags(r, name, docs):
    gs=groups(r,name); any_flags=[]; full_flags=[]
    for d in docs:
        text=norm('Document (Title: '+d['title']+'): '+d['text'])
        hits=[any(a in text for a in g) for g in gs]
        any_flags.append(any(hits)); full_flags.append(all(hits))
    return any_flags,full_flags

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--course-test',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    seed=42; rng=random.Random(seed); evaluation=set(); inputs={}
    for name,p in [(n,args.data_root/n/'test.json') for n in NAMES]+[('course',args.course_test)]:
        rows=load(p);evaluation.update(key(r['question']) for r in rows)
        inputs[name+'/test']={'rows':len(rows),'sha256':sha(p)};del rows
    used=set(); corpus=[];report={'seed':seed,'datasets':{},'evaluation_overlap':0,'asqa_policy':'any_subquestion_answer_present','official_asqa_policy':'all_subquestion_answers_present_in_one_document','reconstruction_not_original_author_indices':True}
    for name in NAMES:
        path=args.data_root/name/'train.json';rows=load(path);inputs[name+'/train']={'rows':len(rows),'sha256':sha(path)}
        pools={n:[] for n in QUOTAS}; eligible=[];seen=set();excluded=Counter();raw_counts=Counter();official_counts=Counter()
        for i,r in enumerate(rows):
            q=key(r['question'])
            if q in evaluation:excluded['evaluation_overlap']+=1;continue
            if q in seen or q in used:excluded['duplicate_question']+=1;continue
            seen.add(q)
            try:
                docs=r['ctxs'][:5]
                if len(docs)!=5:raise ValueError('not five docs')
                for d in docs:float(d['score'])
                a,f=flags(r,name,docs)
            except (ValueError,KeyError,TypeError):excluded['invalid_record']+=1;continue
            pools[sum(a)].append(i);eligible.append(i);raw_counts[sum(a)]+=1;official_counts[sum(f)]+=1
            if i%20000==0: print('CLASSIFY',name,i,flush=True)
        info={'available_partial':dict(sorted(raw_counts.items())),'available_official':dict(sorted(official_counts.items())),'excluded':dict(excluded)}
        report['datasets'][name]=info
        shortages={str(n):QUOTAS[n]-len(pools[n]) for n in QUOTAS if len(pools[n])<QUOTAS[n]}
        if shortages:
            info['shortages']=shortages;report['status']='blocked_shortage';save(args.output/'build_report.json',report)
            raise RuntimeError(f'{name}: insufficient unique questions {shortages}')
        selected=[];chosen=set()
        for n,count in QUOTAS.items():
            for i in rng.sample(pools[n],count):selected.append((i,'H2' if n==0 else 'H34_support_'+str(n),rows[i]['ctxs'][:5],None));chosen.add(i)
        candidates=[i for i in eligible if i not in chosen];rng.shuffle(candidates);noise_skips=0;noise_count=0
        for i in candidates:
            r=rows[i];qt=terms(r['question']);own_docs={(d['title'],d['text']) for d in r['ctxs'][:5]};donor=None
            for _ in range(500):
                j=rng.choice(eligible)
                if i==j or qt & terms(rows[j]['question']):continue
                docs=rows[j]['ctxs'][:5]
                if any((d['title'],d['text']) in own_docs for d in docs):continue
                if any(flags(r,name,docs)[0]):continue
                donor=j;break
            if donor is None:noise_skips+=1;continue
            selected.append((i,'H1_random_other_question',rows[donor]['ctxs'][:5],donor));chosen.add(i);noise_count+=1
            if noise_count==200:break
        if noise_count!=200:raise RuntimeError(f'{name}: only {noise_count} noise candidates')
        info['noise_target_skips']=noise_skips;info['selected']=dict(Counter(s[1] for s in selected));info['duplicate_context_rows']=0
        for i,tier,docs,donor in selected:
            r=rows[i];a,f=flags(r,name,docs);q=key(r['question']);assert q not in used and q not in evaluation;used.add(q)
            if tier.startswith('H34'):assert sum(a)==int(tier.rsplit('_',1)[1])
            else:assert sum(a)==0
            if len({(d['title'],d['text']) for d in docs})<5:info['duplicate_context_rows']+=1
            item={'id':name+':train:'+str(i),'dataset':name,'source_train_index':i,'question':r['question'],'answers':r['answers'],'ctxs':[{**d,'score':float(d['score'])} for d in docs],'hierarchy':tier,'support_flags':a,'official_support_flags':f,'noise_donor_train_index':donor}
            if name=='ASQA':item['qa_pairs']=r['qa_pairs']
            corpus.append(item)
        print('SELECTED',name,json.dumps(info),flush=True);del rows
    assert len(corpus)==10000 and len(used)==10000
    rng.shuffle(corpus)
    target=args.output/'train_10k.jsonl'
    with target.open('w',encoding='utf-8',newline='\n') as f:
        for r in corpus:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    report.update(status='complete',rows=len(corpus),input_files=inputs,train_sha256=sha(target),notes=['ASQA any-subquestion presence authorized by user; official all-subquestion flags also retained.','Substring presence is an author-style proxy, not semantic entailment; yes/no labels may match incidental text.','H1 donors are from other eligible TRAIN questions with disjoint non-stopword question tokens; reject own passages and target-answer matches. Semantic irrelevance requires review.','Duplicate retrieved passages are preserved and counted as in author top-5 code; no passage duplication introduced to fill quotas.','Question deduplication is global across selected datasets, and normalized evaluation questions are excluded. No fuzzy paraphrase detection.','No rationale or critique from original train files is used as supervision.'])
    save(args.output/'build_report.json',report)
    preview=[]
    for name in NAMES:
        for tier in ['H1_random_other_question','H2']+['H34_support_'+str(i) for i in range(1,6)]:
            preview.extend([r for r in corpus if r['dataset']==name and r['hierarchy']==tier][:2])
    save(args.output/'selection_preview70.json',preview)
    print('CLM_DATA_READY rows=10000',flush=True)

if __name__=='__main__':main()
