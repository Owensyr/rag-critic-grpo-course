"""Explicit critic-only LoRA binding and a small observable-effect probe."""
import json
import math
from pathlib import Path
from collections import Counter

def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)

def engine_options(phase):
    opts=dict(max_num_seqs=32,enforce_eager=True,enable_prefix_caching=False)
    if phase==2:opts.update(enable_lora=True,max_lora_rank=8,max_loras=1,lora_dtype='bfloat16')
    return opts

def request_for(phase,adapter):
    if not adapter:return None
    if phase!=2:raise ValueError('RL adapter is critic-only; never attach it to Qwen')
    from vllm.lora.request import LoRARequest
    return LoRARequest('clm_grpo_step016',16,str(Path(adapter).resolve()))

def registered_adapter_ids(model):
    engine=getattr(model,'llm_engine',None)
    if engine is None or not callable(getattr(engine,'list_loras',None)):
        raise RuntimeError('Engine cannot enumerate registered adapters')
    return sorted(int(x) for x in engine.list_loras())

def ensure_adapter_loaded(model,request):
    if request is None:return []
    engine=model.llm_engine
    if request.lora_int_id not in registered_adapter_ids(model):
        engine.add_lora(request)
    ids=registered_adapter_ids(model)
    if request.lora_int_id not in ids:raise ValueError('Requested RL adapter is not registered in the engine')
    print('RL_ADAPTER_REGISTERED',request.lora_int_id,request.lora_path,flush=True)
    return ids

def check_binding(outputs,request,model=None):
    """V1 0.10.2 does not populate RequestOutput.lora_request.

    Missing metadata is not evidence of a missing adapter. Require registration
    instead, retain a per-batch submitted request audit, and separately test effect.
    A returned conflicting identity is still an error on engines that echo it.
    """
    matched=0;missing=0
    ids=registered_adapter_ids(model) if request is not None and model is not None else []
    for out in outputs:
        actual=getattr(out,'lora_request',None)
        if request is None:
            if actual is not None:raise ValueError('Unexpected adapter on base/Qwen request')
        elif actual is None:
            missing+=1
        elif actual.lora_int_id!=request.lora_int_id or Path(actual.lora_path).resolve()!=Path(request.lora_path).resolve():
            raise ValueError('Engine returned a conflicting RL adapter identity')
        else:matched+=1
    if request is not None and request.lora_int_id not in ids:
        raise ValueError('Requested RL adapter missing from engine registry; output metadata alone is insufficient')
    return dict(submitted_lora_request_count=len(outputs) if request else 0,
                returned_metadata_matched_count=matched,returned_metadata_missing_count=missing,
                registered_adapter_ids=ids,
                evidence='Explicit per-batch lora_request plus engine registration; V1 may omit output metadata. This is not an accuracy or numerical-effect claim.')

def encode_output(out):
    c=out.outputs[0]
    return dict(text=c.text,token_ids=list(c.token_ids),finish_reason=c.finish_reason,
                logprobs=[{str(k):float(v.logprob) for k,v in step.items()} for step in (c.logprobs or [])])

def compare_same_prefix(a,b):
    if len(a)!=len(b):raise ValueError('Probe row count mismatch')
    largest=0.;compared=0;different=0
    for left,right in zip(a,b):
        different+=left['token_ids']!=right['token_ids']
        for i,(lt,rt) in enumerate(zip(left['token_ids'],right['token_ids'])):
            # Histories are identical until the first differing token (inclusive).
            lp=left['logprobs'][i];rp=right['logprobs'][i]
            for key in set(lp)&set(rp):
                x,y=lp[key],rp[key]
                if not math.isfinite(x) or not math.isfinite(y):raise ValueError('Nonfinite probe logprob')
                largest=max(largest,abs(x-y));compared+=1
            if lt!=rt:break
    return dict(max_logprob_delta=largest,compared_values=compared,different_token_sequences=different)

def probe(model,prompts,request,output):
    from vllm import SamplingParams
    if request is None:raise ValueError('RL probe requires an adapter')
    ensure_adapter_loaded(model,request)
    params=SamplingParams(temperature=0,max_tokens=32,logprobs=5,seed=42)
    def collect(adapter):
        rows=model.chat(prompts[:5],params,lora_request=adapter,use_tqdm=False)
        evidence=check_binding(rows,adapter,model)
        return [encode_output(r) for r in rows],evidence
    before,base_binding=collect(None);adapted,rl_binding=collect(request);repeat,repeat_binding=collect(None)
    drift=compare_same_prefix(before,repeat);effect=compare_same_prefix(before,adapted)
    stable=drift['different_token_sequences']==0
    detectable=(effect['compared_values']>0 and effect['max_logprob_delta']>max(1e-6,10*drift['max_logprob_delta']))
    good=stable and detectable
    dump(output,dict(status='passed' if good else 'inconclusive',adapter_id=request.lora_int_id,
        adapter_path=request.lora_path,base_repeat=drift,adapter_effect=effect,
        binding=dict(base=base_binding,adapted=rl_binding,base_repeat=repeat_binding),
        base_outputs=before,adapter_outputs=adapted,base_repeat_outputs=repeat,
        limitation='Binding and numerical effect check on five prompts; not an accuracy test or proof of HF/vLLM numerical identity.'))
    # Optional standalone diagnostic only; noisy base repeats are not inference failures.
    print('RL_ADAPTER_EFFECT_CONFIRMED' if good else 'RL_ADAPTER_DIAGNOSTIC_INCONCLUSIVE',effect,flush=True)

def save_runtime(path,outputs,args,request,model):
    evidence=check_binding(outputs,request,model)
    dump(path,dict(phase=args.phase,rows=len(outputs),model=args.critique_model if args.phase==2 else args.model_name_or_path,
        adapter_path=request.lora_path if request else None,adapter_id=request.lora_int_id if request else None,
        **evidence,
        finish_reasons=dict(Counter(o.outputs[0].finish_reason for o in outputs)),
        total_completion_tokens=sum(len(o.outputs[0].token_ids) for o in outputs),engine=engine_options(args.phase),
        dtype='bfloat16',tensor_parallel_size=args.gpu_num,max_tokens=args.max_tokens,max_model_len=args.max_model_len))
