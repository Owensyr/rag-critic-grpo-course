"""Lazy vLLM loader shared by generation and isolated reward verification processes."""
from common import CONFIG as C, path, fingerprint


class Engine:
    def __init__(self):
        self.engine=None

    def generate(self, messages, max_tokens):
        from vllm import LLM, SamplingParams
        if self.engine is None:
            self.engine=LLM(model=str(path('qwen_model')),dtype='bfloat16',tensor_parallel_size=4,
                max_model_len=C['qwen_max_model_len'],max_num_seqs=4,gpu_memory_utilization=.85,
                enforce_eager=True,seed=C['seed'])
        tok=self.engine.get_tokenizer()
        lengths=[len(tok.apply_chat_template(m,tokenize=True,add_generation_prompt=True)) for m in messages]
        if any(n+max_tokens>C['qwen_max_model_len'] for n in lengths):
            raise ValueError('Qwen context exceeds budget; no silent truncation')
        outputs=self.engine.chat(messages,SamplingParams(temperature=0,top_p=1,max_tokens=max_tokens,seed=C['seed']),use_tqdm=False)
        return [dict(text=o.outputs[0].text,finish_reason=o.outputs[0].finish_reason,
                     completion_tokens=len(o.outputs[0].token_ids),prompt_tokens=n,prompt_sha256=fingerprint(m))
                for m,n,o in zip(messages,lengths,outputs)]
