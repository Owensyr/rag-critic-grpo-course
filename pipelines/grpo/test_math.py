"""CPU-only differentiable tests, run with the existing training environment on server."""
import torch
from critic_worker import loss_terms, selected_logprobs
from common import group_advantages, CONFIG as C


def main():
    torch.manual_seed(42)
    adv=torch.tensor(group_advantages([0.0,0.1,0.9,1.0]))
    current=torch.full((4,3),-2.0,requires_grad=True)
    old=current.detach().clone();ref=old.clone()
    losses=[loss_terms(current[i],old[i],ref[i],adv[i],.04,.2)[0] for i in range(4)]
    loss=torch.stack(losses).mean();loss.backward()
    assert current.grad[0].mean()>0 and current.grad[-1].mean()<0
    assert all(x==0 for x in group_advantages([.1]*4))
    x=torch.tensor([-1.5,-2.2],requires_grad=True);o=torch.tensor([-2.,-2.]);r=torch.tensor([-2.,-2.])
    _,kl,_=loss_terms(x,o,r,0.,.04,.2);assert kl.item()>=0
    high=torch.tensor([1.0],requires_grad=True)
    loss_terms(high,torch.tensor([0.]),torch.tensor([1.]),1.,0.,.2)[0].backward()
    assert high.grad.item()==0  # positive advantage clipping branch
    low=torch.tensor([-1.0],requires_grad=True)
    loss_terms(low,torch.tensor([0.]),torch.tensor([-1.]),-1.,0.,.2)[0].backward()
    assert low.grad.item()==0  # negative advantage clipping branch
    # Independent causal toy verifies chunked projection gradients and token shift
    # even where transformers is unavailable locally.
    from types import SimpleNamespace
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__();self.embed=torch.nn.Embedding(19,8)
        def forward(self,input_ids,**kwargs):
            return SimpleNamespace(last_hidden_state=self.embed(input_ids).cumsum(1))
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.model=Backbone();self.lm_head=torch.nn.Linear(8,19,bias=False);self.device=torch.device('cpu')
        def get_base_model(self):return self
    toy=Toy();prefix=[1,5,2];tail=[3,4,7];tokens=torch.tensor([prefix+tail])
    full=toy.lm_head(toy.model(input_ids=tokens).last_hidden_state)[:,len(prefix)-1:-1].float()
    direct=full.log_softmax(-1).gather(-1,torch.tensor(tail)[None,:,None]).flatten()
    (-direct.mean()).backward();expected_grads=[p.grad.clone() for p in toy.parameters()]
    toy.zero_grad(set_to_none=True);previous=C['logit_chunk'];C['logit_chunk']=2
    computed=selected_logprobs(toy,prefix,tail,gradients=True)
    assert torch.allclose(computed.detach(),direct.detach(),atol=1e-6)
    (-computed.mean()).backward()
    assert all(torch.allclose(p.grad,g,atol=1e-6) for p,g in zip(toy.parameters(),expected_grads))
    C['logit_chunk']=previous
    import sys
    if '--loss-only' in sys.argv:
        print('GRPO_LOSS_CLIPPING_COMPLETION_MASK_CHUNK_GRADIENT_OK; tiny Hugging Face model test not run')
        return
    # A tiny Llama + LoRA verifies shifted completion-only logprobs, chunked logits,
    # adapter gradients, and that disabling the adapter freezes the reference.
    from transformers import LlamaConfig,LlamaForCausalLM
    from peft import LoraConfig,get_peft_model
    cfg=LlamaConfig(vocab_size=43,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                    num_attention_heads=4,num_key_value_heads=2,attention_dropout=0.0)
    base=LlamaForCausalLM(cfg)
    model=get_peft_model(base,LoraConfig(task_type='CAUSAL_LM',r=2,lora_alpha=4,lora_dropout=0.,target_modules=['q_proj','v_proj']))
    model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    prompt=[1,5,9];completion=[4,7,2]
    model.eval()
    with torch.no_grad():
        logits=model(input_ids=torch.tensor([prompt+completion]),use_cache=False).logits[:,len(prompt)-1:-1].float()
        expected=logits.log_softmax(-1).gather(-1,torch.tensor(completion)[None,:,None]).flatten()
        old=selected_logprobs(model,prompt,completion)
        assert torch.allclose(old,expected,atol=1e-6)
        with model.disable_adapter():ref=selected_logprobs(model,prompt,completion).clone()
    prev_chunk=C['logit_chunk'];C['logit_chunk']=2
    try:
        model.train()
        cur=selected_logprobs(model,prompt,completion,gradients=True)
        assert torch.allclose(cur.detach(),old,atol=1e-6)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01,weight_decay=0)
        loss_terms(cur,old,ref,1.,.04,.2)[0].backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters() if p.requires_grad)
        opt.step();model.eval()
        with torch.no_grad():
            new=selected_logprobs(model,prompt,completion)
            with model.disable_adapter():ref_after=selected_logprobs(model,prompt,completion)
        assert not torch.allclose(new,old,atol=1e-6)
        assert torch.equal(ref_after,ref)
    finally:C['logit_chunk']=prev_chunk
    print('GRPO_MATH_AND_TINY_LORA_OK')


if __name__=='__main__':main()
