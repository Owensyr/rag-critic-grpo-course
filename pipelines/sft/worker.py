"""Executed by torchrun: unmodified LLaMA-Factory SFT with telemetry callback."""
import json
import math
import os
from pathlib import Path
import sys
import time
import torch
from transformers import TrainerCallback
from llamafactory.train.tuner import run_exp

class Telemetry(TrainerCallback):
    def on_train_begin(self,args,state,control,model=None,**kwargs):
        self.start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if not trainable or any('lora_' not in n for n,p in trainable):raise RuntimeError('Expected LoRA-only trainable parameters')
        self.details={'rank':int(os.environ['RANK']),'trainable_parameters':sum(p.numel() for n,p in trainable),'all_parameters':sum(p.numel() for p in model.parameters()),'gradient_checkpointing':model.is_gradient_checkpointing,'cuda_device':torch.cuda.get_device_name(),'trainable_names': [n for n,p in trainable]}
        if not model.is_gradient_checkpointing:raise RuntimeError('Gradient checkpointing is required on 3090')
    def on_log(self,args,state,control,logs=None,**kwargs):
        for k,v in (logs or {}).items():
            if isinstance(v,(int,float)) and not math.isfinite(v):raise RuntimeError(f'Non-finite {k}: {v}')
    def on_train_end(self,args,state,control,**kwargs):
        self.details.update(global_step=state.global_step,epoch=state.epoch,elapsed_seconds=time.monotonic()-self.start,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
        Path(args.output_dir,f"gpu_rank{os.environ['RANK']}.json").write_text(json.dumps(self.details,indent=2)+'\n')

if __name__=='__main__':
    config=json.loads(Path(sys.argv[1]).read_text())
    run_exp(args=config,callbacks=[Telemetry()])
