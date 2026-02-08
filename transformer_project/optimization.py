import torch
import math
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer

class TransformerLRScheduler(LambdaLR):
    def __init__(self, optimizer: Optimizer, d_model: int, warmup_steps: int = 4000, last_epoch: int = -1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        
        def lr_lambda(step: int):
            if step == 0:
                step = 1
            
            _step = float(step)
            return (self.d_model ** -0.5) * min(_step ** -0.5, _step * (self.warmup_steps ** -1.5))

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)

def build_optimizer(model: torch.nn.Module, learning_rate: float = 0.0, weight_decay: float = 0.1, beta1: float = 0.9, beta2: float = 0.98, eps: float = 1e-9):
    
    decay_params = []
    no_decay_params = []
    
    no_decay_names = ['bias', 'layer_norm', 'norm'] 
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if any(nd in name for nd in no_decay_names):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.AdamW(
        optim_groups, 
        lr=learning_rate, 
        betas=(beta1, beta2), 
        eps=eps
    )
    
    return optimizer


def get_optimizer(model, lr: float = 1.0, weight_decay: float = 0.1, 
                  betas: tuple = (0.9, 0.98), eps: float = 1e-9):
    return build_optimizer(model, learning_rate=lr, weight_decay=weight_decay, 
                           beta1=betas[0], beta2=betas[1], eps=eps)
