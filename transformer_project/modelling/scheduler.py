import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class TransformerLRScheduler(_LRScheduler):
    """
    Learning rate scheduler from "Attention is All You Need" paper.
    
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    
    This increases the learning rate linearly for the first warmup_steps,
    then decreases it proportionally to the inverse square root of the step number.
    """
    
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000, last_epoch: int = -1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step_count = 0
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = max(1, self._step_count)
        scale = self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        self._step_count += 1
        values = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr
        self._last_lr = values


class WarmupLRScheduler:
    """
    Alternative implementation without inheriting from _LRScheduler.
    """
    
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step_count = 0
        self._rate = 0
    
    def get_lr(self):
        return self._rate
    
    def step(self):
        self._step_count += 1
        rate = self._get_rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = rate
        self._rate = rate
        return rate
    
    def _get_rate(self):
        step = self._step_count
        return self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))


def get_optimizer(model, lr: float = 1.0, weight_decay: float = 0.01, 
                  betas: tuple = (0.9, 0.98), eps: float = 1e-9):
    """
    Create AdamW optimizer with separate parameter groups for weight decay.
    
    Bias and LayerNorm parameters should not have weight decay applied.
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'layer_norm' in name or 'LayerNorm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
    return optimizer
