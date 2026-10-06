"""
Optimizer and scheduler utilities.

Supports: AdamW, Lion, SGD
Schedulers: ReduceLROnPlateau, CosineAnnealingLR
"""
from __future__ import annotations

from typing import Literal, Iterator, Union
import math

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


class Lion(torch.optim.Optimizer):
    """
    Lion optimizer (https://arxiv.org/abs/2302.06675).
    
    A simple and effective optimizer that uses sign updates.
    Implemented locally to avoid external dependency.
    """
    
    def __init__(
        self,
        params: Iterator[nn.Parameter],
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        """
        Args:
            params: Model parameters
            lr: Learning rate
            betas: Coefficients for computing running averages
            weight_decay: Weight decay coefficient
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Lion does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                # Weight decay
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                
                # Momentum update
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        
        return loss


def build_optimizer(
    model: nn.Module,
    optimizer_name: Literal['adamw', 'lion', 'sgd'] = 'adamw',
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    momentum: float = 0.9,  # For SGD
    betas: tuple = (0.9, 0.99),  # For AdamW and Lion
) -> torch.optim.Optimizer:
    """
    Build optimizer.
    
    Args:
        model: Model to optimize
        optimizer_name: 'adamw', 'lion', or 'sgd'
        lr: Learning rate
        weight_decay: Weight decay
        momentum: Momentum for SGD
        betas: Betas for AdamW/Lion
    
    Returns:
        Optimizer instance
    """
    params = model.parameters()
    
    if optimizer_name == 'adamw':
        return AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    elif optimizer_name == 'lion':
        return Lion(params, lr=lr, weight_decay=weight_decay, betas=betas)
    elif optimizer_name == 'sgd':
        return SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: Literal['plateau', 'cosine'] = 'plateau',
    mode: str = 'min',
    factor: float = 0.5,
    patience: int = 5,
    min_lr: float = 1e-6,
    num_epochs: int = 50,
) -> Union[ReduceLROnPlateau, CosineAnnealingLR]:
    """
    Build learning-rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_name: 'plateau' (ReduceLROnPlateau) or 'cosine' (CosineAnnealingLR)
        mode: 'min' or 'max' (plateau only)
        factor: Factor to reduce LR by (plateau only)
        patience: Number of epochs to wait (plateau only)
        min_lr: Minimum learning rate
        num_epochs: Cosine annealing period T_max
    
    Returns:
        Scheduler instance
    """
    if scheduler_name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)
    return ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
    )


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


if __name__ == '__main__':
    # Test optimizers
    model = nn.Linear(10, 2)
    
    print("Testing AdamW:")
    opt = build_optimizer(model, 'adamw', lr=1e-3)
    print(f"  Type: {type(opt).__name__}")
    print(f"  LR: {get_lr(opt)}")
    
    print("\nTesting Lion:")
    opt = build_optimizer(model, 'lion', lr=1e-4)
    print(f"  Type: {type(opt).__name__}")
    print(f"  LR: {get_lr(opt)}")
    
    print("\nTesting SGD:")
    opt = build_optimizer(model, 'sgd', lr=1e-2)
    print(f"  Type: {type(opt).__name__}")
    print(f"  LR: {get_lr(opt)}")
    
    print("\nTesting scheduler:")
    scheduler = build_scheduler(opt, mode='min', patience=3)
    print(f"  Type: {type(scheduler).__name__}")
    
    # Simulate steps
    x = torch.randn(4, 10)
    for i in range(5):
        y = model(x)
        loss = y.sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        
        # Simulate validation loss
        val_loss = 1.0 - i * 0.1
        scheduler.step(val_loss)
        print(f"  Epoch {i+1}: val_loss={val_loss:.2f}, lr={get_lr(opt):.6f}")
