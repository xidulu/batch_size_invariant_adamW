# Batch invariant adam

import math
import torch
from torch.optim.optimizer import Optimizer

version_higher = (torch.__version__ >= "1.5.0")


class AdamBI(Optimizer):
    r"""
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        gammas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-16)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-3, gammas=(0.1, 0.001), eps=1e-16, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= gammas[0] < 1.0:
            raise ValueError("Invalid gamma parameter at index 0: {}".format(gammas[0]))
        if not 0.0 <= gammas[1] < 1.0:
            raise ValueError("Invalid gamma parameter at index 1: {}".format(gammas[1]))
        defaults = dict(lr=lr, gammas=gammas, eps=eps, weight_decay=weight_decay)
        super(AdamBI, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamBI, self).__setstate__(state)

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # State initialization
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                    if version_higher else torch.zeros_like(p.data)

                # Exponential moving average of squared momentumized gradient values
                state['exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                    if version_higher else torch.zeros_like(p.data)

    def accumlate(self, kappa):
        """
        Update the summation with respect to k, using gradient from each micro batch.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                gamma1, gamma2 = group['gammas']
                # State initialization
                if len(state) == 0:
                    self.reset()
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']
                # Accumlate (gamma1 / kappa) * g_{t - k+ kappa}
                exp_avg.add_(grad, alpha=(1 / kappa) * gamma1) 
                # Accumlate (gamma2 / kappa) * g_{t - k+ kappa}^2
                exp_avg_var.addcmul_(grad, grad, value=(1 / kappa) * gamma2)

    def step(self, closure=None):
        """
        The step function would:
        1. Compute the bias correction term
        2. Apply weight decay if required
        3. Use the bias-corrected m and v to update to parameters
        4. Decay the m and v for next mini-batch update 
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                gamma1, gamma2 = group['gammas']
                # State initialization
                if len(state) == 0:
                    self.reset()
                # Weight decay
                p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                # get current state variable
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

                state['step'] += 1
                bias_correction1 = 1 - (1 - gamma1) ** (state['step'])
                bias_correction2 = 1 - (1 - gamma2) ** (state['step'])
                denom = (exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2))        
                step_size = group['lr'] / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                # Decay m_{t - kappa} and v_{t - kappa}
                exp_avg.mul_(1 - gamma1)
                exp_avg_var.mul_(1 - gamma2)

        return loss