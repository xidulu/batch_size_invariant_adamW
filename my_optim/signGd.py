import torch
from torch.optim.optimizer import Optimizer

from .types import OptFloat, OptLossClosure, Params, State

__all__ = ("SignSGD",)


class SignSGD(Optimizer):
    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if dampening < 0.0:
            raise ValueError("Invalid dampening value: {}".format(dampening))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening"
            )
        super(SignSGD, self).__init__(params, defaults)

    def __setstate__(self, state: State) -> None:
        super(SignSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        """Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if p.grad.is_sparse:
                    msg = (
                        "SignSGD does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )
                    raise RuntimeError(msg)

                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(
                            d_p
                        ).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                # Apply momentum and take sign
                p.data.add_(d_p.sign(), alpha=-group["lr"])

                # Apply weight decay
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-group["lr"] * weight_decay)
        return loss