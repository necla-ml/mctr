import math
import warnings

import torch
import torch.optim
from torch.optim.lr_scheduler import *

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def update_ema(model, model_ema, decay):
    """Apply exponential moving average update.
    The  weights are updated in-place as follow:
    w_ema = w_ema * decay + (1 - decay) * w
    Args:
        model: active model that is being optimized
        model_ema: running average model
        decay: exponential decay parameter
    """
    with torch.no_grad():
        if hasattr(model, "module"):
            # unwrapping DDP
            model = model.module
        msd = model.state_dict()
        for k, ema_v in model_ema.state_dict().items():
            model_v = msd[k].detach()
            ema_v.copy_(ema_v * decay + (1.0 - decay) * model_v)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The totale number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class WarmUpCosineAnnealingLR(CosineAnnealingLR):
    def __init__(self, optimizer, warm_multiplier, warm_duration, cos_duration, eta_min=0, last_epoch=-1):
        assert warm_duration >= 0
        assert warm_multiplier > 1.0
        self.warm_m = float(warm_multiplier)
        self.warm_d = warm_duration
        self.cos_duration = cos_duration
        self.cos_eta_min = eta_min
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, self.cos_duration, eta_min, last_epoch)

    def get_lr(self):
        if self.warm_d == 0:
            return super(WarmUpCosineAnnealingLR, self).get_lr()
        else:
            if not self._get_lr_called_within_step:
                warnings.warn("To get the last learning rate computed by the scheduler, "
                              "please use `get_last_lr()`.", UserWarning)
            if self.last_epoch == 0:
                return [lr / self.warm_m for lr in self.base_lrs]
                # return self.base_lrs / self.warm_m
            elif self.last_epoch <= self.warm_d:
                return [(self.warm_d + (self.warm_m - 1) * self.last_epoch) / (self.warm_d + (self.warm_m - 1) * (self.last_epoch - 1)) * group['lr'] for group in self.optimizer.param_groups]
            else:
                cos_last_epoch = self.last_epoch - self.warm_d
                if cos_last_epoch == 0:
                    return self.base_lrs
                elif (cos_last_epoch - 1 - self.cos_duration) % (2 * self.cos_duration) == 0:
                    return [group['lr'] + (base_lr - self.cos_eta_min) *
                            (1 - math.cos(math.pi / self.cos_duration)) / 2
                            for base_lr, group in
                            zip(self.base_lrs, self.optimizer.param_groups)]
                return [(1 + math.cos(math.pi * cos_last_epoch / self.cos_duration)) /
                        (1 + math.cos(math.pi * (cos_last_epoch - 1) / self.cos_duration)) *
                        (group['lr'] - self.cos_eta_min) + self.cos_eta_min
                        for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        if self.warm_d == 0:
            return super(WarmUpCosineAnnealingLR, self)._get_closed_form_lr()
        else:
            if self.last_epoch <= self.warm_d:
                return [base_lr * (self.warm_d + (self.warm_m - 1) * self.last_epoch) / (self.warm_d * self.warm_m) for base_lr in self.base_lrs]
            else:
                cos_last_epoch = self.last_epoch - self.warm_d
                return [self.cos_eta_min + (base_lr - self.cos_eta_min) *
                    (1 + math.cos(math.pi * cos_last_epoch / self.cos_duration)) / 2
                    for base_lr in self.base_lrs]