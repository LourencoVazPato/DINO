from typing import List

import numpy as np
import torch
from numpy import pi


class ParamScheduler(object):
    def __init__(self):
        self.step_ = 0

    def step(self):
        self.step_ += 1


class ConstantScheduleWithLinearWarmup(ParamScheduler):
    def __init__(
        self,
        initial_value: float,
        final_value: float,
        num_warmup_epochs: int,
        num_steps_per_epoch: int,
    ):
        super().__init__()
        self.initial_value = initial_value
        self.final_value = final_value
        self.num_steps_per_epoch = num_steps_per_epoch
        self.num_warmup_steps = num_warmup_epochs * num_steps_per_epoch

    def __call__(self):
        if self.step_ < self.num_warmup_steps:
            return (
                self.initial_value
                + (self.final_value - self.initial_value)
                * self.step_
                / self.num_warmup_steps
            )
        else:
            return self.final_value


class CosineScheduler(ParamScheduler):
    def __init__(
        self,
        initial_value: float,
        final_value: float,
        num_epochs: int,
        num_steps_per_epoch: int,
    ):
        super().__init__()
        self.initial_value = initial_value
        self.final_value = final_value
        self.num_steps_per_epoch = num_steps_per_epoch
        self.num_total_steps = num_epochs * num_steps_per_epoch

    def __call__(self) -> float:
        return 1 - (1 - self.initial_value) * 0.5 * (
            np.cos(pi * self.step_ / self.num_total_steps) + 1
        )


class CosineSchedulerLinearWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        num_epochs: int,
        initial_value: float,
        final_value: float,
        num_steps_per_epoch: int,
        num_warmup_epochs: int = 0,
        warmup_start_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        num_warmup_steps = num_warmup_epochs * num_steps_per_epoch
        warmup_schedule = np.array([], dtype=np.float32)
        if num_warmup_steps > 0:
            warmup_schedule = np.linspace(
                warmup_start_lr, initial_value, num_warmup_steps
            )

        steps = np.arange(num_epochs * num_steps_per_epoch - num_warmup_steps)
        cosine_schedule = final_value + 0.5 * (initial_value - final_value) * (
            1 + np.cos(np.pi * steps / len(steps))
        )

        self.schedule = np.concatenate((warmup_schedule, cosine_schedule))
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:
        return [self.schedule[self._step_count]]
