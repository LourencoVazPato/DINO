import numpy as np
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
