from typing import List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from dino.knn import compute_knn
from dino.schedulers import (
    ConstantScheduleWithLinearWarmup,
    CosineScheduler,
    CosineSchedulerLinearWarmup,
)
from dino.vit import Network


class DINO(pl.LightningModule):
    def __init__(
        self,
        batch_size: int,
        num_steps_per_epoch: int,
        max_epochs: int,
        lr_scheduler_warmup_epochs: int = 10,
        num_global_crops: int = 2,
        num_local_crops: int = 6,
        eval_datamodule: Optional[pl.LightningDataModule] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["eval_datamodule"])
        self.eval_datamodule = eval_datamodule

        self.lr = 0.0005 * batch_size / 256
        self.lambda_ = CosineScheduler(
            initial_value=0.996,
            final_value=1,
            num_epochs=max_epochs,
            num_steps_per_epoch=num_steps_per_epoch,
        )
        self.tps = 0.1
        self.tpt = ConstantScheduleWithLinearWarmup(
            initial_value=0.04,
            final_value=0.07,
            num_warmup_epochs=30,
            num_steps_per_epoch=num_steps_per_epoch,
        )
        self.wd = 0.04  # TODO: cosine schedule from 0.04 to 0.1
        self.center = 0
        self.m = 0.9

        self.param_schedulers = [self.tpt, self.lambda_]

        # initialize student and teacher with vision transformer
        self.student = Network()
        self.teacher = Network()
        self.teacher.load_state_dict(self.student.state_dict())

        # disable teacher's gradients, teacher is updated by EMA of student's weights
        for p in self.teacher.parameters():
            p.requires_grad = False

    def configure_optimizers(self):
        student_optimizer = torch.optim.AdamW(
            self.student.parameters(), weight_decay=0.4, lr=self.lr
        )
        lr_scheduler = CosineSchedulerLinearWarmup(
            student_optimizer,
            num_epochs=self.hparams.max_epochs,
            initial_value=self.lr,
            final_value=1e-6,
            num_steps_per_epoch=self.hparams.num_steps_per_epoch,
            num_warmup_epochs=self.hparams.lr_scheduler_warmup_epochs,
            warmup_start_lr=0,
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [student_optimizer], [scheduler]

    def forward(self, x: Tensor) -> Tensor:
        return self.teacher.backbone(x)

    def common_step(self, *args, **kwargs) -> STEP_OUTPUT:
        x, _ = args
        views, _ = x

        teacher_output = self.teacher(torch.cat(views["global_views"])).chunk(
            self.hparams.num_global_crops
        )
        student_output = self.student(torch.cat(views["global_views"])).chunk(
            self.hparams.num_global_crops
        ) + self.student(torch.cat(views["local_views"])).chunk(
            self.hparams.num_local_crops
        )

        return {
            "loss": self.loss(teacher_output, student_output),
            "teacher_output": teacher_output,
        }

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        output = self.common_step(*args, **kwargs)

        # update center
        with torch.no_grad():
            current_center = torch.cat(output["teacher_output"]).mean(dim=0)
            self.center = self.m * self.center + (1 - self.m) * current_center

        self.log("train_loss", output["loss"], prog_bar=True)

        return output["loss"]

    def validation_step(self, *args, **kwargs):
        output = self.common_step(*args, **kwargs)
        self.log("val_loss", output["loss"], prog_bar=True)
        return output

    def on_validation_epoch_end(self) -> None:
        if self.eval_datamodule:
            teacher_knn_acc = compute_knn(
                self.teacher.backbone, self.eval_datamodule, k=20, device=self.device
            )
            student_knn_acc = compute_knn(
                self.student.backbone, self.eval_datamodule, k=20, device=self.device
            )
            self.log("val_teacher_knn_acc", teacher_knn_acc, prog_bar=True)
            self.log("val_student_knn_acc", student_knn_acc, prog_bar=True)

    @torch.no_grad()
    def on_train_batch_end(self, *args, **kwargs):
        super().on_train_batch_end(*args, **kwargs)

        # update teacher weights with exponential moving average of student weights
        for param_t, param_s in zip(
            self.teacher.parameters(), self.student.parameters()
        ):
            param_t.data.mul_(self.lambda_())
            param_t.add_((1 - self.lambda_()) * param_s.detach().data)

        # step param schedulers
        [scheduler.step() for scheduler in self.param_schedulers]

    def loss(
        self, teacher_output: List[Tensor], student_output: List[Tensor]
    ) -> Tensor:
        loss = 0
        for i, t in enumerate(teacher_output):
            t = t.detach()  # stop gradient
            t = torch.softmax(
                (t - self.center) / self.tpt(), dim=-1
            )  # center + sharpen
            for j, s in enumerate(student_output):
                if i == j:
                    continue
                # Compute cross-entropy loss
                s = torch.softmax(s / self.tps, dim=-1)
                loss += -(t * torch.log(s)).sum(dim=1).mean()

        n = len(teacher_output) * len(student_output) - len(teacher_output)

        return torch.div(loss, n)
