import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.loggers import MLFlowLogger

from dino.datamodule import DINODataTransform, ImagenetDataModule, eval_transform
from dino.dino_model import DINO

MAX_EPOCHS = 100
BATCH_SIZE = 64
NUM_WORKERS = 8
NUM_GLOBAL_CROPS = 2
NUM_LOCAL_CROPS = 6
SEED = 42

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--num_global_crops", type=int, default=NUM_GLOBAL_CROPS)
    parser.add_argument("--num_local_crops", type=int, default=NUM_LOCAL_CROPS)
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # data
    transform = DINODataTransform(
        num_global_crops=args.num_global_crops, num_local_crops=args.num_local_crops
    )
    datamodule = ImagenetDataModule(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        transform=transform,
    )
    datamodule.prepare_data()
    datamodule.setup()

    eval_datamodule = ImagenetDataModule(
        image_dir=args.dataset,
        batch_size=2 * args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        transform=eval_transform,
    )

    # model
    dino = DINO(
        batch_size=args.batch_size,
        num_steps_per_epoch=len(datamodule.train_dataloader()),
        max_epochs=args.max_epochs,
        num_global_crops=args.num_global_crops,
        num_local_crops=args.num_local_crops,
        eval_datamodule=eval_datamodule,
    )

    # trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=1,
        accelerator="gpu",
        deterministic=True,
        precision=16,
        gradient_clip_val=3.0,
        logger=MLFlowLogger(experiment_name="DINO_ViT_S_Imagenette"),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(save_top_k=5, monitor="val_loss", mode="min", verbose=True),
            ModelSummary(max_depth=2),
        ],
        log_every_n_steps=25,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
    )
    trainer.fit(model=dino, datamodule=datamodule)
