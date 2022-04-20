import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    GPUStatsMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.loggers import MLFlowLogger

from dino.datamodule import DINODataTransform, ImagenetDataModule
from dino.dino_model import DINO

IMAGENET_PATH = "/home/pato/imagenet"


MAX_EPOCHS = 100
BATCH_SIZE = 64
NUM_WORKERS = 8
NUM_GLOBAL_CROPS = 2
NUM_LOCAL_CROPS = 6

if __name__ == "__main__":

    pl.seed_everything(42)

    # data
    transform = DINODataTransform(
        num_global_crops=NUM_GLOBAL_CROPS, num_local_crops=NUM_LOCAL_CROPS
    )
    datamodule = ImagenetDataModule(
        IMAGENET_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        transform=transform,
    )
    datamodule.prepare_data()
    datamodule.setup()

    # model
    dino = DINO(
        batch_size=BATCH_SIZE,
        num_steps_per_epoch=len(datamodule.train_dataloader()),
        max_epochs=MAX_EPOCHS,
        num_global_crops=NUM_GLOBAL_CROPS,
        num_local_crops=NUM_LOCAL_CROPS,
    )

    # trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        gpus=1,
        accelerator="gpu",
        deterministic=True,
        precision=16,
        logger=MLFlowLogger(experiment_name="DINO_tiny"),
        callbacks=[
            GPUStatsMonitor(memory_utilization=True, gpu_utilization=True),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(save_top_k=3, monitor="val_loss", mode="min", verbose=True),
            ModelSummary(max_depth=2),
        ],
        limit_train_batches=1.0,
        log_every_n_steps=50,
        limit_val_batches=1.0,
        limit_test_batches=1.0,
    )
    trainer.fit(model=dino, datamodule=datamodule)
