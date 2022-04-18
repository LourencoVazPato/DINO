import pytorch_lightning as pl
from pl_bolts.datamodules import ImagenetDataModule
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    GPUStatsMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.profiler import SimpleProfiler

from dino.datamodule import DINODataTransform
from dino.dino_model import DINO

IMAGENET_PATH = "/home/pato/imagenet"


MAX_EPOCHS = 1
BATCH_SIZE = 128

if __name__ == "__main__":

    pl.seed_everything(42)

    # data
    datamodule = ImagenetDataModule(IMAGENET_PATH, num_workers=6, batch_size=BATCH_SIZE)

    # transforms
    datamodule.train_transforms = DINODataTransform()
    datamodule.val_transforms = DINODataTransform()
    datamodule.prepare_data()
    datamodule.setup()

    # model
    dino = DINO(
        batch_size=BATCH_SIZE,
        num_steps_per_epoch=len(datamodule.train_dataloader()),
        max_epochs=MAX_EPOCHS,
    )

    # trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        gpus=1,
        deterministic=True,
        precision=16,
        logger=MLFlowLogger(experiment_name="DINO_tiny"),
        callbacks=[
            GPUStatsMonitor(memory_utilization=True, gpu_utilization=True),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(save_top_k=5, monitor="val_loss", mode="min", verbose=True),
        ],
        profiler=SimpleProfiler(),
        limit_train_batches=10,
        log_every_n_steps=50,
        limit_val_batches=10,
        limit_test_batches=1.0,
    )
    trainer.fit(model=dino, datamodule=datamodule)
