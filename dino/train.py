import pytorch_lightning as pl
from pl_bolts.datamodules import ImagenetDataModule

from dino.datamodule import DINODataTransform
from dino.dino_model import DINO

IMAGENET_PATH = "/Users/lourenco/Downloads/imagenet"

MAX_EPOCHS = 6
BATCH_SIZE = 8

if __name__ == "__main__":

    pl.seed_everything(42)

    # data
    datamodule = ImagenetDataModule(IMAGENET_PATH, num_workers=8, batch_size=BATCH_SIZE)

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
        deterministic=True,
        max_epochs=MAX_EPOCHS,
        log_every_n_steps=1,
        precision="bf16",
        limit_train_batches=3,
        limit_val_batches=3,
        limit_test_batches=3,
    )
    trainer.fit(model=dino, datamodule=datamodule)
