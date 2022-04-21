from typing import Dict, List, Union

import torch.nn
from PIL.Image import Image
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms as T, datasets
from torchvision.transforms import InterpolationMode


eval_transform = T.Compose(
    [
        T.Resize(size=256, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(size=224),
        T.ToTensor(),
        imagenet_normalization(),
    ]
)


class DINODataTransform(torch.nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        global_crop_size: int = 224,
        local_crop_size: int = 96,
        num_global_crops: int = 2,
        num_local_crops: int = 6,
    ) -> None:
        super(DINODataTransform, self).__init__()
        self.image_size = image_size
        self.crop_size = global_crop_size
        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops

        # Multi-crop with bicubic interpolation
        global_crop = T.RandomResizedCrop(
            size=global_crop_size,
            scale=(0.3, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        )
        local_crop = T.RandomResizedCrop(
            size=local_crop_size,
            scale=(0.05, 0.3),
            interpolation=InterpolationMode.BICUBIC,
        )

        # BYOL augmentations
        data_transforms = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                transforms=[
                    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
                ],
                p=0.8,
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=(23, 23))], p=1.0),
            T.RandomSolarize(threshold=0.5, p=0),
            T.ToTensor(),
            imagenet_normalization(),
        ]

        # Assemble transforms and export to TorchScript
        self.global_transform = T.Compose([global_crop] + data_transforms)
        self.local_transform = T.Compose([local_crop] + data_transforms)

    def __call__(self, im: Image) -> Dict[str, List[Tensor]]:
        global_views = [self.global_transform(im) for _ in range(self.num_global_crops)]
        local_views = [self.local_transform(im) for _ in range(self.num_local_crops)]
        return {"global_views": global_views, "local_views": local_views}


class ImagenetDataModule(LightningDataModule):
    def __init__(
        self,
        image_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        drop_last: bool,
        transform: Union[T.Compose, torch.nn.Module],
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["transform"])
        self.train_dataset = datasets.ImageFolder(
            self.hparams.image_dir + "/train", transform=transform
        )
        self.val_dataset = datasets.ImageFolder(
            self.hparams.image_dir + "/val", transform=transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
        )
