from typing import Dict, List

from PIL.Image import Image
from pl_bolts.datamodules import ImagenetDataModule
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from torch import Tensor
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

IMAGENET_PATH = "/Users/lourenco/Downloads/imagenet"


class DINODataTransform(object):
    def __init__(
        self,
        image_size: int = 224,
        global_crop_size: int = 224,
        local_crop_size: int = 96,
        num_global_crops: int = 2,
        num_local_crops: int = 6,
    ) -> None:
        self.image_size = image_size
        self.crop_size = global_crop_size
        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops

        global_crop = T.RandomResizedCrop(
            size=global_crop_size,
            scale=(0.3, 1),
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
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=(23, 23))], p=1.0),
            T.RandomSolarize(threshold=0.5, p=0),
            T.ToTensor(),
            imagenet_normalization(),
        ]

        # Multi-crop with bicubic interpolation
        self.global_transform = T.Compose([global_crop] + data_transforms)
        self.local_transform = T.Compose([local_crop] + data_transforms)

    def __call__(self, im: Image) -> Dict[str, List[Tensor]]:
        global_views = [self.global_transform(im) for _ in range(self.num_global_crops)]
        local_views = [self.local_transform(im) for _ in range(self.num_local_crops)]
        return {"global_views": global_views, "local_views": local_views}


if __name__ == "__main__":
    dm = ImagenetDataModule(
        IMAGENET_PATH,
        meta_dir=None,
        num_imgs_per_val_class=50,
        image_size=224,
        num_workers=0,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )

    dm.prepare_data()
    dm.setup()
    print(dm)
