import argparse

import torch

from dino.datamodule import ImagenetDataModule, eval_transform
from dino.dino_model import DINO
from dino.knn import compute_knn
from dino.train import IMAGENETTE_PATH

BATCH_SIZE = 128
NUM_WORKERS = 8

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=IMAGENETTE_PATH)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    eval_datamodule = ImagenetDataModule(
        image_dir=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        transform=eval_transform,
    )

    # Load PyTorch Lightning model checkpoint
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DINO.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.to(device)
    acc = compute_knn(model, eval_datamodule, k=20, device=device)
    print(f"Accuracy: {acc:.1%}")
