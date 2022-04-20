from torch import nn as nn, Tensor
from torch.nn.functional import normalize
from vit_pytorch import ViT


class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_size: int = 384,
        hidden_size: int = 2048,
        bottleneck_size: int = 256,
        k: int = 65536,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, bottleneck_size),
        )
        # weight normalized fully connected layer
        self.linear_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_size, k, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x)
        x = normalize(x, p=2, dim=1)  # L2 normalize
        x = self.linear_layer(x)
        return x


class Network(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.num_classes = num_classes
        # Vision transformer - tiny ViT
        self.backbone = ViT(
            image_size=224,
            patch_size=16,
            num_classes=1,
            dim=384,
            depth=12,
            heads=6,
            mlp_dim=768,
            pool="cls",
            channels=3,
        )
        self.backbone.mlp_head = nn.Identity()
        self.projection_head = ProjectionHead(
            input_size=384,
            hidden_size=2048,
            bottleneck_size=256,
            k=2048,
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.projection_head(y)
        return y
