import torch.types
from pytorch_lightning import LightningModule, LightningDataModule
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def compute_knn(
    model: LightningModule,
    datamodule: LightningDataModule,
    k: int = 20,
    device: torch.types.Device = None,
) -> float:
    # run inference pytorch lightning model on datamodule
    x_train = []
    y_train = []
    x_val = []
    y_val = []

    model.eval()
    for x, y in datamodule.train_dataloader():
        out = model(x.to(device))
        # split tensor in dim=0
        out = out.split(1, dim=0)
        x_train += [o.detach().squeeze().cpu().numpy() for o in out]
        y_train += y.detach().cpu().tolist()

    for x, y in datamodule.val_dataloader():
        out = model(x.to(device))
        # split tensor in dim=0
        out = out.split(1, dim=0)
        x_val += [o.detach().squeeze().cpu().numpy() for o in out]
        y_val += y.detach().cpu().tolist()

    estimator = KNeighborsClassifier(n_neighbors=k, weights="distance")
    estimator.fit(x_train, y_train)
    y_val_pred = estimator.predict(x_val)

    return accuracy_score(y_val, y_val_pred)
