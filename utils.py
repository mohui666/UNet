import os.path

import albumentations as A
import cv2
import numpy as np
import tifffile
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import Dataset


def crop(X, tar_shape):
    H, W = X.shape[-2:]
    targetH, targetW = tar_shape[-2:]
    assert H >= targetH and W >= targetW
    top = (H - targetH) // 2
    left = (W - targetW) // 2
    return X[..., top:top + targetH, left:left + targetW]


def get_aug():
    train_tf = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.0, rotate_limit=180,
                           interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ElasticTransform(alpha=30, sigma=10, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT_101,
                           p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
        A.Normalize(mean=(0.0,), std=(1.0,)),
        ToTensorV2()
    ])
    test_tf = A.Compose([
        A.Normalize(mean=(0.0,), std=(1.0,)),
        ToTensorV2()
    ])
    return train_tf, test_tf


class ISBI2012(Dataset):
    def __init__(self, path="ISBI-2012-challenge", is_train=True, transform=None, slices=None):
        self.X = tifffile.imread(os.path.join(path, "train-volume.tif")) if is_train else tifffile.imread(
            os.path.join(path, "test-volume.tif"))
        self.Y = tifffile.imread(os.path.join(path, "train-labels.tif")) if is_train else tifffile.imread(
            os.path.join(path, "test-labels.tif"))
        if (self.Y is not None):
            self.Y = (self.Y > 0).astype("uint8")
        self.transform = transform
        self.slices = slices if slices is not None else np.arange(self.X.shape[0])

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        d = self.slices[idx]
        img = self.X[d]
        if self.Y is None:
            if self.transform:
                out = self.transform(image=img[..., None])
                return out["image"]
            return torch.tensor(img[None, ...], dtype=torch.float32)

        mask = self.Y[d]
        if self.transform:
            out = self.transform(image=img[..., None], mask=mask)
            img, mask = out["image"], out["mask"]
            return img.float(), mask.long()

        img = torch.tensor(img[None, ...], dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)
        return img, mask


def kfold(D=30):
    slices = np.arange(D)
    kf = KFold(n_splits=5)

    folds = []
    for train_idx, val_idx in kf.split(slices):
        folds.append((train_idx, val_idx))

    return folds


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def accuracy_recall_precision(y_hat, y):
    y = crop(y, y_hat.shape)
    y_hat = torch.argmax(y_hat, dim=1)
    acc = (y == y_hat).sum().item() / (y.shape[-1] * y.shape[-2])
    tp = ((y == y_hat) & (y > 0)).sum().item()
    fp = ((y != y_hat) & (y_hat > 0)).sum().item()
    fn = ((y != y_hat) & (y > 0)).sum().item()

    recall = tp / (tp + fp)
    precision = tp / (tp + fn)
    return acc, recall, precision
