import os

import numpy as np
import tifffile
import torch
from skimage import measure
from torch.utils.data import DataLoader

from model import u_net
from utils import get_aug, ISBI2012, accuracy_recall_precision


@torch.no_grad()
def infer():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = u_net().to(device)

    state = torch.load(f"unet_isbi2012_wc.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()

    _, test_transform = get_aug()
    test_ds = ISBI2012(is_train=False, transform=test_transform)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    preds = []

    total_acc = 0.0
    total_recall = 0.0
    total_precision = 0.0
    for idx, (x, y) in enumerate(test_dl, 1):
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)

        pred = torch.argmax(y_hat, dim=1)
        pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)

        pred = remove_small_objects(pred)
        pred = pred * 255
        preds.append(pred)

        acc, recall, precision = accuracy_recall_precision(y_hat, y)

        total_acc += acc
        total_recall += recall
        total_precision += precision

    print(
        f"acc={total_acc / len(test_dl):.4f} recall={total_recall / len(test_dl):.4f} precision={total_precision / len(test_dl):.4f}")

    volume_pred = np.stack(preds, axis=0)  # (D,H_out,W_out)
    save_path = os.path.join("ISBI-2012-challenge", f"test-volume-pred.tif")
    tifffile.imwrite(save_path, volume_pred)
    print("saved in:", save_path)


def remove_small_objects(mask, min_size=1):
    # mask: (H,W), 0/1
    labeled = measure.label(mask, connectivity=1)
    out = np.zeros_like(mask)
    for region in measure.regionprops(labeled):
        if region.area >= min_size:
            out[labeled == region.label] = 1
    return out


if __name__ == "__main__":
    infer()
