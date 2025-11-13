import tifffile


def get_wc(classes=2, eps=1e-12):
    y = tifffile.imread("ISBI-2012-challenge/train-labels.tif")

    if set(np.unique(y).tolist()) <= {0, 255}:
        y = (y > 0).astype(np.int64)

    num_classes = classes
    counts = np.bincount(y.reshape(-1), minlength=num_classes).astype(np.float64)

    wc = counts.sum() / (num_classes * counts + eps)

    class_weights = torch.from_numpy(wc)
    return class_weights


import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi


def make_weight_map_per_sample(label_np, wc_np, w0=10.0, sigma=5.0):
    """
    label_np: (H,W) numpy int64, 0..C-1
    wc_np   : (C,)  numpy float32
    返回: (H,W) torch.float32（在 CPU 上）
    """
    # 类别权重底图
    wc_map = np.take(wc_np.astype(np.float32), label_np.astype(np.int64))  # (H,W)

    # 前景/背景二值图
    fg = (label_np > 0).astype(np.uint8)
    bg = 1 - fg

    # 距离变换
    d_to_fg = ndi.distance_transform_edt(bg)  # 背景里到最近前景
    d_to_bg = ndi.distance_transform_edt(fg)  # 前景里到最近背景
    dsum = d_to_fg + d_to_bg

    # U-Net 边界项
    border = w0 * np.exp(-(dsum ** 2) / (2.0 * (sigma ** 2))).astype(np.float32)

    w = wc_map + border  # (H,W) float32
    return torch.from_numpy(w)  # CPU tensor


def per_pixel_weight(label, wc, w0=10.0, sigma=5.0):
    """
    label: (B,H,W) long tensor（在 cuda 或 cpu 上）
    wc   : (C,)    float tensor（在 cuda 或 cpu 上）
    返回: (B,H,W) float tensor，和 label 在同一个 device
    """
    device = label.device

    # 搬到 CPU + 变 numpy
    label_np = label.detach().cpu().numpy()  # (B,H,W)
    wc_np = wc.detach().cpu().numpy()  # (C,)

    ws = []
    for i in range(label_np.shape[0]):
        w_i = make_weight_map_per_sample(label_np[i], wc_np, w0=w0, sigma=sigma)  # (H,W) CPU
        ws.append(w_i)
    w = torch.stack(ws, dim=0)  # (B,H,W) CPU

    return w.to(device)  # 搬回原来的 device


def dice_score(y_hat, y, eps=1e-6):
    probs = F.softmax(y_hat, dim=1)
    pred_fg = (probs[:, 1] > 0.9).float()
    target_fg = (y == 1).float()

    inter = (pred_fg * target_fg).sum(dim=(1, 2))
    union = pred_fg.sum(dim=(1, 2)) + target_fg.sum(dim=(1, 2))

    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()
