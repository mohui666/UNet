"""
损失函数和评估指标模块

本模块实现了 U-Net 论文中提到的边界感知加权损失函数，以及常用的评估指标。

主要功能：
1. get_wc(): 计算类别权重，平衡类别不均衡问题
2. per_pixel_weight(): 生成像素级权重图，强调边界区域
3. dice_score(): 计算 Dice 系数，评估分割质量
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
import tifffile


def get_wc(classes=2, eps=1e-12):
    """
    计算类别权重（Class Weights）
    
    用于平衡类别不均衡问题。在医学图像分割中，前景（如细胞膜）像素通常远少于背景像素，
    导致模型倾向于预测背景。通过给少数类更高的权重，可以缓解这个问题。
    
    计算公式：
        wc[i] = total_pixels / (num_classes × count[i])
    
    Args:
        classes (int): 类别数量，默认 2（背景/前景）
        eps (float): 防止除零的小数
    
    Returns:
        torch.Tensor: 类别权重，shape: (classes,)
    
    示例：
        如果有 90% 背景、10% 前景，则前景权重约为背景权重的 9 倍
    """
    # 读取训练标签
    y = tifffile.imread("ISBI-2012-challenge/train-labels.tif")

    # 将 0/255 标签转换为 0/1
    if set(np.unique(y).tolist()) <= {0, 255}:
        y = (y > 0).astype(np.int64)

    num_classes = classes
    # 统计每个类别的像素数量
    counts = np.bincount(y.reshape(-1), minlength=num_classes).astype(np.float64)

    # 计算权重：总像素数 / (类别数 × 该类别像素数)
    wc = counts.sum() / (num_classes * counts + eps)

    class_weights = torch.from_numpy(wc)
    return class_weights


def make_weight_map_per_sample(label_np, wc_np, w0=10.0, sigma=5.0):
    """
    为单个样本生成像素级权重图
    
    实现 U-Net 论文中的边界感知权重图，让模型更关注难以分割的边界区域。
    
    权重计算公式：
        w(x) = wc(x) + w0 × exp(-(d1(x) + d2(x))² / (2σ²))
    
    其中：
    - wc(x): 像素 x 所属类别的权重
    - d1(x): 像素 x 到最近前景对象的距离
    - d2(x): 像素 x 到第二近前景对象的距离
    - w0, σ: 超参数，控制边界权重的强度和衰减速度
    
    Args:
        label_np (np.ndarray): 标签图，shape: (H, W)，取值 0..C-1
        wc_np (np.ndarray): 类别权重，shape: (C,)
        w0 (float): 边界权重因子，默认 10.0
        sigma (float): 高斯核标准差，控制权重衰减，默认 5.0
    
    Returns:
        torch.Tensor: 权重图，shape: (H, W)，在 CPU 上
    
    示例：
        两个细胞之间的间隙会获得很高的权重，促使模型准确分割边界
    """
    # 1. 根据标签生成类别权重底图
    wc_map = np.take(wc_np.astype(np.float32), label_np.astype(np.int64))  # (H,W)

    # 2. 生成前景/背景二值图
    fg = (label_np > 0).astype(np.uint8)  # 前景：细胞膜
    bg = 1 - fg                           # 背景

    # 3. 计算距离变换
    # d_to_fg: 背景中每个像素到最近前景像素的距离
    # d_to_bg: 前景中每个像素到最近背景像素的距离
    d_to_fg = ndi.distance_transform_edt(bg)
    d_to_bg = ndi.distance_transform_edt(fg)
    dsum = d_to_fg + d_to_bg

    # 4. 计算边界权重（高斯衰减）
    # 在两个前景对象之间的边界处，dsum 较小，权重较高
    border = w0 * np.exp(-(dsum ** 2) / (2.0 * (sigma ** 2))).astype(np.float32)

    # 5. 合并类别权重和边界权重
    w = wc_map + border  # (H,W) float32
    return torch.from_numpy(w)  # 返回 CPU tensor


def per_pixel_weight(label, wc, w0=10.0, sigma=5.0):
    """
    为一批样本生成像素级权重图
    
    对批次中的每个样本调用 make_weight_map_per_sample，然后堆叠成批次。
    
    Args:
        label (torch.Tensor): 标签张量，shape: (B, H, W)，long 类型
        wc (torch.Tensor): 类别权重，shape: (C,)，float 类型
        w0 (float): 边界权重因子
        sigma (float): 高斯核标准差
    
    Returns:
        torch.Tensor: 权重图，shape: (B, H, W)，与 label 在同一 device 上
    
    注意：
        为了使用 scipy 的距离变换，需要先将数据移到 CPU，计算完成后再移回原 device
    """
    device = label.device

    # 将张量转换为 numpy 数组（在 CPU 上）
    label_np = label.detach().cpu().numpy()  # (B,H,W)
    wc_np = wc.detach().cpu().numpy()        # (C,)

    # 对每个样本生成权重图
    ws = []
    for i in range(label_np.shape[0]):
        w_i = make_weight_map_per_sample(label_np[i], wc_np, w0=w0, sigma=sigma)  # (H,W) CPU
        ws.append(w_i)
    w = torch.stack(ws, dim=0)  # (B,H,W) CPU

    return w.to(device)  # 移回原来的 device


def dice_score(y_hat, y, eps=1e-6):
    """
    计算 Soft Dice 系数
    
    Dice 系数（也称 F1 Score）是分割任务中常用的评估指标，衡量预测与真实标签的重叠程度。
    
    计算公式：
        Dice = (2 × |预测 ∩ 真实|) / (|预测| + |真实|)
    
    取值范围：[0, 1]，越接近 1 表示分割越准确。
    
    Args:
        y_hat (torch.Tensor): 模型输出的 logits，shape: (B, C, H, W)
        y (torch.Tensor): 真实标签，shape: (B, H, W)，long 类型
        eps (float): 防止除零的小数
    
    Returns:
        float: 批次的平均 Dice 系数
    
    实现细节：
        - 使用 softmax 将 logits 转换为概率
        - 只计算前景类别（class 1）的 Dice
        - 使用阈值 0.9 将概率二值化
    """
    # 1. 将 logits 转换为概率分布
    probs = F.softmax(y_hat, dim=1)  # (B, C, H, W)
    
    # 2. 提取前景类别的概率，并二值化
    pred_fg = (probs[:, 1] > 0.9).float()  # (B, H, W)
    target_fg = (y == 1).float()           # (B, H, W)

    # 3. 计算交集和并集
    inter = (pred_fg * target_fg).sum(dim=(1, 2))      # (B,)
    union = pred_fg.sum(dim=(1, 2)) + target_fg.sum(dim=(1, 2))  # (B,)

    # 4. 计算 Dice 系数
    dice = (2 * inter + eps) / (union + eps)  # (B,)
    
    return dice.mean().item()  # 返回批次平均值
