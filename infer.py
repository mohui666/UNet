"""
推理脚本模块

本模块用于加载训练好的模型对测试集进行推理。

主要功能：
1. 加载训练好的模型权重
2. 对测试集进行批量预测
3. 后处理：移除小连通域、二值化输出
4. 保存预测结果为 TIFF 格式
5. 计算测试集评估指标（如果有标签）
"""

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
    """
    推理主函数
    
    执行流程：
    1. 加载训练好的模型
    2. 加载测试集数据
    3. 对每张图像进行预测
    4. 后处理：移除小连通域，二值化
    5. 保存结果为 TIFF 文件
    6. 如果有标签，计算评估指标
    
    输出：
        - test-volume-pred.tif: 预测的分割图 (D, H, W)，像素值 0/255
        - 控制台输出：准确率、召回率、精确率
    """
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==================== 加载模型 ====================
    model = u_net().to(device)

    # 加载训练好的权重
    state = torch.load(f"unet_isbi2012_wc.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()  # 切换到评估模式

    # ==================== 数据准备 ====================
    _, test_transform = get_aug()  # 只使用测试集的增强（归一化）
    test_ds = ISBI2012(is_train=False, transform=test_transform)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    preds = []  # 存储所有预测结果

    # 评估指标累积
    total_acc = 0.0
    total_recall = 0.0
    total_precision = 0.0
    
    # ==================== 推理循环 ====================
    for idx, (x, y) in enumerate(test_dl, 1):
        x = x.to(device)
        y = y.to(device)
        
        # 前向传播
        y_hat = model(x)

        # 后处理：获取预测类别
        pred = torch.argmax(y_hat, dim=1)  # (B, H, W)
        pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)  # 转为 numpy

        # 移除小连通域（去噪）
        pred = remove_small_objects(pred)
        
        # 映射为 0/255 的二值图
        pred = pred * 255
        preds.append(pred)

        # 如果有标签，计算评估指标
        acc, recall, precision = accuracy_recall_precision(y_hat, y)

        total_acc += acc
        total_recall += recall
        total_precision += precision

    # ==================== 打印评估结果 ====================
    print(
        f"acc={total_acc / len(test_dl):.4f} "
        f"recall={total_recall / len(test_dl):.4f} "
        f"precision={total_precision / len(test_dl):.4f}"
    )

    # ==================== 保存结果 ====================
    volume_pred = np.stack(preds, axis=0)  # (D, H_out, W_out)
    save_path = os.path.join("ISBI-2012-challenge", f"test-volume-pred.tif")
    tifffile.imwrite(save_path, volume_pred)
    print("saved in:", save_path)


def remove_small_objects(mask, min_size=1):
    """
    移除小连通域
    
    使用连通域分析去除面积小于阈值的区域，主要用于去除噪声。
    
    Args:
        mask (np.ndarray): 二值分割图，shape: (H, W)，取值 0/1
        min_size (int): 最小连通域大小（像素数），默认 1
    
    Returns:
        np.ndarray: 清理后的分割图，shape: (H, W)，取值 0/1
    
    原理：
        1. 使用连通域标记算法标记每个连通区域
        2. 计算每个区域的面积
        3. 保留面积 >= min_size 的区域，丢弃小区域
    
    示例：
        >>> mask = np.array([[1, 0, 0, 1],
        ...                  [0, 0, 0, 1],
        ...                  [1, 1, 0, 0]])
        >>> cleaned = remove_small_objects(mask, min_size=2)
        # 左上角的单个像素被移除，其他保留
    """
    # 连通域标记（connectivity=1 表示 4-邻域）
    labeled = measure.label(mask, connectivity=1)
    
    # 初始化输出
    out = np.zeros_like(mask)
    
    # 遍历每个连通域
    for region in measure.regionprops(labeled):
        # 保留面积足够大的区域
        if region.area >= min_size:
            out[labeled == region.label] = 1
    
    return out


if __name__ == "__main__":
    infer()
