"""
工具函数模块

本模块提供项目所需的各种辅助功能：
1. 数据集加载和预处理（ISBI2012, TiffDateset）
2. 数据增强配置（get_aug）
3. 图像处理（crop）
4. 交叉验证划分（kfold）
5. 模型权重初始化（init_weights）
6. 评估指标计算（accuracy_recall_precision）
"""

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
    """
    中心裁剪函数
    
    从输入张量的中心区域裁剪出目标尺寸的子区域。
    主要用于：
    1. 将编码器特征图裁剪到与解码器特征图相同尺寸（跳跃连接）
    2. 将标签裁剪到与模型输出相同尺寸
    
    Args:
        X (torch.Tensor): 输入张量，shape: (..., H, W)
        tar_shape (tuple/list): 目标形状，至少包含最后两个维度 (H_target, W_target)
    
    Returns:
        torch.Tensor: 裁剪后的张量，shape: (..., H_target, W_target)
    
    示例：
        >>> x = torch.randn(1, 64, 568, 568)
        >>> y = torch.randn(1, 64, 392, 392)
        >>> x_cropped = crop(x, y.shape)  # (1, 64, 392, 392)
    """
    H, W = X.shape[-2:]
    targetH, targetW = tar_shape[-2:]
    assert H >= targetH and W >= targetW, f"输入尺寸 ({H}, {W}) 必须大于等于目标尺寸 ({targetH}, {targetW})"
    
    # 计算裁剪起始位置（中心裁剪）
    top = (H - targetH) // 2
    left = (W - targetW) // 2
    
    return X[..., top:top + targetH, left:left + targetW]


def get_aug():
    """
    获取数据增强配置
    
    返回训练集和测试集的数据增强 pipeline。
    训练集使用强数据增强以应对小样本问题，测试集仅进行归一化。
    
    训练集增强包括：
    - ShiftScaleRotate: 随机平移、旋转（±180°）
    - HorizontalFlip: 水平翻转（50%概率）
    - VerticalFlip: 垂直翻转（50%概率）
    - ElasticTransform: 弹性形变，模拟组织的自然变形
    - RandomBrightnessContrast: 亮度和对比度调整
    - Normalize: 归一化到均值0，标准差1
    
    Returns:
        tuple: (训练集增强, 测试集增强)
    """
    train_tf = A.Compose([
        # 旋转和平移增强
        A.ShiftScaleRotate(
            shift_limit=0.05,      # 平移范围 ±5%
            scale_limit=0.0,       # 不进行缩放
            rotate_limit=180,      # 旋转范围 ±180°
            interpolation=cv2.INTER_CUBIC,          # 三次插值
            border_mode=cv2.BORDER_REFLECT_101,     # 边界反射填充
            p=1.0                  # 100% 应用
        ),
        # 翻转增强
        A.HorizontalFlip(p=0.5),   # 50% 概率水平翻转
        A.VerticalFlip(p=0.5),     # 50% 概率垂直翻转
        # 弹性形变（模拟组织变形）
        A.ElasticTransform(
            alpha=30,              # 形变强度
            sigma=10,              # 高斯核标准差
            interpolation=cv2.INTER_CUBIC,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7                  # 70% 概率应用
        ),
        # 亮度和对比度调整
        A.RandomBrightnessContrast(
            brightness_limit=0.1,  # 亮度调整范围 ±10%
            contrast_limit=0.1,    # 对比度调整范围 ±10%
            p=1.0
        ),
        # 归一化
        A.Normalize(mean=(0.0,), std=(1.0,)),
        # 转换为 PyTorch Tensor
        ToTensorV2()
    ])
    
    # 测试集仅进行归一化
    test_tf = A.Compose([
        A.Normalize(mean=(0.0,), std=(1.0,)),
        ToTensorV2()
    ])
    
    return train_tf, test_tf


class ISBI2012(Dataset):
    """
    ISBI 2012 EM Segmentation Challenge 数据集类
    
    加载和处理 ISBI 2012 电子显微镜图像分割数据集。
    数据集包含 30 张 512×512 的 EM 图像和对应的细胞膜标签。
    
    Args:
        path (str): 数据集根目录，默认 "ISBI-2012-challenge"
        is_train (bool): True 加载训练集，False 加载测试集
        transform: Albumentations 数据增强 pipeline
        slices (array-like, optional): 要加载的切片索引列表，None 表示加载全部
    
    数据格式：
        - train-volume.tif: (30, 512, 512) 训练图像
        - train-labels.tif: (30, 512, 512) 训练标签，0=背景，255=前景
        - test-volume.tif: (30, 512, 512) 测试图像
        - test-labels.tif: (30, 512, 512) 测试标签（可选）
    
    返回：
        - 有标签时: (image, label) - (C, H, W) 图像和 (H, W) 标签
        - 无标签时: image - (C, H, W) 图像
    """
    def __init__(self, path="ISBI-2012-challenge", is_train=True, transform=None, slices=None):
        # 加载图像和标签
        self.X = tifffile.imread(os.path.join(path, "train-volume.tif")) if is_train else tifffile.imread(
            os.path.join(path, "test-volume.tif"))
        self.Y = tifffile.imread(os.path.join(path, "train-labels.tif")) if is_train else tifffile.imread(
            os.path.join(path, "test-labels.tif"))
        
        # 将标签从 0/255 转换为 0/1
        if self.Y is not None:
            self.Y = (self.Y > 0).astype("uint8")
        
        self.transform = transform
        # 使用指定的切片索引，或默认使用全部切片
        self.slices = slices if slices is not None else np.arange(self.X.shape[0])

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx (int): 样本索引
        
        Returns:
            有标签: (image, mask) - 增强后的图像和标签
            无标签: image - 增强后的图像
        """
        d = self.slices[idx]
        img = self.X[d]
        
        # 测试集可能没有标签
        if self.Y is None:
            if self.transform:
                out = self.transform(image=img[..., None])
                return out["image"]
            return torch.tensor(img[None, ...], dtype=torch.float32)

        mask = self.Y[d]
        if self.transform:
            # 同时对图像和标签应用相同的增强（保持空间对应关系）
            out = self.transform(image=img[..., None], mask=mask)
            img, mask = out["image"], out["mask"]
            return img.float(), mask.long()

        img = torch.tensor(img[None, ...], dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)
        return img, mask


class TiffDateset(Dataset):
    """
    通用 TIFF 格式数据集类
    
    支持自定义路径的 TIFF 图像加载，适用于不同的医学图像数据集。
    
    Args:
        train_volume_path (str): 训练/测试图像路径
        train_label_path (str): 训练标签路径
        test_volume_path (str): 测试图像路径
        test_label_path (str): 测试标签路径
        is_train (bool): True 使用训练路径，False 使用测试路径
        transform: 数据增强 pipeline
        slices (array-like, optional): 切片索引列表
    """
    def __init__(self, train_volume_path=None, train_label_path=None, test_volume_path=None, test_label_path=None,
                 is_train=True, transform=None, slices=None):
        self.X = tifffile.imread(train_volume_path) if is_train else tifffile.imread(train_volume_path)
        self.Y = tifffile.imread(train_label_path) if is_train else tifffile.imread(test_label_path)
        self.transform = transform
        self.slices = slices if slices is not None else np.arange(self.X.shape[0])

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        d = self.slices[idx]
        img = self.X[d]
        # 确保图像是 3 维的 (H, W, C)
        if len(img.shape) == 2:
            img = img[..., None]

        assert len(img.shape) == 3, "图像必须是 3 维的 (H, W, C)"

        if self.Y is None:
            if self.transform:
                out = self.transform(image=img)
                return out["image"]
            return torch.tensor(img, dtype=torch.float32)

        mask = self.Y[d]
        if self.transform:
            out = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
            return img.float(), mask.long()

        img = torch.tensor(img, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)
        return img, mask


def kfold(D=30):
    """
    生成 K-Fold 交叉验证划分
    
    将数据集划分为 K 折（默认 5 折），用于交叉验证训练。
    
    Args:
        D (int): 数据集大小（切片数量），默认 30
    
    Returns:
        list: K 个元组的列表，每个元组包含 (训练索引, 验证索引)
    
    示例：
        >>> folds = kfold(30)
        >>> len(folds)  # 5 折
        5
        >>> train_idx, val_idx = folds[0]
        >>> len(train_idx), len(val_idx)  # (24, 6)
        (24, 6)
    """
    slices = np.arange(D)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 添加 shuffle 和 random_state 以保证可复现

    folds = []
    for train_idx, val_idx in kf.split(slices):
        folds.append((train_idx, val_idx))

    return folds


def init_weights(m):
    """
    权重初始化函数
    
    使用 Kaiming 初始化（He 初始化）对模型权重进行初始化。
    专为 ReLU 激活函数设计，有助于保持前向传播和反向传播的方差稳定。
    
    Args:
        m (nn.Module): 神经网络层
    
    初始化规则：
    - Conv2d/ConvTranspose2d: Kaiming normal 初始化权重，零初始化偏置
    - BatchNorm2d: 权重初始化为 1，偏置初始化为 0
    
    使用方法：
        >>> model = u_net()
        >>> model.apply(init_weights)
    """
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
    """
    计算像素级分类指标
    
    计算准确率、召回率和精确率，用于评估分割质量。
    
    Args:
        y_hat (torch.Tensor): 模型输出的 logits，shape: (B, C, H, W)
        y (torch.Tensor): 真实标签，shape: (B, H, W)
    
    Returns:
        tuple: (accuracy, recall, precision)
        - accuracy: 正确分类的像素比例
        - recall: TP / (TP + FN)，找到了多少真实前景
        - precision: TP / (TP + FP)，预测为前景的有多少是对的
    
    注意：
        为了避免除零错误，分母加上小常数 eps
    """
    # 裁剪标签以匹配输出尺寸
    y = crop(y, y_hat.shape)
    
    # 获取预测类别（argmax）
    y_hat = torch.argmax(y_hat, dim=1)
    
    # 准确率：正确预测的像素比例
    acc = (y == y_hat).sum().item() / (y.shape[-1] * y.shape[-2])
    
    # 计算 TP, FP, FN
    tp = ((y == y_hat) & (y > 0)).sum().item()  # 真阳性：正确预测为前景
    fp = ((y != y_hat) & (y_hat > 0)).sum().item()  # 假阳性：错误预测为前景
    fn = ((y != y_hat) & (y > 0)).sum().item()  # 假阴性：错误预测为背景

    # 召回率和精确率（添加小常数避免除零）
    eps = 1e-8
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    
    return acc, recall, precision
