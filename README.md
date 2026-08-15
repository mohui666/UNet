# 📘 UNet for EM Membrane Segmentation (ISBI 2012)

本项目实现了基于 U-Net 的电子显微镜 (EM) 膜结构分割模型，面向 ISBI 2012 EM Segmentation Challenge 数据集。  
包含完整的数据加载、增强、模型、损失、训练与推理流程。

## 📖 仓库概述

这是一个完整的深度学习图像分割项目，使用经典的 U-Net 架构对电子显微镜图像中的细胞膜进行语义分割。项目的主要特点是忠实还原了原始 U-Net 论文（Ronneberger et al., 2015）的设计思想，包括：
- 对称的编码器-解码器结构
- 跳跃连接（Skip Connections）融合多尺度特征
- 基于距离变换的边界加权损失
- 强数据增强策略应对小样本问题

---

## 🌟 Features

- 原版 U-Net (Ronneberger et al., 2015) 结构实现  
- ISBI 2012 EM segmentation 数据集全流程支持  
- 强数据增强：旋转、弹性形变、翻转、亮度对比度调整  
- 边界感知加权损失（基于距离变换的 weight map）  
- 训练使用加权 CrossEntropyLoss + Soft Dice 评估  
- 支持 Early Stopping 与学习率调度器 StepLR  
- 推理输出 test-volume-pred.tif，并进行小连通域移除

---

## 📂 项目结构详解

```
UNet/
│
├── model.py                # U-Net 模型核心实现
│                          # - 定义了完整的 U-Net 网络结构
│                          # - 包含编码器（下采样）和解码器（上采样）路径
│                          # - 实现跳跃连接（skip connections）
│
├── train.py                # 训练脚本（含调度器与早停）
│                          # - train(): 使用全部数据训练
│                          # - train_val(): 使用 K-Fold 交叉验证训练
│                          # - evaluate(): 验证函数
│                          # - 支持 Early Stopping 和学习率调度
│
├── infer.py                # 推理脚本
│                          # - 加载训练好的模型权重
│                          # - 对测试集进行预测
│                          # - 后处理：移除小连通域、二值化输出
│
├── utils.py                # 工具函数集合
│                          # - ISBI2012: 数据集加载类
│                          # - get_aug(): 数据增强配置
│                          # - crop(): 中心裁剪函数（用于对齐 feature map）
│                          # - kfold(): K 折交叉验证划分
│                          # - init_weights(): 权重初始化
│                          # - accuracy_recall_precision(): 评估指标计算
│
├── loss.py                 # 损失函数与评估指标
│                          # - get_wc(): 计算类别权重
│                          # - per_pixel_weight(): 生成像素级权重图
│                          # - make_weight_map_per_sample(): 边界加权实现
│                          # - dice_score(): Dice 系数计算
│
├── test.py                 # 模型结构/前向传播测试
│                          # - 快速验证模型是否能正常运行
│
└── ISBI-2012-challenge/    # 本地数据集目录（不在仓库中）
    ├── train-volume.tif    # 训练图像 (30, 512, 512)
    ├── train-labels.tif    # 训练标签 (30, 512, 512)
    ├── test-volume.tif     # 测试图像 (30, 512, 512)
    └── test-labels.tif     # 测试标签 (30, 512, 512)
```

---

## 🏗️ U-Net 架构详解

### 网络结构概览
U-Net 是一种对称的全卷积神经网络，形状类似字母 "U"：

**编码器路径（Encoder / Contracting Path）**：
- 通过连续的卷积和池化操作逐步降低分辨率
- 提取图像的高级语义特征
- 每个阶段使用 2 个 3x3 卷积 + ReLU 激活
- 使用 2x2 MaxPooling 进行下采样

**解码器路径（Decoder / Expanding Path）**：
- 通过转置卷积（ConvTranspose2d）逐步恢复分辨率
- 融合编码器的跳跃连接特征
- 每个阶段使用 2 个 3x3 卷积 + ReLU 激活

**跳跃连接（Skip Connections）**：
- 将编码器每层的特征图拷贝并裁剪后，与解码器对应层的特征图拼接
- 保留空间细节信息，有助于精确定位边界

### 模型参数
```python
输入: (1, 1, 572, 572)  # Batch, Channel, Height, Width
输出: (1, 2, 388, 388)  # 2 个类别（背景/前景）的概率图

总参数量：约 31M
```

### 特征图尺寸变化
```
输入: 572x572
  ↓ down1 (2×Conv3x3)     → 568x568 (64 channels)
  ↓ down2 (Pool+2×Conv3x3) → 280x280 (128 channels)
  ↓ down3 (Pool+2×Conv3x3) → 136x136 (256 channels)
  ↓ down4 (Pool+2×Conv3x3) → 64x64   (512 channels)
  ↓ down5 (Pool+2×Conv3x3) → 28x28   (1024 channels) ← Bottleneck
  ↑ up0 (ConvTranspose2x2) → 56x56   (512 channels)
  ↑ up1 (Concat+2×Conv3x3+ConvTranspose) → 104x104 (256 channels)
  ↑ up2 (Concat+2×Conv3x3+ConvTranspose) → 200x200 (128 channels)
  ↑ up3 (Concat+2×Conv3x3+ConvTranspose) → 392x392 (64 channels)
  ↑ final (Concat+2×Conv3x3+Conv1x1) → 388x388 (2 channels)
输出: 388x388
```

**注意**：由于使用的是 valid padding（不填充），每次卷积都会损失边缘像素，因此输出尺寸小于输入尺寸。

---

## 🧬 Dataset (ISBI 2012)

数据集来源：  
http://brainiac2.mit.edu/isbi_challenge/home

典型文件结构：

    train-volume.tif      (30, 512, 512)
    train-labels.tif      (30, 512, 512)
    test-volume.tif       (30, 512, 512)

由于体积限制，本仓库不包含.pth 模型权重文件，需要用户自行下载与训练。

---

## 🚀 训练流程详解

### 数据增强策略
为了应对训练数据量少（仅 30 张切片）的问题，项目使用了强数据增强：

```python
- ShiftScaleRotate: 旋转 ±180°、轻微平移
- HorizontalFlip: 水平翻转（50% 概率）
- VerticalFlip: 垂直翻转（50% 概率）
- ElasticTransform: 弹性形变，模拟组织的自然变形
- RandomBrightnessContrast: 亮度/对比度调整
```

### 损失函数
项目使用 **加权交叉熵损失（Weighted Cross-Entropy Loss）**，结合边界感知的权重图：

1. **类别权重（Class Weights）**：
   ```python
   wc = total_pixels / (num_classes × class_pixel_count)
   ```
   平衡类别不均衡问题（前景像素通常远少于背景）

2. **边界权重（Border Weights）**：
   ```python
   border_weight = w0 × exp(-(d1 + d2)² / (2σ²))
   ```
   - d1, d2: 距离最近的两个分割对象的距离
   - w0=10, σ=5 (超参数)
   - 目的：让模型更关注细胞间的分界线

3. **最终像素权重**：
   ```python
   w(x) = wc(x) + border_weight(x)
   ```

### 训练配置
```python
优化器: Adam (lr=1e-4, weight_decay=1e-5)
学习率调度: ReduceLROnPlateau (factor=0.5, patience=3)
Early Stopping: 8 轮验证集不提升则停止
Batch Size: 1 (由于输入图像较大)
Epochs: 最多 60 轮
```

### 两种训练模式

#### 1. 全数据训练（推荐用于最终模型）
```bash
python train.py
```
- 使用全部 30 张切片进行训练
- 生成模型权重: `unet_isbi2012_wc.pth`
- 适合最终提交或实际应用

#### 2. K-Fold 交叉验证训练
```python
# 在 train.py 中取消注释
train_val()  # 替代 train()
```
- 5 折交叉验证，每折生成一个模型
- 生成权重: `unet_isbi2012_best_0.pth` ~ `unet_isbi2012_best_4.pth`
- 适合评估模型泛化能力、调参

---

## 🔍 推理流程详解

运行推理脚本：

```bash
python infer.py
```

### 推理步骤
1. **加载模型**：读取训练好的权重文件 `unet_isbi2012_wc.pth`
2. **数据预处理**：归一化（与训练时一致）
3. **前向传播**：输入图像 → 输出 2 通道概率图
4. **后处理**：
   - `softmax → argmax`：获取每个像素的预测类别
   - `remove_small_objects`：移除小于阈值的连通域（去噪）
   - 映射为 0/255 的二值图像
5. **保存结果**：`ISBI-2012-challenge/test-volume-pred.tif`

### 输出格式
```
输入: test-volume.tif (30, 512, 512)
输出: test-volume-pred.tif (30, 388, 388)  # 注意尺寸变化
      - 像素值: 0=背景, 255=前景（细胞膜）
```

---

## 📈 评估指标详解

### 训练/验证阶段使用的指标

1. **Soft Dice Coefficient（Dice 系数）**
   ```python
   Dice = 2 × |预测前景 ∩ 真实前景| / (|预测前景| + |真实前景|)
   ```
   - 范围：0~1，越接近 1 越好
   - 衡量预测与真实标签的重叠程度
   - 对类别不均衡问题更鲁棒

2. **Accuracy（准确率）**
   ```python
   Accuracy = 正确预测像素数 / 总像素数
   ```

3. **Recall（召回率）**
   ```python
   Recall = TP / (TP + FN)
   ```
   - TP: 正确预测为前景的像素
   - FN: 实际是前景但预测为背景的像素
   - 衡量模型找到所有前景的能力

4. **Precision（精确率）**
   ```python
   Precision = TP / (TP + FP)
   ```
   - FP: 实际是背景但预测为前景的像素
   - 衡量预测为前景的像素中有多少是正确的

### ISBI 2012 官方评估指标（未实现）
- **Rand Error**：像素对分类错误率
- **Warping Error**：考虑拓扑形变的误差
- 需要提交预测结果到官方服务器进行评估

---

## 🔧 环境配置与依赖

### Python 环境
```bash
Python 3.9+
```

### 核心依赖
```bash
# 深度学习框架
torch >= 1.12.0

# 数据增强
albumentations >= 1.3.0

# 图像处理
scikit-image >= 0.19.0
tifffile >= 2022.0.0
opencv-python >= 4.6.0

# 数值计算
numpy >= 1.21.0
scipy >= 1.7.0

# 其他
scikit-learn  # 用于 K-Fold
```

### 安装方法

#### 方法 1：使用 pip 直接安装
```bash
pip install torch torchvision
pip install albumentations scikit-image tifffile opencv-python scipy scikit-learn
```

#### 方法 2：使用 requirements.txt（推荐）
创建 `requirements.txt` 文件后：
```bash
pip install -r requirements.txt
```

### GPU 支持（可选但推荐）
```bash
# 安装 CUDA 版本的 PyTorch（根据你的 CUDA 版本选择）
# 示例：CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 💡 使用技巧与注意事项

### 1. 数据集准备
- 确保数据集目录结构正确：`ISBI-2012-challenge/`
- 文件名必须与代码中一致：
  - `train-volume.tif`
  - `train-labels.tif`
  - `test-volume.tif`
  - `test-labels.tif`（可选，仅用于评估）

### 2. 训练建议
- **首次训练**：建议先运行 `test.py` 确保环境配置正确
- **GPU 内存不足**：减小 batch_size（当前已为 1）或降低输入分辨率
- **过拟合**：增强数据增强强度，添加更多 Dropout
- **欠拟合**：增加训练轮数，调高学习率

### 3. 调参建议
常见超参数调整：
```python
# 学习率
optimizer = torch.optim.Adam(lr=1e-4)  # 可尝试 1e-3 或 1e-5

# 边界权重参数（loss.py）
w0 = 10.0   # 增大会更关注边界
sigma = 5.0 # 控制权重衰减速度

# Early Stopping
patience = 8  # 增大会训练更久

# 数据增强强度（utils.py）
ElasticTransform(alpha=30, sigma=10, p=0.7)
```

### 4. 常见问题

**Q: 输出尺寸为什么比输入小？**  
A: 使用了 valid padding（无填充卷积），每次卷积会损失边缘像素。可改用 padded UNet 保持尺寸。

**Q: 训练很慢怎么办？**  
A: 使用 GPU 训练，确保安装了 CUDA 版本的 PyTorch。

**Q: 如何可视化结果？**  
A: 可使用 ImageJ/Fiji 或 Python 的 matplotlib 打开 .tif 文件查看。

**Q: 模型权重文件在哪里？**  
A: 训练后生成在项目根目录，文件名如 `unet_isbi2012_wc.pth`。注意不要上传到 Git（已在 .gitignore 中）。

---

## 🔬 代码示例

### 快速测试模型结构
```python
from model import u_net
import torch

# 创建模型
model = u_net()

# 创建随机输入
x = torch.randn(1, 1, 572, 572)

# 前向传播
output = model(x)
print(output.shape)  # torch.Size([1, 2, 388, 388])
```

### 自定义数据集路径
```python
from utils import TiffDateset

# 使用自定义路径
dataset = TiffDateset(
    train_volume_path="path/to/your/train-volume.tif",
    train_label_path="path/to/your/train-labels.tif",
    is_train=True
)
```

### 单张图像推理
```python
import torch
import tifffile
from model import u_net
from utils import get_aug

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型
model = u_net().to(device)
model.load_state_dict(torch.load("unet_isbi2012_wc.pth"))
model.eval()

# 读取图像
img = tifffile.imread("your_image.tif")
_, test_transform = get_aug()

# 预处理
transformed = test_transform(image=img[..., None])
x = transformed["image"].unsqueeze(0).to(device)

# 推理
with torch.no_grad():
    output = model(x)
    pred = torch.argmax(output, dim=1)
    
# 保存结果
pred_np = pred.squeeze().cpu().numpy()
tifffile.imwrite("prediction.tif", pred_np.astype("uint8") * 255)
```

---

## 🗒 TODO & 未来改进

- [ ] 添加 ISBI 官方评估指标（Warping / Rand）  
- [ ] 接入 TensorBoard 或其他可视化工具  
- [ ] 支持 patch-based 训练以提升分辨率与效果  
- [ ] 改为 padded UNet（避免大量 crop 操作）
- [ ] 添加模型集成（Ensemble）功能
- [ ] 支持 3D U-Net 版本
- [ ] 添加预训练权重下载链接
- [ ] 实现在线数据增强可视化工具

---

## 📚 参考资料

### 原始论文
- **U-Net: Convolutional Networks for Biomedical Image Segmentation**  
  Olaf Ronneberger, Philipp Fischer, Thomas Brox  
  MICCAI 2015  
  [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

### 数据集
- **ISBI 2012 EM Segmentation Challenge**  
  http://brainiac2.mit.edu/isbi_challenge/home

### 相关资源
- [PyTorch 官方文档](https://pytorch.org/docs/)
- [Albumentations 数据增强库](https://albumentations.ai/)
- [医学图像分割入门教程](https://github.com/topics/medical-image-segmentation)

---

## 👥 贡献指南

欢迎提交 Issue 和 Pull Request！

### 贡献方式
1. Fork 本仓库
2. 创建新分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范
- 遵循 PEP 8 Python 代码规范
- 添加必要的注释说明
- 更新相关文档

---

## 📞 联系方式

如有问题或建议，欢迎通过以下方式联系：

- 提交 GitHub Issue
- 发送邮件至项目维护者

---

## ⚖️ 开源协议

本项目采用 **MIT License** 开源协议。

详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- 感谢 Ronneberger 等人提出的 U-Net 架构
- 感谢 ISBI 2012 EM Segmentation Challenge 提供的数据集
- 感谢所有开源库的贡献者

---

## 📝 更新日志

### v1.0.0 (当前版本)
- ✅ 实现完整的 U-Net 模型
- ✅ 支持 ISBI 2012 数据集加载
- ✅ 实现边界加权损失函数
- ✅ 支持数据增强和 K-Fold 交叉验证
- ✅ 完善的训练和推理流程
- ✅ 添加详细的中文文档

---

**如果这个项目对你有帮助，请给个 ⭐️ Star 支持一下！**
