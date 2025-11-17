# 📚 U-Net 使用教程

本教程将手把手教你如何使用这个 U-Net 项目进行医学图像分割。

---

## 📋 目录

1. [环境准备](#环境准备)
2. [数据集准备](#数据集准备)
3. [快速开始](#快速开始)
4. [训练模型](#训练模型)
5. [模型推理](#模型推理)
6. [结果可视化](#结果可视化)
7. [进阶使用](#进阶使用)
8. [常见问题](#常见问题)
9. [调参建议](#调参建议)

---

## 环境准备

### Step 1: 安装 Python

确保你的系统已安装 Python 3.9 或更高版本：

```bash
python --version
# 应该显示：Python 3.9.x 或更高
```

### Step 2: 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv unet_env

# 激活虚拟环境
# Windows:
unet_env\Scripts\activate

# Linux/Mac:
source unet_env/bin/activate
```

### Step 3: 安装依赖

创建 `requirements.txt` 文件：

```text
torch>=1.12.0
torchvision>=0.13.0
albumentations>=1.3.0
scikit-image>=0.19.0
tifffile>=2022.0.0
opencv-python>=4.6.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
```

安装所有依赖：

```bash
pip install -r requirements.txt
```

### Step 4: 验证 GPU 支持（可选）

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

如果显示 `CUDA available: True`，说明可以使用 GPU 加速训练。

---

## 数据集准备

### Step 1: 下载数据集

访问 [ISBI 2012 Challenge 官网](http://brainiac2.mit.edu/isbi_challenge/home) 下载数据集。

数据集包含：
- `train-volume.tif` - 训练图像（30 张切片）
- `train-labels.tif` - 训练标签（30 张切片）
- `test-volume.tif` - 测试图像（30 张切片）
- `test-labels.tif` - 测试标签（30 张切片，用于本地评估）

### Step 2: 组织目录结构

将下载的数据放入项目目录：

```
UNet/
├── model.py
├── train.py
├── infer.py
├── utils.py
├── loss.py
├── test.py
└── ISBI-2012-challenge/          ← 创建此目录
    ├── train-volume.tif
    ├── train-labels.tif
    ├── test-volume.tif
    └── test-labels.tif
```

### Step 3: 验证数据集

运行以下 Python 代码验证数据集是否正确加载：

```python
import tifffile
import numpy as np

# 加载数据
train_images = tifffile.imread("ISBI-2012-challenge/train-volume.tif")
train_labels = tifffile.imread("ISBI-2012-challenge/train-labels.tif")

print(f"训练图像形状: {train_images.shape}")  # 应该是 (30, 512, 512)
print(f"训练标签形状: {train_labels.shape}")  # 应该是 (30, 512, 512)
print(f"图像值范围: [{train_images.min()}, {train_images.max()}]")
print(f"标签值范围: [{train_labels.min()}, {train_labels.max()}]")
```

预期输出：
```
训练图像形状: (30, 512, 512)
训练标签形状: (30, 512, 512)
图像值范围: [0, 255]
标签值范围: [0, 255]
```

---

## 快速开始

### 测试模型结构

首先确保模型能正常运行：

```bash
python test.py
```

预期输出：
```
torch.Size([1, 2, 388, 388])
```

这说明模型结构正确，可以接受 572×572 的输入并输出 388×388 的分割图。

### 测试数据加载

```python
from utils import ISBI2012, get_aug

# 创建数据集
train_transform, _ = get_aug()
dataset = ISBI2012(is_train=True, transform=train_transform)

# 加载一个样本
image, label = dataset[0]
print(f"图像形状: {image.shape}")   # (1, 572, 572) 或类似
print(f"标签形状: {label.shape}")   # (572, 572) 或类似
```

---

## 训练模型

### 方法 1: 全数据训练（推荐用于最终模型）

这是最简单的训练方式，使用全部 30 张切片进行训练：

```bash
python train.py
```

训练过程中会显示：

```
==> Epoch 1 
avg_loss=0.4523

==> Epoch 2 
avg_loss=0.3892

...

模型已保存
```

训练完成后，模型权重保存为 `unet_isbi2012_wc.pth`。

### 方法 2: K-Fold 交叉验证训练

如果你想评估模型的泛化能力或进行调参，可以使用交叉验证：

**Step 1**: 修改 `train.py` 文件

```python
if __name__ == "__main__":
    train_val()  # 改为调用 train_val()
    # train()    # 注释掉 train()
```

**Step 2**: 运行训练

```bash
python train.py
```

这会进行 5 折交叉验证，每折训练一个模型：
- `unet_isbi2012_best_0.pth`
- `unet_isbi2012_best_1.pth`
- ...
- `unet_isbi2012_best_4.pth`

### 训练参数说明

在 `train.py` 中可以调整以下参数：

```python
# 优化器
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=1e-4,           # 学习率
    weight_decay=1e-5  # L2 正则化
)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',      # 监控最小化指标（loss）
    factor=0.5,      # 学习率衰减因子
    patience=3       # 容忍轮数
)

# 早停
patience = 8         # 连续 8 轮验证集不提升则停止
epochs = 60          # 最大训练轮数
```

### 监控训练过程

训练时会打印以下信息：

```python
Epoch 10 
val_loss=0.3245      # 验证集损失
val_dice=0.8712      # Dice 系数（越高越好，范围 0-1）
acc=0.9432           # 准确率
recall=0.8654        # 召回率
precision=0.8891     # 精确率
```

**关键指标**：
- **val_dice**: 最重要，表示分割质量
- **val_loss**: 用于学习率调度和早停
- **recall**: 找到了多少真实的前景像素
- **precision**: 预测为前景的像素中有多少是正确的

---

## 模型推理

训练完成后，使用模型对测试集进行推理：

### 基本推理

```bash
python infer.py
```

推理完成后会显示：

```
acc=0.9456 recall=0.8723 precision=0.8942
saved in: ISBI-2012-challenge/test-volume-pred.tif
```

输出文件 `test-volume-pred.tif` 包含 30 张预测的分割图。

### 推理流程说明

推理过程包括以下步骤：

1. **加载模型权重**
   ```python
   model = u_net().to(device)
   model.load_state_dict(torch.load("unet_isbi2012_wc.pth"))
   model.eval()
   ```

2. **数据预处理**
   - 归一化（均值 0，标准差 1）
   - 转换为 PyTorch Tensor

3. **前向传播**
   ```python
   y_hat = model(x)  # 输出 (1, 2, 388, 388)
   ```

4. **后处理**
   - `argmax`: 选择概率最高的类别
   - `remove_small_objects`: 移除小于阈值的连通域（去噪）
   - 映射为 0/255 的二值图像

### 单张图像推理

如果你想对单张图像进行推理：

```python
import torch
import tifffile
from model import u_net
from utils import get_aug
from infer import remove_small_objects
import numpy as np

# 配置
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型
model = u_net().to(device)
model.load_state_dict(torch.load("unet_isbi2012_wc.pth", map_location=device))
model.eval()

# 读取图像
img = tifffile.imread("your_image.tif")  # 假设是 512×512 灰度图

# 预处理
_, test_transform = get_aug()
transformed = test_transform(image=img[..., None])
x = transformed["image"].unsqueeze(0).to(device)

# 推理
with torch.no_grad():
    output = model(x)
    pred = torch.argmax(output, dim=1)
    pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)

# 后处理
pred_clean = remove_small_objects(pred_np, min_size=50)
pred_clean = pred_clean * 255

# 保存结果
tifffile.imwrite("prediction.tif", pred_clean)
print("推理完成！结果保存为 prediction.tif")
```

---

## 结果可视化

### 使用 ImageJ/Fiji

[ImageJ](https://imagej.net/software/fiji/) 是专业的医学图像查看工具：

1. 下载并安装 Fiji
2. 打开 Fiji，选择 `File` → `Open`
3. 选择 `test-volume-pred.tif`
4. 使用滑块查看不同切片

### 使用 Python + Matplotlib

```python
import matplotlib.pyplot as plt
import tifffile

# 加载结果
predictions = tifffile.imread("ISBI-2012-challenge/test-volume-pred.tif")
test_images = tifffile.imread("ISBI-2012-challenge/test-volume.tif")
test_labels = tifffile.imread("ISBI-2012-challenge/test-labels.tif")

# 可视化某一张切片（例如第 15 张）
slice_idx = 15

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(test_images[slice_idx], cmap='gray')
axes[0].set_title('原始图像')
axes[0].axis('off')

axes[1].imshow(test_labels[slice_idx], cmap='gray')
axes[1].set_title('真实标签')
axes[1].axis('off')

axes[2].imshow(predictions[slice_idx], cmap='gray')
axes[2].set_title('预测结果')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('visualization.png', dpi=150)
plt.show()

print(f"可视化已保存为 visualization.png")
```

### 叠加显示

将预测结果叠加到原图上：

```python
import matplotlib.pyplot as plt
import tifffile
import numpy as np

slice_idx = 15

# 加载数据
img = tifffile.imread("ISBI-2012-challenge/test-volume.tif")[slice_idx]
pred = tifffile.imread("ISBI-2012-challenge/test-volume-pred.tif")[slice_idx]

# 创建叠加图像
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img, cmap='gray')
ax.imshow(pred, cmap='Reds', alpha=0.5)  # 红色半透明覆盖
ax.set_title('预测结果叠加到原图')
ax.axis('off')

plt.tight_layout()
plt.savefig('overlay.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 进阶使用

### 使用自定义数据集

如果你有自己的医学图像数据：

**Step 1**: 准备数据格式

确保你的数据是 TIFF 格式，形状为 `(D, H, W)`：
- D: 切片数量
- H, W: 图像高度和宽度

**Step 2**: 使用 `TiffDateset` 类

```python
from utils import TiffDateset, get_aug

train_transform, _ = get_aug()

# 自定义路径
dataset = TiffDateset(
    train_volume_path="path/to/your/train-volume.tif",
    train_label_path="path/to/your/train-labels.tif",
    is_train=True,
    transform=train_transform
)

# 正常使用
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
```

### 修改数据增强

编辑 `utils.py` 中的 `get_aug()` 函数：

```python
def get_aug():
    train_tf = A.Compose([
        # 旋转
        A.ShiftScaleRotate(
            shift_limit=0.1,      # 增加平移范围
            scale_limit=0.2,      # 添加缩放
            rotate_limit=180,
            p=1.0
        ),
        # 翻转
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # 弹性形变
        A.ElasticTransform(
            alpha=50,             # 增加形变强度
            sigma=10,
            p=0.8                 # 提高应用概率
        ),
        # 亮度对比度
        A.RandomBrightnessContrast(
            brightness_limit=0.2, # 增加亮度调整范围
            contrast_limit=0.2,
            p=1.0
        ),
        # 添加噪声（可选）
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        # 标准化
        A.Normalize(mean=(0.0,), std=(1.0,)),
        ToTensorV2()
    ])
    # ... test_tf 保持不变
```

### 调整网络结构

如果你的图像尺寸不是 512×512，可以修改模型：

**方法 1**: 调整输入尺寸（简单）

```python
# 假设你的图像是 256×256
model = u_net(in_size=256, in_channel=1)
```

**方法 2**: 修改网络层数（高级）

在 `model.py` 中增加或减少编码器/解码器层。

---

## 常见问题

### Q1: 训练时显示 CUDA out of memory

**原因**：GPU 显存不足

**解决方案**：
1. 减小 batch size（当前已经是 1）
2. 降低输入分辨率
3. 使用 CPU 训练（速度较慢）

```python
# 在 train.py 中强制使用 CPU
device = "cpu"  # 而不是自动选择
```

### Q2: 训练很慢

**可能原因**：
1. 没有使用 GPU
2. 数据增强太复杂

**解决方案**：
1. 确保安装了 CUDA 版本的 PyTorch
2. 减少数据增强的复杂度
3. 使用多进程数据加载

```python
# 在 DataLoader 中添加
dataloader = DataLoader(
    dataset, 
    batch_size=1, 
    shuffle=True,
    num_workers=4  # 使用 4 个进程加载数据
)
```

### Q3: 模型不收敛 / Loss 不下降

**可能原因**：
1. 学习率设置不当
2. 数据预处理有问题
3. 标签格式不正确

**解决方案**：
1. 降低学习率：`lr=1e-5`
2. 检查数据和标签是否正确加载
3. 确保标签是 0/1 格式（不是 0/255）

### Q4: 预测结果全是黑色或全是白色

**原因**：模型过于偏向某一类别

**解决方案**：
1. 检查类别权重是否正确计算
2. 增加边界权重参数 `w0`
3. 调整 Dice 阈值（在 `loss.py` 中）

```python
# 在 dice_score() 函数中
pred_fg = (probs[:, 1] > 0.5).float()  # 降低阈值从 0.9 到 0.5
```

### Q5: 如何继续训练一个已有的模型？

```python
# 在 train.py 的训练函数开始处添加
model = u_net().to(device)

# 加载已有权重
try:
    model.load_state_dict(torch.load("unet_isbi2012_wc.pth"))
    print("从已有模型继续训练")
except FileNotFoundError:
    print("从头开始训练")
    model.apply(init_weights)

# 继续正常训练流程
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # 建议使用更小的学习率
```

---

## 调参建议

### 学习率

| 学习率 | 适用场景 | 效果 |
|-------|---------|------|
| 1e-3 | 快速原型验证 | 收敛快但可能不稳定 |
| 1e-4 | 默认选择（推荐） | 平衡速度和稳定性 |
| 1e-5 | 精细调优 | 收敛慢但更稳定 |

### 边界权重参数

在 `loss.py` 的 `per_pixel_weight()` 函数中：

```python
w0 = 10.0   # 边界权重因子
sigma = 5.0 # 高斯核标准差
```

| w0 值 | 效果 |
|------|------|
| 5.0 | 轻度强调边界 |
| 10.0 | 中度强调边界（默认） |
| 20.0 | 重度强调边界 |

| sigma 值 | 效果 |
|---------|------|
| 3.0 | 权重衰减快，只影响紧邻边界的区域 |
| 5.0 | 适中（默认） |
| 10.0 | 权重衰减慢，影响更大范围 |

### 数据增强强度

| 参数 | 弱增强 | 中等增强 | 强增强 |
|-----|-------|---------|--------|
| rotate_limit | ±45° | ±90° | ±180° |
| ElasticTransform alpha | 10 | 30 | 50 |
| brightness_limit | 0.05 | 0.1 | 0.2 |

**建议**：
- 数据量少（<50 张）：使用强增强
- 数据量中等（50-200 张）：使用中等增强
- 数据量多（>200 张）：使用弱增强

### Early Stopping

```python
patience = 8  # 容忍轮数
```

| Patience | 适用场景 |
|----------|---------|
| 5 | 快速实验，容易欠拟合 |
| 8 | 默认选择（推荐） |
| 15 | 充分训练，但训练时间长 |

---

## 实验记录模板

建议在训练时记录实验参数和结果：

```
实验编号：exp001
日期：2024-01-15
目标：基线模型

超参数：
- 学习率：1e-4
- Batch Size：1
- Epochs：60
- Early Stopping Patience：8
- w0：10.0
- sigma：5.0

结果：
- 最佳 Dice：0.8712
- 最佳 Epoch：45
- 训练时间：2 小时 15 分钟
- 测试集 Accuracy：0.9456

备注：基线表现良好，可以尝试增加 w0 提高边界分割质量
```

---

## 总结

通过本教程，你应该能够：

✅ 正确配置环境和数据集  
✅ 运行训练和推理流程  
✅ 可视化和评估结果  
✅ 调整超参数优化性能  
✅ 应用到自己的数据集  

如有问题，欢迎提交 GitHub Issue！

**祝你训练顺利！🚀**
