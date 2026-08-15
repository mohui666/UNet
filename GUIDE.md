# 🎓 仓库总览与学习指南

本文档是 UNet 项目的学习指南，帮助你快速了解整个仓库的结构和内容。

---

## 📚 文档导航

本仓库包含以下文档，建议按顺序阅读：

### 1. [README.md](README.md) - 项目主文档 ⭐
**推荐首先阅读**

包含内容：
- 项目概述和特点
- 详细的项目结构说明
- U-Net 架构概览
- 训练和推理指南
- 环境配置说明
- 使用技巧和常见问题
- 代码示例

**适合**：想要快速了解项目全貌的读者

---

### 2. [TUTORIAL.md](TUTORIAL.md) - 详细使用教程 📖
**推荐实际使用前阅读**

包含内容：
- 环境准备的详细步骤
- 数据集下载和准备
- 完整的训练流程
- 推理和可视化教程
- 进阶使用方法
- 常见问题解答
- 调参建议

**适合**：准备动手训练模型的用户

---

### 3. [ARCHITECTURE.md](ARCHITECTURE.md) - 架构深度解析 🏗️
**推荐深入学习时阅读**

包含内容：
- U-Net 设计哲学
- 编码器和解码器详解
- 跳跃连接原理
- 损失函数数学推导
- 数据流动详解
- 关键技术点分析
- 与原论文的对比

**适合**：想要深入理解 U-Net 原理的读者

---

## 🗂️ 代码文件说明

### 核心文件

#### model.py - U-Net 模型实现
```python
# 主要内容
- class u_net: U-Net 网络结构定义
  - __init__(): 定义编码器、解码器、跳跃连接
  - forward(): 前向传播逻辑
```

**关键点**：
- 对称的 U 型结构
- 跳跃连接实现
- Dropout 防止过拟合

---

#### loss.py - 损失函数和评估指标
```python
# 主要内容
- get_wc(): 计算类别权重
- per_pixel_weight(): 生成像素级权重图
- make_weight_map_per_sample(): 边界权重生成
- dice_score(): 计算 Dice 系数
```

**关键点**：
- 边界感知的权重图
- 类别不均衡处理
- Dice 系数评估

---

#### train.py - 训练脚本
```python
# 主要内容
- train_val(): K-Fold 交叉验证训练
- train(): 全数据训练
- evaluate(): 验证函数
```

**关键点**：
- Early Stopping
- 学习率调度
- 加权损失计算

---

#### utils.py - 工具函数
```python
# 主要内容
- class ISBI2012: 数据集加载类
- get_aug(): 数据增强配置
- crop(): 中心裁剪
- kfold(): K 折划分
- init_weights(): 权重初始化
- accuracy_recall_precision(): 指标计算
```

**关键点**：
- 强数据增强
- Kaiming 初始化
- 多指标评估

---

#### infer.py - 推理脚本
```python
# 主要内容
- infer(): 推理主函数
- remove_small_objects(): 连通域去噪
```

**关键点**：
- 批量推理
- 后处理去噪
- 结果保存

---

#### test.py - 模型测试
```python
# 快速测试脚本
# 用于验证模型能否正常运行
```

---

## 🎯 学习路径建议

### 初学者路径（第一次接触医学图像分割）

1. **第一步**：阅读 [README.md](README.md) 的前半部分
   - 了解项目是做什么的
   - 了解 U-Net 的基本概念

2. **第二步**：阅读 [TUTORIAL.md](TUTORIAL.md) 的"快速开始"部分
   - 配置环境
   - 运行 `test.py` 验证环境

3. **第三步**：阅读 [TUTORIAL.md](TUTORIAL.md) 的"训练模型"部分
   - 准备数据集
   - 运行第一次训练

4. **第四步**：阅读代码注释
   - 从 `model.py` 开始，理解网络结构
   - 然后看 `utils.py`，理解数据处理
   - 最后看 `train.py` 和 `infer.py`

5. **第五步**：阅读 [ARCHITECTURE.md](ARCHITECTURE.md)
   - 深入理解 U-Net 原理
   - 了解设计细节

---

### 进阶用户路径（有深度学习基础）

1. **第一步**：快速浏览 [README.md](README.md)
   - 了解项目特点和结构

2. **第二步**：直接阅读代码
   - `model.py` → `loss.py` → `train.py` → `infer.py`
   - 关注代码注释中的技术细节

3. **第三步**：阅读 [ARCHITECTURE.md](ARCHITECTURE.md)
   - 理解损失函数设计
   - 对比原论文的实现差异

4. **第四步**：根据需求修改代码
   - 参考 [TUTORIAL.md](TUTORIAL.md) 的"进阶使用"部分
   - 尝试不同的超参数配置

---

### 研究者路径（需要改进或扩展）

1. **第一步**：阅读 [ARCHITECTURE.md](ARCHITECTURE.md)
   - 深入理解每个设计选择的原因

2. **第二步**：研究代码实现
   - 关注损失函数、数据增强、网络结构
   - 对比原论文，理解改进点

3. **第三步**：实验和改进
   - 参考 [TUTORIAL.md](TUTORIAL.md) 的调参建议
   - 尝试新的网络结构或损失函数

4. **第四步**：记录实验
   - 使用 TUTORIAL.md 中的实验记录模板
   - 比较不同方法的效果

---

## 💡 关键概念速查

### U-Net 是什么？
一种专为医学图像分割设计的全卷积神经网络，特点是：
- 对称的 U 型结构
- 跳跃连接保留细节
- 适合小样本学习

### 为什么输出比输入小？
使用 valid padding（无填充卷积），每次卷积都会损失边缘像素：
- 输入：572×572
- 输出：388×388

### 什么是边界加权损失？
在细胞间的边界处给予更高的权重，让模型更关注难以分割的区域。

### 什么是数据增强？
通过旋转、翻转、形变等方式人工扩充数据，应对小样本问题。

### 什么是跳跃连接？
将编码器的特征直接传递给解码器，保留空间细节信息。

---

## 🔗 快速链接

### 常用命令

```bash
# 测试模型结构
python test.py

# 训练模型（全数据）
python train.py

# 推理
python infer.py

# 查看帮助
python train.py --help  # 如果实现了命令行参数
```

### 重要文件路径

```
模型权重：unet_isbi2012_wc.pth
数据集目录：ISBI-2012-challenge/
预测结果：ISBI-2012-challenge/test-volume-pred.tif
```

---

## 📊 性能指标说明

### Dice 系数（最重要）
- 范围：0~1
- 越接近 1 越好
- 衡量预测与真实标签的重叠程度

### 准确率（Accuracy）
- 正确预测的像素比例
- 可能被背景主导（背景像素多）

### 召回率（Recall）
- 找到了多少真实的前景像素
- 高召回率：不会漏掉前景

### 精确率（Precision）
- 预测为前景的像素中有多少是正确的
- 高精确率：不会误判背景为前景

---

## 🐛 遇到问题？

1. **代码问题**：查看代码文件中的详细注释
2. **使用问题**：查看 [TUTORIAL.md](TUTORIAL.md) 的"常见问题"部分
3. **原理问题**：查看 [ARCHITECTURE.md](ARCHITECTURE.md)
4. **其他问题**：提交 GitHub Issue

---

## 🌟 推荐学习资源

### 论文
- U-Net 原论文：[arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

### 在线课程
- Deep Learning Specialization (Andrew Ng)
- Fast.ai Practical Deep Learning

### 相关项目
- GitHub 搜索：`medical image segmentation`
- Papers with Code：Medical Image Segmentation

---

## ✅ 检查清单

使用本项目前，确保：

- [ ] 已阅读 README.md
- [ ] 已安装所有依赖（`pip install -r requirements.txt`）
- [ ] 已下载并准备好数据集
- [ ] 已运行 `test.py` 验证环境
- [ ] 理解了基本的 U-Net 概念
- [ ] 知道如何查看文档获取帮助

---

**祝你学习愉快！如果这个项目对你有帮助，请给个 ⭐️ Star！**

---

*最后更新：2024年*
*作者：mohui666*
*协议：MIT License*
