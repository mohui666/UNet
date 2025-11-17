# 📘 UNet for EM Membrane Segmentation (ISBI 2012)

本项目实现了基于 U-Net 的电子显微镜 (EM) 膜结构分割模型，面向 ISBI 2012 EM Segmentation Challenge 数据集。  
包含完整的数据加载、增强、模型、损失、训练与推理流程。

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

## 📂 Project Structure

    UNet/
    │
    ├── model.py                # U-Net 模型
    ├── train.py                # 训练脚本（含调度器与早停）
    ├── infer.py                # 推理脚本
    ├── utils.py                # 数据集、增强、裁剪、KFold、init_weights
    ├── loss.py                 # 权重图、Dice 计算等
    ├── .gitignore              # 忽略大文件、数据集、权重等
    └── ISBI-2012-challenge/    # 本地数据集目录（不在仓库中）

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

## 🚀 Training

在项目根目录下运行：

    python train.py

训练结束后会在当前目录生成类似：

    unet_isbi2012_wc_best.pth

该权重文件未被加入版本控制，请自行备份或另行托管。

---

## 🔍 Inference

在已训练好权重的前提下，运行：

    python infer.py

推理完成后，将在数据集目录下生成：

    ISBI-2012-challenge/test-volume-pred.tif

每张切片会经过：
- softmax → argmax 得到前景标签
- 移除小连通域噪声
- 映射为 0/255 的二值 mask

---

## 📈 Evaluation

当前实现的度量包括：

- Soft Dice（验证阶段评估）
- Accuracy / Recall / Precision（在有标签时使用）
- Weighted CrossEntropyLoss（训练主损失）
- 边界加权图（参考 U-Net 原论文的边界项设计）

---

## 🔧 Requirements

环境示例：

    Python 3.9+
    PyTorch >= 1.12
    Albumentations
    scikit-image
    tifffile
    opencv-python
    numpy

安装依赖（可根据实际情况编写 requirements.txt）：

    pip install -r requirements.txt

或手动安装上述包。

---

## 📜 License

MIT License


