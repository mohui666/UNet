# UNet for EM Membrane Segmentation (ISBI 2012)

本项目实现了基于 **U-Net** 的电子显微镜(EM)膜结构分割模型，面向 **ISBI 2012 EM Segmentation Challenge** 数据集。  
包含：模型结构、数据增强、加权损失 (boundary-aware weight map)、Dice 评估、训练脚本、推理脚本等完整流程。

---

## 📌 Features

- **原版 U-Net (Ronneberger et al., 2015)**  
  无 padding 卷积 + skip connection + 多次裁剪对齐。

- **ISBI 2012 数据集完整支持**  
  - train-volume.tif (30 × 512 × 512)  
  - train-labels.tif  
  - test-volume.tif  

- **强数据增强**  
  使用 Albumentations：旋转、弹性形变、镜像、亮度对比度等。

- **边界权重损失（UNet 论文同款）**  
  基于 foreground/background 距离的 exponential border weight。

- **训练流程完整实现**
  - CrossEntropyLoss（像素加权）
  - soft Dice 评估
  - Early Stopping
  - 学习率调度器 StepLR / ReduceLROnPlateau（可选）

- **推理脚本**
  - argmax segmentation
  - 移除小连通域
  - 输出 3D prediction tif 文件

---

## 📂 Project Structure

