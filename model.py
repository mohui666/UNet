import torch
from torch import nn

from utils import crop


class u_net(nn.Module):
    """
    U-Net 模型实现（Ronneberger et al., 2015）
    
    这是一个用于图像分割的全卷积神经网络，具有对称的编码器-解码器结构。
    
    架构特点：
    1. 编码器路径（Contracting Path）：通过卷积和池化逐步提取高级特征
    2. 解码器路径（Expanding Path）：通过上采样和卷积恢复空间分辨率
    3. 跳跃连接（Skip Connections）：将编码器的特征图与解码器融合，保留细节信息
    
    参数:
        in_size (int): 输入图像尺寸，默认 572x572
        in_channel (int): 输入通道数，默认为 1（灰度图）
    
    输入输出：
        输入: (batch, 1, 572, 572)
        输出: (batch, 2, 388, 388)  # 2个类别的概率图（背景/前景）
    """
    def __init__(self, in_size=572, in_channel=1):
        super().__init__()
        self.in_size = in_size
        self.in_channel = in_channel
        
        # ==================== 编码器路径（下采样）====================
        # down1: 572x572x1 -> 568x568x64
        # 第一层编码器，提取低级特征（边缘、纹理等）
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3),      # 3x3 卷积，valid padding
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU()
        )
        # down2: 568x568x64 -> (MaxPool) -> 284x284x64 -> 280x280x128
        # 第二层编码器，提取中级特征
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),          # 2x2 最大池化，降低分辨率
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU()
        )
        # down3: 280x280x128 -> (MaxPool) -> 140x140x128 -> 136x136x256
        # 第三层编码器
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),
            nn.ReLU()
        )
        # down4: 136x136x256 -> (MaxPool) -> 68x68x256 -> 64x64x512
        # 第四层编码器
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3),
            nn.ReLU()
        )
        # down5up0: 64x64x512 -> (MaxPool) -> 32x32x512 -> 28x28x1024 -> (Up) -> 56x56x512
        # 瓶颈层（Bottleneck）+ 第一层解码器
        # 这是网络最深处，特征图最小但通道数最多，提取最抽象的高级语义特征
        self.down5up0 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, 3),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),      # Dropout 防止过拟合
            nn.Conv2d(1024, 1024, 3),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.ConvTranspose2d(1024, 512, 2, 2)  # 转置卷积上采样 2 倍
        )
        
        # ==================== 解码器路径（上采样）====================
        # up1: (56x56x512 + 64x64x512裁剪后拼接) -> 52x52x512 -> (Up) -> 104x104x256
        # 第二层解码器，融合编码器 down4 的跳跃连接
        self.up1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3),  # 输入通道是 1024，因为拼接了两个 512 通道的特征图
            nn.ReLU(),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 2, 2)
        )
        # up2: (104x104x256 + 136x136x256裁剪后拼接) -> 100x100x256 -> (Up) -> 200x200x128
        # 第三层解码器，融合编码器 down3 的跳跃连接
        self.up2 = nn.Sequential(
            nn.Conv2d(512, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, 2)
        )
        # up3: (200x200x128 + 280x280x128裁剪后拼接) -> 196x196x128 -> (Up) -> 392x392x64
        # 第四层解码器，融合编码器 down2 的跳跃连接
        self.up3 = nn.Sequential(
            nn.Conv2d(256, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, 2)
        )
        
        # ==================== 最终输出层 ====================
        # final: (392x392x64 + 568x568x64裁剪后拼接) -> 388x388x64 -> 388x388x2
        # 融合编码器 down1 的跳跃连接，输出每个像素属于 2 个类别的概率
        self.final = nn.Sequential(
            nn.Conv2d(128, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1)       # 1x1 卷积，将 64 通道映射到 2 个类别
        )

    def forward(self, X):
        """
        前向传播函数
        
        执行流程：
        1. 编码器路径：逐层提取特征，保存每层输出用于跳跃连接
        2. 瓶颈层：最深层的特征提取
        3. 解码器路径：逐层上采样并融合编码器特征
        
        Args:
            X (torch.Tensor): 输入图像，shape: (batch, 1, 572, 572)
        
        Returns:
            torch.Tensor: 输出概率图，shape: (batch, 2, 388, 388)
        """
        # ==================== 编码器前向传播 ====================
        copy1 = self.down1(X)      # (B, 64, 568, 568) - 保存用于跳跃连接
        copy2 = self.down2(copy1)  # (B, 128, 280, 280)
        copy3 = self.down3(copy2)  # (B, 256, 136, 136)
        copy4 = self.down4(copy3)  # (B, 512, 64, 64)

        # ==================== 瓶颈层 + 解码器前向传播 ====================
        # 第一层解码：瓶颈层输出并开始上采样
        low = self.down5up0(copy4)  # (B, 512, 56, 56)
        
        # 裁剪 copy4 使其与 low 尺寸匹配，然后在通道维度拼接
        # 跳跃连接 1: down4 -> up1
        crop4 = crop(copy4, low.shape)         # (B, 512, 56, 56)
        cat1 = torch.cat((crop4, low), dim=1)  # (B, 1024, 56, 56)
        up1 = self.up1(cat1)                   # (B, 256, 104, 104)

        # 跳跃连接 2: down3 -> up2
        crop3 = crop(copy3, up1.shape)         # (B, 256, 104, 104)
        cat2 = torch.cat((crop3, up1), dim=1)  # (B, 512, 104, 104)
        up2 = self.up2(cat2)                   # (B, 128, 200, 200)

        # 跳跃连接 3: down2 -> up3
        crop2 = crop(copy2, up2.shape)         # (B, 128, 200, 200)
        cat3 = torch.cat((crop2, up2), dim=1)  # (B, 256, 200, 200)
        up3 = self.up3(cat3)                   # (B, 64, 392, 392)

        # 跳跃连接 4: down1 -> final
        crop1 = crop(copy1, up3.shape)         # (B, 64, 392, 392)
        cat4 = torch.cat((crop1, up3), dim=1)  # (B, 128, 392, 392)

        # 最终输出层
        output = self.final(cat4)              # (B, 2, 388, 388)
        return output


if __name__ == "__main__":
    # 测试模型结构
    net = u_net()
    X = torch.empty(size=(1, 1, 572, 572))
    output = net(X)
    print(f"输入形状: {X.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数总量: {sum(p.numel() for p in net.parameters()):,}")
