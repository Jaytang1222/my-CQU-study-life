"""
UNet 模型实现（适用于蒙特卡洛渲染去噪）
"""
import jittor as jt
from jittor import nn


class DoubleConv(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN -> ReLU，带残差连接"""
    def __init__(self, in_ch, out_ch, mid_ch=None, use_residual=True):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.use_residual = use_residual and (in_ch == out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def execute(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x  # 残差连接
        return out


class ChannelAttention(nn.Module):
    """通道注意力机制：增强重要特征通道"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 使用全局平均池化和最大池化（简化版）
        self.reduction = reduction
        # 使用1x1卷积代替Linear（因为Jittor可能没有Linear或AdaptivePool）
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def execute(self, x):
        B, C, H, W = x.shape
        # 全局平均池化：逐维求平均（使用位置参数，避免不支持的dim关键字）
        avg_out = jt.mean(x, 2)           # [B, C, W]  （对 H 维求平均）
        avg_out = jt.mean(avg_out, 2)     # [B, C]     （再对 W 维求平均）
        avg_out = avg_out.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]

        # 全局最大池化：逐维取最大（使用位置参数）
        max_out = jt.max(x, 2)            # [B, C, W]  （对 H 维取最大）
        max_out = jt.max(max_out, 2)      # [B, C]     （再对 W 维取最大）
        max_out = max_out.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        
        # 通过FC层（1x1卷积）
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        # 合并并应用
        out = avg_out + max_out
        return x * out


class SpatialAttention(nn.Module):
    """空间注意力机制：增强重要空间位置"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def execute(self, x):
        # 在通道维度上计算平均值和最大值
        avg_out = jt.mean(x, 1)            # [B, H, W]
        avg_out = avg_out.unsqueeze(1)     # [B, 1, H, W]
        max_out = jt.max(x, 1)             # [B, H, W]
        max_out = max_out.unsqueeze(1)     # [B, 1, H, W]
        # 拼接并通过卷积
        out = jt.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return x * out


class AttentionDoubleConv(nn.Module):
    """带注意力机制的双卷积块"""
    def __init__(self, in_ch, out_ch, mid_ch=None, use_attention=True):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.conv = DoubleConv(in_ch, out_ch, mid_ch, use_residual=False)
        self.use_attention = use_attention
        if use_attention and out_ch >= 16:
            self.channel_att = ChannelAttention(out_ch)
            self.spatial_att = SpatialAttention()
    
    def execute(self, x):
        x = self.conv(x)
        if self.use_attention and hasattr(self, 'channel_att'):
            x = self.channel_att(x)
            x = self.spatial_att(x)
        return x


class RefineBlock(nn.Module):
    """输出后的小型细化模块：轻量滤波+残差锐化，稳定暗部和高噪细节"""
    def __init__(self, channels=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.res_scale = 0.5

    def execute(self, x):
        refine = self.block(x)
        return x + self.res_scale * refine


class Down(nn.Module):
    """下采样：MaxPool + DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Pool(2, op="maximum"),
            DoubleConv(in_ch, out_ch),
        )

    def execute(self, x):
        return self.net(x)


class Up(nn.Module):
    """上采样：Upsample/Deconv + 拼接 + DoubleConv"""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def execute(self, x1, x2):
        x1 = self.up(x1)
        # padding 对齐
        diff_y = x2.shape[2] - x1.shape[2]
        diff_x = x2.shape[3] - x1.shape[3]
        x1 = nn.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                         diff_y // 2, diff_y - diff_y // 2])
        x = jt.concat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def execute(self, x):
        return self.conv(x)




class UNet(nn.Module):
    """
    改进的UNet模型用于蒙特卡洛渲染去噪
    添加了残差连接、注意力机制和残差学习
    
    Args:
        n_channels: 输入通道数（通常为3，RGB图像）
        n_classes: 输出通道数（通常为3，RGB图像）
        bilinear: 是否使用双线性插值进行上采样
        use_attention: 是否使用注意力机制
        use_residual_learning: 是否使用残差学习（学习噪声而不是直接学习干净图像）
    """
    
    def __init__(self, n_channels=3, n_classes=3, bilinear=True, use_attention=True, use_residual_learning=True, use_refine_head=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_residual_learning = use_residual_learning
        self.use_refine_head = use_refine_head
        
        # 编码器（下采样路径）- 使用注意力机制
        self.inc = AttentionDoubleConv(n_channels, 64, use_attention=use_attention)
        self.down1 = Down(64, 128)
        self.down1_att = AttentionDoubleConv(128, 128, use_attention=use_attention) if use_attention else None
        self.down2 = Down(128, 256)
        self.down2_att = AttentionDoubleConv(256, 256, use_attention=use_attention) if use_attention else None
        self.down3 = Down(256, 512)
        self.down3_att = AttentionDoubleConv(512, 512, use_attention=use_attention) if use_attention else None
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.down4_att = AttentionDoubleConv(1024 // factor, 1024 // factor, use_attention=use_attention) if use_attention else None
        
        # 解码器（上采样路径）
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up1_att = AttentionDoubleConv(512 // factor, 512 // factor, use_attention=use_attention) if use_attention else None
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up2_att = AttentionDoubleConv(256 // factor, 256 // factor, use_attention=use_attention) if use_attention else None
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up3_att = AttentionDoubleConv(128 // factor, 128 // factor, use_attention=use_attention) if use_attention else None
        self.up4 = Up(128, 64, bilinear)
        self.up4_att = AttentionDoubleConv(64, 64, use_attention=use_attention) if use_attention else None
        
        # 输出层：如果使用残差学习，输出噪声；否则直接输出去噪图像
        self.outc = OutConv(64, n_classes)
        if self.use_refine_head:
            self.refine = RefineBlock(n_classes)
    
    def execute(self, x):
        # 保存原始输入（用于残差学习）
        if self.use_residual_learning:
            noisy_input = x
        
        # 编码路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        if self.down1_att:
            x2 = self.down1_att(x2)
        x3 = self.down2(x2)
        if self.down2_att:
            x3 = self.down2_att(x3)
        x4 = self.down3(x3)
        if self.down3_att:
            x4 = self.down3_att(x4)
        x5 = self.down4(x4)
        if self.down4_att:
            x5 = self.down4_att(x5)
        
        # 解码路径
        x = self.up1(x5, x4)
        if self.up1_att:
            x = self.up1_att(x)
        x = self.up2(x, x3)
        if self.up2_att:
            x = self.up2_att(x)
        x = self.up3(x, x2)
        if self.up3_att:
            x = self.up3_att(x)
        x = self.up4(x, x1)
        if self.up4_att:
            x = self.up4_att(x)
        
        logits = self.outc(x)
        
        # 残差学习：如果启用，模型学习噪声，然后从输入中减去
        if self.use_residual_learning:
            out = noisy_input - logits
        else:
            out = logits

        # 末端细化，缓解暗光/高噪下的残留噪点与模糊
        if self.use_refine_head and hasattr(self, "refine"):
            out = self.refine(out)
        return out
    
    def use_checkpointing(self):
        """启用梯度检查点以节省内存"""
        self.checkpoint = True


if __name__ == "__main__":
    # 测试模型
    model = UNet(n_channels=3, n_classes=3)
    x = jt.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print("Model test passed!")

