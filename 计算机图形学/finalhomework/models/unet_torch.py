"""
PyTorch implementation of the UNet model compatible with the jittor UNet design.
This is intended as a fallback when Jittor is not available.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, use_residual=True):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.use_residual = use_residual and in_ch == out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            return out + x
        return out


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = F.adaptive_avg_pool2d(x, 1)
        max_out = F.adaptive_max_pool2d(x, 1)
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        out = avg_out + max_out
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return x * out


class AttentionDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, use_attention=True):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.conv = DoubleConv(in_ch, out_ch, mid_ch, use_residual=False)
        self.use_attention = use_attention
        if use_attention and out_ch >= 16:
            self.channel_att = ChannelAttention(out_ch)
            self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        if self.use_attention and hasattr(self, 'channel_att'):
            x = self.channel_att(x)
            x = self.spatial_att(x)
        return x


class RefineBlock(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
        )
        self.res_scale = 0.5

    def forward(self, x):
        refine = self.block(x)
        return x + self.res_scale * refine


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # padding alignment
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=True, use_attention=True, use_residual_learning=True, use_refine_head=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_residual_learning = use_residual_learning
        self.use_refine_head = use_refine_head

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

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up1_att = AttentionDoubleConv(512 // factor, 512 // factor, use_attention=use_attention) if use_attention else None
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up2_att = AttentionDoubleConv(256 // factor, 256 // factor, use_attention=use_attention) if use_attention else None
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up3_att = AttentionDoubleConv(128 // factor, 128 // factor, use_attention=use_attention) if use_attention else None
        self.up4 = Up(128, 64, bilinear)
        self.up4_att = AttentionDoubleConv(64, 64, use_attention=use_attention) if use_attention else None

        self.outc = OutConv(64, n_classes)
        if self.use_refine_head:
            self.refine = RefineBlock(n_classes)

    def forward(self, x):
        if self.use_residual_learning:
            noisy_input = x

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
        if self.use_residual_learning:
            out = noisy_input - logits
        else:
            out = logits
        if self.use_refine_head and hasattr(self, 'refine'):
            out = self.refine(out)
        return out


if __name__ == '__main__':
    model = UNet()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(x.shape, y.shape)
