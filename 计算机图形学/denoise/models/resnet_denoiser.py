import jittor as jt
from jittor import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm(channels)
        self.conv2 = nn.Conv(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm(channels)
        self.act = nn.ReLU()

    def execute(self, x: jt.Var) -> jt.Var:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + x)


class ResidualDenoiser(nn.Module):
    """
    轻量级残差 CNN，适合 CPU：
    - 仅 3x3 卷积，无上/下采样
    - 若干残差块，保持分辨率
    输入: 4 通道 (noisy RGB + sample count)
    输出: 3 通道干净 RGB
    """

    def __init__(self, in_channels: int = 4, base_channels: int = 32, out_channels: int = 3, num_blocks: int = 5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm(base_channels),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([ResidualBlock(base_channels) for _ in range(num_blocks)])
        self.head = nn.Conv(base_channels, out_channels, 3, padding=1)

    def execute(self, x: jt.Var) -> jt.Var:
        out = self.stem(x)
        for block in self.blocks:
            out = block(out)
        out = self.head(out)
        return jt.clamp(out, 0.0, 1.0)

