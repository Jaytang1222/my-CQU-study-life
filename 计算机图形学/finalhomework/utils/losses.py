"""
损失函数
"""
import jittor as jt
from jittor import nn
import numpy as np


class L1Loss(nn.Module):
    def execute(self, pred, target):
        return jt.mean(jt.abs(pred - target))


class L2Loss(nn.Module):
    def execute(self, pred, target):
        return jt.mean((pred - target) ** 2)


class SSIMLoss(nn.Module):
    """改进版 SSIM 损失：基于patch的SSIM，对纯色区域更敏感"""
    def __init__(self, window_size=11, patch_based=True):
        super().__init__()
        self.window_size = window_size
        self.patch_based = patch_based
    
    def execute(self, pred, target):
        if self.patch_based:
            # 基于patch的SSIM（更准确）
            return self._patch_ssim_loss(pred, target)
        else:
            # 全局SSIM（简化版，向后兼容）
            mu1 = jt.mean(pred)
            mu2 = jt.mean(target)
            sigma1 = jt.mean((pred - mu1) ** 2)
            sigma2 = jt.mean((target - mu2) ** 2)
            sigma12 = jt.mean((pred - mu1) * (target - mu2))
            C1, C2 = 0.01 ** 2, 0.03 ** 2
            ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)
            )
            return 1 - ssim
    
    def _patch_ssim_loss(self, pred, target):
        """基于patch的SSIM损失，对纯色区域更敏感"""
        # 使用平均池化模拟滑动窗口
        kernel_size = min(self.window_size, pred.shape[-1] // 4, pred.shape[-2] // 4)
        if kernel_size < 3:
            kernel_size = 3
        
        # 计算局部均值和方差
        mu1 = nn.avg_pool2d(pred, kernel_size, stride=1, padding=kernel_size//2)
        mu2 = nn.avg_pool2d(target, kernel_size, stride=1, padding=kernel_size//2)
        
        sigma1_sq = nn.avg_pool2d(pred * pred, kernel_size, stride=1, padding=kernel_size//2) - mu1 * mu1
        sigma2_sq = nn.avg_pool2d(target * target, kernel_size, stride=1, padding=kernel_size//2) - mu2 * mu2
        sigma12 = nn.avg_pool2d(pred * target, kernel_size, stride=1, padding=kernel_size//2) - mu1 * mu2
        
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return 1 - jt.mean(ssim_map)


class PerceptualLoss(nn.Module):
    """
    感知损失：使用多尺度特征匹配，减少结构失真
    通过在不同尺度上比较特征，更好地保持图像结构
    """
    def __init__(self, perceptual_weight=1.0):
        super().__init__()
        self.perceptual_weight = perceptual_weight
    
    def compute_features(self, img, scale=1):
        """计算图像的多尺度特征"""
        # 使用简单的卷积特征提取（替代VGG）
        # 在不同尺度上提取特征
        features = []
        
        # 原始尺度
        if scale >= 1:
            # 使用平均池化提取局部特征
            feat = nn.avg_pool2d(img, kernel_size=3, stride=1, padding=1)
            features.append(feat)
        
        # 下采样尺度
        if scale >= 2:
            feat = nn.avg_pool2d(img, kernel_size=2, stride=2)
            feat = nn.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)
            features.append(feat)
        
        # 更下采样尺度
        if scale >= 3:
            feat = nn.avg_pool2d(img, kernel_size=4, stride=4)
            feat = nn.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)
            features.append(feat)
        
        return features
    
    def execute(self, pred, target):
        """计算多尺度感知损失"""
        pred_features = self.compute_features(pred, scale=3)
        target_features = self.compute_features(target, scale=3)
        
        # 在每个尺度上计算L2损失
        loss = 0.0
        for pf, tf in zip(pred_features, target_features):
            loss += jt.mean((pf - tf) ** 2)
        
        return self.perceptual_weight * loss / len(pred_features)


class FrequencyLoss(nn.Module):
    """频率域损失：在频域中比较，减少伪影和失真"""
    def __init__(self, freq_weight=1.0):
        super().__init__()
        self.freq_weight = freq_weight
    
    def fft_loss(self, pred, target):
        """使用FFT在频率域计算损失"""
        # 转换为灰度图
        pred_gray = jt.mean(pred, dim=1, keepdims=True)
        target_gray = jt.mean(target, dim=1, keepdims=True)
        
        # FFT（简化版：使用实部）
        # 注意：Jittor可能没有直接的FFT，这里使用简化版本
        # 实际可以使用numpy的FFT或实现DCT
        
        # 简化：使用高频和低频分离
        # 低频：大核平均池化
        pred_low = nn.avg_pool2d(pred_gray, kernel_size=5, stride=1, padding=2)
        target_low = nn.avg_pool2d(target_gray, kernel_size=5, stride=1, padding=2)
        
        # 高频：原图 - 低频
        pred_high = pred_gray - pred_low
        target_high = target_gray - target_low
        
        # 分别计算低频和高频损失
        low_loss = jt.mean((pred_low - target_low) ** 2)
        high_loss = jt.mean((pred_high - target_high) ** 2)
        
        # 高频权重更高，因为失真通常在细节（高频）中更明显
        return 0.3 * low_loss + 0.7 * high_loss
    
    def execute(self, pred, target):
        """计算频率域损失"""
        return self.freq_weight * self.fft_loss(pred, target)


class MultiScaleLoss(nn.Module):
    """多尺度损失：在不同尺度上计算损失，减少失真"""
    def __init__(self, ms_weight=1.0, scales=[1, 2, 4]):
        super().__init__()
        self.ms_weight = ms_weight
        self.scales = scales
    
    def execute(self, pred, target):
        """在多尺度上计算L1损失"""
        loss = 0.0
        for scale in self.scales:
            if scale == 1:
                p, t = pred, target
            else:
                # 下采样
                p = nn.avg_pool2d(pred, kernel_size=scale, stride=scale)
                t = nn.avg_pool2d(target, kernel_size=scale, stride=scale)
            
            # L1损失
            loss += jt.mean(jt.abs(p - t))
        
        return self.ms_weight * loss / len(self.scales)


class SmoothLoss(nn.Module):
    """平滑损失（Total Variation Loss），用于减少纯色区域的噪声"""
    def __init__(self, tv_weight=1.0):
        super().__init__()
        self.tv_weight = tv_weight
    
    def execute(self, pred):
        """
        计算Total Variation Loss，鼓励图像平滑
        对纯色区域特别有效，可以减少雪花状噪声
        """
        batch_size = pred.shape[0]
        # 计算水平和垂直方向的梯度
        h_tv = jt.mean(jt.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]))
        w_tv = jt.mean(jt.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]))
        tv_loss = h_tv + w_tv
        return self.tv_weight * tv_loss


class ColorLoss(nn.Module):
    """颜色损失：在Lab颜色空间中计算损失，对色差更敏感"""
    def __init__(self, color_weight=1.0):
        super().__init__()
        self.color_weight = color_weight
    
    def rgb_to_lab(self, rgb):
        """将RGB转换为Lab颜色空间（简化版，使用线性近似）"""
        # RGB转XYZ（简化版，假设sRGB）
        # 输入范围[0,1]
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        
        # 线性化
        def f(x):
            return jt.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
        
        r_lin = f(r)
        g_lin = f(g)
        b_lin = f(b)
        
        # RGB to XYZ (sRGB D65)
        x = 0.4124564 * r_lin + 0.3575761 * g_lin + 0.1804375 * b_lin
        y = 0.2126729 * r_lin + 0.7151522 * g_lin + 0.0721750 * b_lin
        z = 0.0193339 * r_lin + 0.1191920 * g_lin + 0.9503041 * b_lin
        
        # XYZ to Lab (D65 white point)
        x_norm = x / 0.95047
        y_norm = y / 1.00000
        z_norm = z / 1.08883
        
        def f_xyz(t):
            return jt.where(t > 0.008856, t ** (1.0/3.0), (7.787 * t + 16.0/116.0))
        
        fx = f_xyz(x_norm)
        fy = f_xyz(y_norm)
        fz = f_xyz(z_norm)
        
        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)
        
        return jt.concat([L, a, b], dim=1)
    
    def execute(self, pred, target):
        """计算Lab颜色空间中的L1损失，对色差更敏感"""
        pred_lab = self.rgb_to_lab(pred)
        target_lab = self.rgb_to_lab(target)
        
        # 在Lab空间中，a和b通道代表色度，对色差更敏感
        # 可以只使用a和b通道，或者给a、b更高的权重
        lab_diff = jt.abs(pred_lab - target_lab)
        
        # 给色度通道（a, b）更高的权重
        l_loss = jt.mean(lab_diff[:, 0:1])  # 亮度
        ab_loss = jt.mean(lab_diff[:, 1:3])  # 色度（a和b通道）
        
        # 色度损失权重更高
        return self.color_weight * (0.3 * l_loss + 0.7 * ab_loss)


class ChromaLoss(nn.Module):
    """色度损失：只关注颜色差异，忽略亮度差异"""
    def __init__(self, chroma_weight=1.0):
        super().__init__()
        self.chroma_weight = chroma_weight
    
    def execute(self, pred, target):
        """
        计算色度损失：在RGB空间中，归一化后计算颜色差异
        这样可以忽略亮度差异，只关注颜色差异
        """
        # 计算每个像素的亮度
        pred_lum = jt.mean(pred, dim=1, keepdims=True)  # [B, 1, H, W]
        target_lum = jt.mean(target, dim=1, keepdims=True)
        
        # 避免除零
        pred_lum = jt.maximum(pred_lum, 1e-6)
        target_lum = jt.maximum(target_lum, 1e-6)
        
        # 归一化到色度（chroma）
        pred_chroma = pred / pred_lum  # [B, 3, H, W]
        target_chroma = target / target_lum
        
        # 计算色度差异
        chroma_diff = jt.abs(pred_chroma - target_chroma)
        return self.chroma_weight * jt.mean(chroma_diff)


class GradientLoss(nn.Module):
    """梯度损失：保持图像的边缘和细节"""
    def __init__(self, grad_weight=1.0):
        super().__init__()
        self.grad_weight = grad_weight
    
    def compute_gradient(self, img):
        """计算图像的梯度（简化版，使用差分）"""
        # 转换为灰度图
        gray = jt.mean(img, dim=1, keepdims=True)  # [B, 1, H, W]
        B, C, H, W = gray.shape
        
        # 计算水平和垂直方向的梯度（使用差分）
        # grad_h: [B, 1, H-1, W] - 垂直方向梯度
        grad_h_diff = gray[:, :, 1:, :] - gray[:, :, :-1, :]
        # grad_w: [B, 1, H, W-1] - 水平方向梯度
        grad_w_diff = gray[:, :, :, 1:] - gray[:, :, :, :-1]
        
        # 填充到原始尺寸：复制边界值
        # 对于grad_h，在底部填充最后一行
        grad_h_last = grad_h_diff[:, :, -1:, :]  # [B, 1, 1, W]
        grad_h = jt.concat([grad_h_diff, grad_h_last], dim=2)  # [B, 1, H, W]
        
        # 对于grad_w，在右侧填充最后一列
        grad_w_last = grad_w_diff[:, :, :, -1:]  # [B, 1, H, 1]
        grad_w = jt.concat([grad_w_diff, grad_w_last], dim=3)  # [B, 1, H, W]
        
        # 计算梯度幅值
        grad_mag = jt.sqrt(grad_h ** 2 + grad_w ** 2 + 1e-6)
        return grad_mag
    
    def execute(self, pred, target):
        """
        计算梯度损失：保持pred和target的梯度一致性
        这样可以保持边缘和细节，避免过度平滑
        """
        pred_grad = self.compute_gradient(pred)
        target_grad = self.compute_gradient(target)
        
        # L1损失在梯度上，更关注边缘
        grad_loss = jt.mean(jt.abs(pred_grad - target_grad))
        return self.grad_weight * grad_loss


class EdgePreservingLoss(nn.Module):
    """边缘保持损失：只在平滑区域应用平滑，保护边缘"""
    def __init__(self, edge_weight=1.0, threshold=0.1):
        super().__init__()
        self.edge_weight = edge_weight
        self.threshold = threshold
    
    def compute_edge_mask(self, img):
        """计算边缘掩码：边缘区域为1，平滑区域为0"""
        # 转换为灰度图
        gray = jt.mean(img, dim=1, keepdims=True)  # [B, 1, H, W]
        
        # 计算梯度（简化版）
        # grad_h: [B, 1, H-1, W] - 垂直方向梯度
        grad_h_diff = gray[:, :, 1:, :] - gray[:, :, :-1, :]
        # grad_w: [B, 1, H, W-1] - 水平方向梯度
        grad_w_diff = gray[:, :, :, 1:] - gray[:, :, :, :-1]
        
        # 填充到原始尺寸：复制边界值
        # 对于grad_h，在底部填充最后一行
        grad_h_last = grad_h_diff[:, :, -1:, :]  # [B, 1, 1, W]
        grad_h = jt.concat([grad_h_diff, grad_h_last], dim=2)  # [B, 1, H, W]
        
        # 对于grad_w，在右侧填充最后一列
        grad_w_last = grad_w_diff[:, :, :, -1:]  # [B, 1, H, 1]
        grad_w = jt.concat([grad_w_diff, grad_w_last], dim=3)  # [B, 1, H, W]
        
        # 计算梯度幅值
        grad_mag = jt.sqrt(grad_h ** 2 + grad_w ** 2 + 1e-6)
        
        # 边缘掩码：梯度大于阈值的地方是边缘（1），否则是平滑区域（0）
        edge_mask = (grad_mag > self.threshold).float()
        return edge_mask
    
    def execute(self, pred, target):
        """
        边缘保持损失：只在平滑区域鼓励平滑，保护边缘区域
        """
        # 使用GT计算边缘掩码（更准确）
        edge_mask = self.compute_edge_mask(target)
        smooth_mask = 1.0 - edge_mask
        
        # 只在平滑区域计算TV损失
        pred_h_diff = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        pred_w_diff = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        
        # 填充掩码
        smooth_mask_h = smooth_mask[:, :, :-1, :]
        smooth_mask_w = smooth_mask[:, :, :, :-1]
        
        # 只在平滑区域计算TV
        tv_h = jt.mean(jt.abs(pred_h_diff) * smooth_mask_h)
        tv_w = jt.mean(jt.abs(pred_w_diff) * smooth_mask_w)
        
        return self.edge_weight * (tv_h + tv_w)


class CombinedLoss(nn.Module):
    """可组合 L1/L2/SSIM/平滑/颜色/梯度/边缘保持/感知/频率/多尺度损失"""
    def __init__(self, l1_weight=0.15, l2_weight=0.15, ssim_weight=0.2, smooth_weight=0.05, 
                 color_weight=0.08, chroma_weight=0.05, grad_weight=0.1, edge_weight=0.05,
                 perceptual_weight=0.1, freq_weight=0.05, multiscale_weight=0.07, use_patch_ssim=True):
        super().__init__()
        self.l1 = L1Loss()
        self.l2 = L2Loss()
        self.ssim = SSIMLoss(patch_based=use_patch_ssim)
        self.smooth = SmoothLoss()
        self.color = ColorLoss()
        self.chroma = ChromaLoss()
        self.grad = GradientLoss()
        self.edge = EdgePreservingLoss()
        self.perceptual = PerceptualLoss()
        self.freq = FrequencyLoss()
        self.multiscale = MultiScaleLoss()
        self.w1 = l1_weight
        self.w2 = l2_weight
        self.ws = ssim_weight
        self.w_smooth = smooth_weight
        self.w_color = color_weight
        self.w_chroma = chroma_weight
        self.w_grad = grad_weight
        self.w_edge = edge_weight
        self.w_perceptual = perceptual_weight
        self.w_freq = freq_weight
        self.w_multiscale = multiscale_weight

    def execute(self, pred, target):
        # 基础损失
        base_loss = (
            self.w1 * self.l1(pred, target) +
            self.w2 * self.l2(pred, target) +
            self.ws * self.ssim(pred, target)
        )
        # 平滑损失（减少，避免过度平滑）
        smooth_loss = self.w_smooth * self.smooth(pred)
        # 颜色损失（Lab颜色空间，对色差更敏感）
        color_loss = self.w_color * self.color(pred, target)
        # 色度损失（只关注颜色，忽略亮度）
        chroma_loss = self.w_chroma * self.chroma(pred, target)
        # 梯度损失（保持边缘和细节）
        grad_loss = self.w_grad * self.grad(pred, target)
        # 边缘保持损失（只在平滑区域平滑，保护边缘）
        edge_loss = self.w_edge * self.edge(pred, target)
        # 感知损失（多尺度特征匹配，减少结构失真）
        perceptual_loss = self.w_perceptual * self.perceptual(pred, target)
        # 频率域损失（减少伪影和失真）
        freq_loss = self.w_freq * self.freq(pred, target)
        # 多尺度损失（在不同尺度上保持一致性）
        multiscale_loss = self.w_multiscale * self.multiscale(pred, target)
        return (base_loss + smooth_loss + color_loss + chroma_loss + grad_loss + 
                edge_loss + perceptual_loss + freq_loss + multiscale_loss)


def get_loss_function(loss_type="l2", weights=None):
    loss_type = loss_type.lower()
    if loss_type == "l1":
        return L1Loss()
    if loss_type == "l2":
        return L2Loss()
    if loss_type == "ssim":
        return SSIMLoss(patch_based=True)
    if loss_type == "perceptual":
        return PerceptualLoss()
    if loss_type == "combined":
        w = weights or {"l1": 0.15, "l2": 0.15, "ssim": 0.2, "smooth": 0.05, "color": 0.08, 
                       "chroma": 0.05, "grad": 0.1, "edge": 0.05, "perceptual": 0.1, 
                       "freq": 0.05, "multiscale": 0.07}
        return CombinedLoss(
            l1_weight=w.get("l1", 0.15),
            l2_weight=w.get("l2", 0.15),
            ssim_weight=w.get("ssim", 0.2),
            smooth_weight=w.get("smooth", 0.05),
            color_weight=w.get("color", 0.08),
            chroma_weight=w.get("chroma", 0.05),
            grad_weight=w.get("grad", 0.1),
            edge_weight=w.get("edge", 0.05),
            perceptual_weight=w.get("perceptual", 0.1),
            freq_weight=w.get("freq", 0.05),
            multiscale_weight=w.get("multiscale", 0.07),
            use_patch_ssim=w.get("use_patch_ssim", True)
        )
    raise ValueError(f"Unknown loss type: {loss_type}")

