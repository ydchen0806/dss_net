"""
数据增强工具
包含各种针对无线信道数据的增强方法
"""

import torch
import numpy as np
from typing import Tuple, Optional
import random


class TemporalSlicing:
    """时间切片增强"""
    
    def __init__(self, slice_length: int = 100, overlap: float = 0.5):
        self.slice_length = slice_length
        self.overlap = overlap
    
    def __call__(
        self,
        data: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            data: (T, 2, H, W) - 时间序列数据
        Returns:
            sliced: (T', 2, H, W) - 切片后的数据
        """
        T = data.shape[0]
        
        if T <= self.slice_length:
            return data
        
        stride = int(self.slice_length * (1 - self.overlap))
        start_idx = random.randint(0, max(0, T - self.slice_length))
        
        return data[start_idx:start_idx + self.slice_length]


class SpatialMasking:
    """空间遮掩增强"""
    
    def __init__(self, mask_ratio: float = 0.1, block_size: int = 5):
        self.mask_ratio = mask_ratio
        self.block_size = block_size
    
    def __call__(
        self,
        data: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            data: (2, H, W) or (T, 2, H, W)
        Returns:
            masked: same shape as data
        """
        masked = data.clone()
        
        if len(data.shape) == 3:
            # (2, H, W)
            _, H, W = data.shape
        else:
            # (T, 2, H, W)
            _, _, H, W = data.shape
        
        # 计算需要遮掩的块数
        num_blocks_h = H // self.block_size
        num_blocks_w = W // self.block_size
        total_blocks = num_blocks_h * num_blocks_w
        num_masked = int(total_blocks * self.mask_ratio)
        
        # 随机选择块
        masked_indices = random.sample(range(total_blocks), num_masked)
        
        for idx in masked_indices:
            block_h = idx // num_blocks_w
            block_w = idx % num_blocks_w
            
            h_start = block_h * self.block_size
            h_end = min(h_start + self.block_size, H)
            w_start = block_w * self.block_size
            w_end = min(w_start + self.block_size, W)
            
            if len(data.shape) == 3:
                masked[:, h_start:h_end, w_start:w_end] = 0
            else:
                masked[:, :, h_start:h_end, w_start:w_end] = 0
        
        return masked


class NoiseInjection:
    """噪声注入增强"""
    
    def __init__(self, snr_range: Tuple[float, float] = (20.0, 35.0)):
        self.snr_range = snr_range
    
    def __call__(
        self,
        data: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            data: (2, H, W) or (T, 2, H, W)
        Returns:
            noisy: same shape as data
        """
        # 随机选择SNR
        snr_db = random.uniform(*self.snr_range)
        snr_linear = 10 ** (snr_db / 10)
        
        # 计算信号功率
        signal_power = torch.mean(data ** 2)
        
        # 计算噪声功率
        noise_power = signal_power / snr_linear
        
        # 生成噪声
        noise = torch.randn_like(data) * torch.sqrt(noise_power)
        
        return data + noise


class PhaseRotation:
    """相位旋转增强（仅适用于复数表示）"""
    
    def __init__(self, max_angle: float = np.pi / 4):
        self.max_angle = max_angle
    
    def __call__(
        self,
        data: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            data: (2, H, W) or (T, 2, H, W) - [real, imag]
        Returns:
            rotated: same shape as data
        """
        # 随机旋转角度
        angle = random.uniform(-self.max_angle, self.max_angle)
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        if len(data.shape) == 3:
            # (2, H, W)
            real = data[0]
            imag = data[1]
            
            new_real = real * cos_a - imag * sin_a
            new_imag = real * sin_a + imag * cos_a
            
            return torch.stack([new_real, new_imag], dim=0)
        
        else:
            # (T, 2, H, W)
            real = data[:, 0]
            imag = data[:, 1]
            
            new_real = real * cos_a - imag * sin_a
            new_imag = real * sin_a + imag * cos_a
            
            return torch.stack([new_real, new_imag], dim=1)


class FrequencyPerturbation:
    """频域扰动增强"""
    
    def __init__(self, perturb_ratio: float = 0.1):
        self.perturb_ratio = perturb_ratio
    
    def __call__(
        self,
        data: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            data: (2, H, W) - [real, imag]
        Returns:
            perturbed: same shape as data
        """
        # 转换到频域
        complex_data = torch.complex(data[0], data[1])
        fft_data = torch.fft.fft2(complex_data)
        
        # 添加扰动
        magnitude = torch.abs(fft_data)
        phase = torch.angle(fft_data)
        
        # 扰动幅度
        magnitude_perturb = magnitude * (1 + self.perturb_ratio * torch.randn_like(magnitude))
        
        # 重构
        fft_perturbed = magnitude_perturb * torch.exp(1j * phase)
        
        # 转回时域
        complex_perturbed = torch.fft.ifft2(fft_perturbed)
        
        real = torch.real(complex_perturbed)
        imag = torch.imag(complex_perturbed)
        
        return torch.stack([real, imag], dim=0)


class Mixup:
    """Mixup数据增强"""
    
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
    
    def __call__(
        self,
        data1: torch.Tensor,
        data2: torch.Tensor,
        label1: torch.Tensor,
        label2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            data1: 第一个样本
            data2: 第二个样本
            label1: 第一个标签
            label2: 第二个标签
        Returns:
            mixed_data: 混合后的数据
            mixed_label: 混合后的标签
        """
        lam = np.random.beta(self.alpha, self.alpha)
        
        mixed_data = lam * data1 + (1 - lam) * data2
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_data, mixed_label


class CutMix:
    """CutMix数据增强"""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(
        self,
        data1: torch.Tensor,
        data2: torch.Tensor,
        label1: torch.Tensor,
        label2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            data1: (2, H, W)
            data2: (2, H, W)
            label1: 第一个标签
            label2: 第二个标签
        Returns:
            cutmixed_data: (2, H, W)
            cutmixed_label: 混合后的标签
        """
        _, H, W = data1.shape
        
        # 生成随机框
        lam = np.random.beta(self.alpha, self.alpha)
        
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        
        # 随机位置
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # 混合
        cutmixed_data = data1.clone()
        cutmixed_data[:, bby1:bby2, bbx1:bbx2] = data2[:, bby1:bby2, bbx1:bbx2]
        
        # 调整lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        cutmixed_label = lam * label1 + (1 - lam) * label2
        
        return cutmixed_data, cutmixed_label


class ComposedAugmentation:
    """组合多个增强方法"""
    
    def __init__(self, augmentations: list, probabilities: Optional[list] = None):
        """
        Args:
            augmentations: 增强方法列表
            probabilities: 每个增强方法的应用概率
        """
        self.augmentations = augmentations
        
        if probabilities is None:
            self.probabilities = [0.5] * len(augmentations)
        else:
            assert len(probabilities) == len(augmentations)
            self.probabilities = probabilities
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data: 输入数据
        Returns:
            augmented: 增强后的数据
        """
        augmented = data
        
        for aug, prob in zip(self.augmentations, self.probabilities):
            if random.random() < prob:
                augmented = aug(augmented)
        
        return augmented


def create_augmentation_pipeline(config: dict) -> ComposedAugmentation:
    """
    创建数据增强流水线
    
    Args:
        config: 配置字典
    Returns:
        pipeline: 增强流水线
    """
    augmentations = []
    probabilities = []
    
    aug_config = config.get('augmentation', {})
    
    # 时间切片
    if aug_config.get('temporal_slice', False):
        augmentations.append(TemporalSlicing())
        probabilities.append(0.3)
    
    # 空间遮掩
    if 'spatial_masking' in aug_config:
        mask_config = aug_config['spatial_masking']
        augmentations.append(SpatialMasking(
            mask_ratio=mask_config.get('mask_ratio', 0.1)
        ))
        probabilities.append(mask_config.get('prob', 0.3))
    
    # 噪声注入
    if 'noise_injection' in aug_config:
        noise_config = aug_config['noise_injection']
        augmentations.append(NoiseInjection(
            snr_range=noise_config.get('snr_range', [20, 35])
        ))
        probabilities.append(noise_config.get('prob', 0.3))
    
    # 相位旋转
    if aug_config.get('phase_rotation', False):
        augmentations.append(PhaseRotation())
        probabilities.append(0.2)
    
    # 频域扰动
    if aug_config.get('frequency_perturbation', False):
        augmentations.append(FrequencyPerturbation())
        probabilities.append(0.1)
    
    pipeline = ComposedAugmentation(augmentations, probabilities)
    
    return pipeline


# 示例使用
if __name__ == '__main__':
    # 创建测试数据
    data = torch.randn(2, 100, 150)  # [real, imag], H, W
    
    print("Original shape:", data.shape)
    print("Original mean:", data.mean().item())
    print("Original std:", data.std().item())
    
    # 测试各种增强
    print("\n" + "="*60)
    print("Testing augmentations...")
    print("="*60)
    
    # 空间遮掩
    spatial_mask = SpatialMasking(mask_ratio=0.1)
    masked = spatial_mask(data)
    print(f"\nSpatial Masking: {torch.sum(masked == 0).item()} zeros")
    
    # 噪声注入
    noise_inject = NoiseInjection(snr_range=(25, 30))
    noisy = noise_inject(data)
    print(f"Noise Injection: SNR ≈ {10 * torch.log10(data.var() / (noisy - data).var()):.2f} dB")
    
    # 相位旋转
    phase_rot = PhaseRotation(max_angle=np.pi/6)
    rotated = phase_rot(data)
    print(f"Phase Rotation: magnitude preserved = {torch.allclose(torch.abs(torch.complex(data[0], data[1])), torch.abs(torch.complex(rotated[0], rotated[1])), atol=1e-5)}")
    
    # 频域扰动
    freq_perturb = FrequencyPerturbation(perturb_ratio=0.05)
    perturbed = freq_perturb(data)
    print(f"Frequency Perturbation: correlation = {torch.corrcoef(torch.stack([data.flatten(), perturbed.flatten()]))[0, 1]:.4f}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
