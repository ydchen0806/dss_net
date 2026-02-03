"""
信道分解模型 - U-Net架构
分离静态和动态分量
修复版本：移除动态decoder的通道数增强，保持对称结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DoubleConv(nn.Module):
    """双卷积块"""
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        norm_type: str = "batch",
        dropout: float = 0.0
    ):
        super().__init__()
        
        # 第一个卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, padding=1, bias=False)
        self.norm1 = self._get_norm(norm_type, out_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        
        # 第二个卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, padding=1, bias=False)
        self.norm2 = self._get_norm(norm_type, out_channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def _get_norm(self, norm_type: str, channels: int):
        if norm_type == "batch":
            return nn.BatchNorm2d(channels)
        elif norm_type == "instance":
            return nn.InstanceNorm2d(channels)
        elif norm_type == "layer":
            return nn.GroupNorm(1, channels)
        else:
            return nn.Identity()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        
        x = self.dropout(x)
        return x


class Down(nn.Module):
    """下采样块"""
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        norm_type: str = "batch",
        dropout: float = 0.0
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, norm_type, dropout)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """上采样块"""
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        norm_type: str = "batch",
        dropout: float = 0.0
    ):
        super().__init__()
        
        # 上采样
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, 
            kernel_size=2, stride=2
        )
        
        # 卷积
        self.conv = DoubleConv(in_channels, out_channels, norm_type, dropout)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 处理尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 拼接
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNetDecomposer(nn.Module):
    """
    改进的U-Net信道分解网络
    
    关键改进：
    1. 在bottleneck添加attention机制
    2. 静态和动态decoder保持对称结构（修复通道数不匹配问题）
    3. 添加refinement层提升输出质量
    
    输入: H_noise (2, H, W)  [real, imag]
    输出: H_static (2, H, W), H_dynamic (2, H, W)
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        depth: int = 4,
        norm_type: str = "batch",
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.depth = depth
        self.use_attention = use_attention
        
        # ============================================
        # Encoder (共享特征提取)
        # ============================================
        self.inc = DoubleConv(in_channels, base_channels, norm_type, dropout)
        
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for i in range(depth):
            self.down_blocks.append(
                Down(ch, ch * 2, norm_type, dropout)
            )
            ch = ch * 2
        
        # 🆕 Bottleneck attention（可选）
        if use_attention:
            self.bottleneck_attention = nn.Sequential(
                nn.Conv2d(ch, ch // 8, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch // 8, ch, 1),
                nn.Sigmoid()
            )
        
        # ============================================
        # Decoder for Static Component（标准结构）
        # ============================================
        self.up_static = nn.ModuleList()
        ch = base_channels * (2 ** depth)
        for i in range(depth):
            self.up_static.append(
                Up(ch, ch // 2, norm_type, dropout)
            )
            ch = ch // 2
        
        self.out_static = nn.Conv2d(base_channels, 2, kernel_size=1)
        
        # ============================================
        # Decoder for Dynamic Component（保持对称结构，修复通道数问题）
        # ============================================
        self.up_dynamic = nn.ModuleList()
        ch = base_channels * (2 ** depth)
        
        # ✅ 使用与静态decoder完全相同的通道配置
        for i in range(depth):
            self.up_dynamic.append(
                Up(ch, ch // 2, norm_type, dropout)
            )
            ch = ch // 2
        
        # 🆕 动态分量输出前添加refinement层
        self.dynamic_refine = DoubleConv(base_channels, base_channels, norm_type, dropout)
        self.out_dynamic = nn.Conv2d(base_channels, 2, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 2, H, W) - [real, imag]
        
        Returns:
            {
                'static': (B, 2, H, W),
                'dynamic': (B, 2, H, W),
                'total': (B, 2, H, W)
            }
        """
        # ============================================
        # Encoder (共享特征)
        # ============================================
        x1 = self.inc(x)
        
        skip_connections = [x1]
        x_down = x1
        
        for down in self.down_blocks:
            x_down = down(x_down)
            skip_connections.append(x_down)
        
        # 🆕 Apply attention at bottleneck
        bottleneck = skip_connections[-1]
        if self.use_attention:
            attention_weights = self.bottleneck_attention(bottleneck)
            bottleneck = bottleneck * attention_weights
            skip_connections[-1] = bottleneck
        
        # ============================================
        # Decoder for Static (稀疏分量)
        # ============================================
        x_static = skip_connections[-1]
        for i, up in enumerate(self.up_static):
            skip = skip_connections[-(i + 2)]
            x_static = up(x_static, skip)
        
        static = self.out_static(x_static)
        
        # ============================================
        # Decoder for Dynamic (低秩分量)
        # ============================================
        x_dynamic = skip_connections[-1]
        for i, up in enumerate(self.up_dynamic):
            skip = skip_connections[-(i + 2)]
            x_dynamic = up(x_dynamic, skip)
        
        # 🆕 Refinement for dynamic component
        x_dynamic = self.dynamic_refine(x_dynamic)
        dynamic = self.out_dynamic(x_dynamic)
        
        # ============================================
        # Total = Static + Dynamic
        # ============================================
        total = static + dynamic
        
        return {
            'static': static,
            'dynamic': dynamic,
            'total': total
        }


class UNetBaseline(nn.Module):
    """
    基线U-Net模型（不分离，直接重建）
    用于消融实验：证明分离动静态分量的必要性
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 64,
        depth: int = 4,
        norm_type: str = "batch",
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.depth = depth
        
        # ============================================
        # Encoder
        # ============================================
        self.inc = DoubleConv(in_channels, base_channels, norm_type, dropout)
        
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for i in range(depth):
            self.down_blocks.append(
                Down(ch, ch * 2, norm_type, dropout)
            )
            ch = ch * 2
        
        # ============================================
        # Decoder (单一重建路径)
        # ============================================
        self.up_blocks = nn.ModuleList()
        ch = base_channels * (2 ** depth)
        for i in range(depth):
            self.up_blocks.append(
                Up(ch, ch // 2, norm_type, dropout)
            )
            ch = ch // 2
        
        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 2, H, W) - [real, imag]
        
        Returns:
            {
                'total': (B, 2, H, W),  # 直接重建，无分离
                'static': None,
                'dynamic': None
            }
        """
        # Encoder
        x1 = self.inc(x)
        
        skip_connections = [x1]
        x_down = x1
        
        for down in self.down_blocks:
            x_down = down(x_down)
            skip_connections.append(x_down)
        
        # Decoder
        x_up = skip_connections[-1]
        for i, up in enumerate(self.up_blocks):
            skip = skip_connections[-(i + 2)]
            x_up = up(x_up, skip)
        
        total = self.out(x_up)
        
        # 为了兼容性，返回相同的字典格式
        return {
            'total': total,
            'static': None,
            'dynamic': None
        }


def count_parameters(model):
    """统计模型参数"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Model size: {total * 4 / 1024**2:.2f} MB (float32)")
    
    return total, trainable


if __name__ == '__main__':
    # 测试分解模型
    print("="*60)
    print("🔬 Testing Decomposition Model")
    print("="*60)
    
    model_decomp = UNetDecomposer(
        in_channels=2,
        base_channels=64,
        depth=4,
        dropout=0.1,
        use_attention=True
    )
    
    count_parameters(model_decomp)
    
    # 测试前向传播
    x = torch.randn(4, 2, 100, 150)  # (B, C, H, W)
    
    with torch.no_grad():
        output = model_decomp(x)
    
    print("\n📊 Output Shapes:")
    for key, val in output.items():
        if val is not None:
            print(f"  {key:10s}: {tuple(val.shape)}")
    
    # 测试基线模型
    print("\n" + "="*60)
    print("📊 Testing Baseline Model (Ablation)")
    print("="*60)
    
    model_baseline = UNetBaseline(
        in_channels=2,
        out_channels=2,
        base_channels=64,
        depth=4,
        dropout=0.1
    )
    
    count_parameters(model_baseline)
    
    with torch.no_grad():
        output_baseline = model_baseline(x)
    
    print("\n📊 Output Shapes:")
    for key, val in output_baseline.items():
        if val is not None:
            print(f"  {key:10s}: {tuple(val.shape)}")
        else:
            print(f"  {key:10s}: None (not decomposed)")