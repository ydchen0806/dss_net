"""
可视化工具 - 信道分解结果可视化
用于TensorBoard展示
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
from typing import Dict, Optional
import io
from PIL import Image


def complex_to_magnitude(tensor: torch.Tensor) -> torch.Tensor:
    """
    将复数表示的张量转换为幅度
    
    Args:
        tensor: (B, 2, H, W) - [real, imag]
    
    Returns:
        magnitude: (B, H, W)
    """
    real = tensor[:, 0, :, :]
    imag = tensor[:, 1, :, :]
    magnitude = torch.sqrt(real**2 + imag**2)
    return magnitude


def complex_to_phase(tensor: torch.Tensor) -> torch.Tensor:
    """
    将复数表示的张量转换为相位
    
    Args:
        tensor: (B, 2, H, W) - [real, imag]
    
    Returns:
        phase: (B, H, W) - 范围[-π, π]
    """
    real = tensor[:, 0, :, :]
    imag = tensor[:, 1, :, :]
    phase = torch.atan2(imag, real)
    return phase


def normalize_to_01(tensor: torch.Tensor) -> torch.Tensor:
    """
    将张量归一化到[0, 1]范围
    
    Args:
        tensor: (B, H, W) or (H, W)
    
    Returns:
        normalized: same shape as input
    """
    # 计算最小值和最大值（保持维度）
    if tensor.dim() == 3:
        # (B, H, W)
        batch_min = tensor.view(tensor.size(0), -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
        batch_max = tensor.view(tensor.size(0), -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
        
        normalized = (tensor - batch_min) / (batch_max - batch_min + 1e-8)
    else:
        # (H, W)
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        normalized = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)
    
    return normalized


def create_comparison_grid(
    inputs: torch.Tensor,
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    is_baseline: bool = False,
    num_samples: int = 4,
    mode: str = 'magnitude'
) -> np.ndarray:
    """
    创建对比网格图像
    
    Args:
        inputs: (B, 2, H, W) - 输入的带噪声信道
        pred: {
            'static': (B, 2, H, W) or None,
            'dynamic': (B, 2, H, W) or None,
            'total': (B, 2, H, W)
        }
        target: {
            'static': (B, 2, H, W),
            'dynamic': (B, 2, H, W),
            'target': (B, 2, H, W)
        }
        is_baseline: 是否为基线模型
        num_samples: 展示的样本数量
        mode: 'magnitude' or 'phase'
    
    Returns:
        grid_image: (H, W, 3) - RGB图像
    """
    # 选取前num_samples个样本
    batch_size = inputs.size(0)
    num_samples = min(num_samples, batch_size)
    
    # 转换为幅度或相位
    if mode == 'magnitude':
        convert_fn = complex_to_magnitude
        cmap = 'viridis'
        title_suffix = '(Magnitude)'
    else:
        convert_fn = complex_to_phase
        cmap = 'twilight'  # 相位用循环colormap
        title_suffix = '(Phase)'
    
    # 转换数据
    input_mag = convert_fn(inputs[:num_samples]).cpu().numpy()
    
    if is_baseline:
        # 基线模型：只有输入、预测总量、真值总量
        pred_total = convert_fn(pred['total'][:num_samples]).cpu().numpy()
        target_total = convert_fn(target['target'][:num_samples]).cpu().numpy()
        error_total = np.abs(pred_total - target_total)
        
        # 布局：4行（样本）x 4列（输入、预测、真值、误差）
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # 输入
            im0 = axes[i, 0].imshow(input_mag[i], cmap=cmap, aspect='auto')
            axes[i, 0].set_title(f'Sample {i+1}: Input {title_suffix}')
            axes[i, 0].axis('off')
            plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)
            
            # 预测总量
            im1 = axes[i, 1].imshow(pred_total[i], cmap=cmap, aspect='auto')
            axes[i, 1].set_title('Predicted Total')
            axes[i, 1].axis('off')
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
            
            # 真值总量
            im2 = axes[i, 2].imshow(target_total[i], cmap=cmap, aspect='auto')
            axes[i, 2].set_title('Ground Truth')
            axes[i, 2].axis('off')
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
            
            # 误差
            im3 = axes[i, 3].imshow(error_total[i], cmap='hot', aspect='auto')
            axes[i, 3].set_title('Absolute Error')
            axes[i, 3].axis('off')
            plt.colorbar(im3, ax=axes[i, 3], fraction=0.046)
    
    else:
        # 分解模型：展示静态、动态分量
        pred_static = convert_fn(pred['static'][:num_samples]).cpu().numpy()
        pred_dynamic = convert_fn(pred['dynamic'][:num_samples]).cpu().numpy()
        pred_total = convert_fn(pred['total'][:num_samples]).cpu().numpy()
        
        target_static = convert_fn(target['static'][:num_samples]).cpu().numpy()
        target_dynamic = convert_fn(target['dynamic'][:num_samples]).cpu().numpy()
        target_total = convert_fn(target['target'][:num_samples]).cpu().numpy()
        
        # 计算误差
        error_static = np.abs(pred_static - target_static)
        error_dynamic = np.abs(pred_dynamic - target_dynamic)
        error_total = np.abs(pred_total - target_total)
        
        # 布局：num_samples行 x 10列
        # 列：输入 | 预测静态 | 真值静态 | 静态误差 | 预测动态 | 真值动态 | 动态误差 | 预测总量 | 真值总量 | 总误差
        fig, axes = plt.subplots(num_samples, 10, figsize=(40, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            col = 0
            
            # 输入
            im = axes[i, col].imshow(input_mag[i], cmap=cmap, aspect='auto')
            axes[i, col].set_title(f'Sample {i+1}\nInput {title_suffix}', fontsize=10)
            axes[i, col].axis('off')
            plt.colorbar(im, ax=axes[i, col], fraction=0.046, pad=0.04)
            col += 1
            
            # 预测静态
            im = axes[i, col].imshow(pred_static[i], cmap=cmap, aspect='auto')
            axes[i, col].set_title('Pred\nStatic', fontsize=10)
            axes[i, col].axis('off')
            plt.colorbar(im, ax=axes[i, col], fraction=0.046, pad=0.04)
            col += 1
            
            # 真值静态
            im = axes[i, col].imshow(target_static[i], cmap=cmap, aspect='auto')
            axes[i, col].set_title('GT\nStatic', fontsize=10)
            axes[i, col].axis('off')
            plt.colorbar(im, ax=axes[i, col], fraction=0.046, pad=0.04)
            col += 1
            
            # 静态误差
            im = axes[i, col].imshow(error_static[i], cmap='hot', aspect='auto')
            axes[i, col].set_title('Error\nStatic', fontsize=10)
            axes[i, col].axis('off')
            plt.colorbar(im, ax=axes[i, col], fraction=0.046, pad=0.04)
            col += 1
            
            # 预测动态
            im = axes[i, col].imshow(pred_dynamic[i], cmap=cmap, aspect='auto')
            axes[i, col].set_title('Pred\nDynamic', fontsize=10)
            axes[i, col].axis('off')
            plt.colorbar(im, ax=axes[i, col], fraction=0.046, pad=0.04)
            col += 1
            
            # 真值动态
            im = axes[i, col].imshow(target_dynamic[i], cmap=cmap, aspect='auto')
            axes[i, col].set_title('GT\nDynamic', fontsize=10)
            axes[i, col].axis('off')
            plt.colorbar(im, ax=axes[i, col], fraction=0.046, pad=0.04)
            col += 1
            
            # 动态误差
            im = axes[i, col].imshow(error_dynamic[i], cmap='hot', aspect='auto')
            axes[i, col].set_title('Error\nDynamic', fontsize=10)
            axes[i, col].axis('off')
            plt.colorbar(im, ax=axes[i, col], fraction=0.046, pad=0.04)
            col += 1
            
            # 预测总量
            im = axes[i, col].imshow(pred_total[i], cmap=cmap, aspect='auto')
            axes[i, col].set_title('Pred\nTotal', fontsize=10)
            axes[i, col].axis('off')
            plt.colorbar(im, ax=axes[i, col], fraction=0.046, pad=0.04)
            col += 1
            
            # 真值总量
            im = axes[i, col].imshow(target_total[i], cmap=cmap, aspect='auto')
            axes[i, col].set_title('GT\nTotal', fontsize=10)
            axes[i, col].axis('off')
            plt.colorbar(im, ax=axes[i, col], fraction=0.046, pad=0.04)
            col += 1
            
            # 总误差
            im = axes[i, col].imshow(error_total[i], cmap='hot', aspect='auto')
            axes[i, col].set_title('Error\nTotal', fontsize=10)
            axes[i, col].axis('off')
            plt.colorbar(im, ax=axes[i, col], fraction=0.046, pad=0.04)
            col += 1
    
    plt.tight_layout()
    
    # 转换为numpy数组
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    
    plt.close(fig)
    buf.close()
    
    return img_array


def create_error_histogram(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    is_baseline: bool = False
) -> np.ndarray:
    """
    创建误差分布直方图
    
    Args:
        pred: 预测结果
        target: 真值
        is_baseline: 是否为基线模型
    
    Returns:
        histogram_image: (H, W, 3) - RGB图像
    """
    # 计算误差
    if is_baseline:
        error_total = (pred['total'] - target['target']).flatten().cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 误差直方图
        axes[0].hist(error_total, bins=100, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_title('Total Reconstruction Error Distribution')
        axes[0].set_xlabel('Error')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # 累积分布
        axes[1].hist(error_total, bins=100, cumulative=True, density=True, 
                     alpha=0.7, color='green', edgecolor='black')
        axes[1].set_title('Cumulative Distribution')
        axes[1].set_xlabel('Error')
        axes[1].set_ylabel('Cumulative Probability')
        axes[1].grid(True, alpha=0.3)
    
    else:
        error_static = (pred['static'] - target['static']).flatten().cpu().numpy()
        error_dynamic = (pred['dynamic'] - target['dynamic']).flatten().cpu().numpy()
        error_total = (pred['total'] - target['target']).flatten().cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        
        # 静态分量误差
        axes[0].hist(error_static, bins=100, alpha=0.7, color='red', edgecolor='black', label='Static')
        axes[0].set_title('Static Component Error Distribution')
        axes[0].set_xlabel('Error')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 动态分量误差
        axes[1].hist(error_dynamic, bins=100, alpha=0.7, color='blue', edgecolor='black', label='Dynamic')
        axes[1].set_title('Dynamic Component Error Distribution')
        axes[1].set_xlabel('Error')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 总重建误差
        axes[2].hist(error_total, bins=100, alpha=0.7, color='green', edgecolor='black', label='Total')
        axes[2].set_title('Total Reconstruction Error Distribution')
        axes[2].set_xlabel('Error')
        axes[2].set_ylabel('Frequency')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 转换为numpy数组
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    
    plt.close(fig)
    buf.close()
    
    return img_array


def create_temporal_variation_plot(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    is_baseline: bool = False,
    dim: int = -1
) -> Optional[np.ndarray]:
    """
    创建时间变化曲线图（沿W或H维度）
    
    Args:
        pred: 预测结果
        target: 真值
        is_baseline: 是否为基线模型
        dim: 计算变化的维度（-1表示W，-2表示H）
    
    Returns:
        plot_image: (H, W, 3) - RGB图像，如果是基线模型则返回None
    """
    if is_baseline:
        return None
    
    # 计算相邻时刻的变化（取第一个样本）
    static_pred = complex_to_magnitude(pred['static'][0:1]).squeeze(0).cpu().numpy()  # (H, W)
    dynamic_pred = complex_to_magnitude(pred['dynamic'][0:1]).squeeze(0).cpu().numpy()
    
    static_gt = complex_to_magnitude(target['static'][0:1]).squeeze(0).cpu().numpy()
    dynamic_gt = complex_to_magnitude(target['dynamic'][0:1]).squeeze(0).cpu().numpy()
    
    if dim == -1:
        # W维度：对H取平均，得到(W,)
        static_pred_curve = static_pred.mean(axis=0)
        dynamic_pred_curve = dynamic_pred.mean(axis=0)
        static_gt_curve = static_gt.mean(axis=0)
        dynamic_gt_curve = dynamic_gt.mean(axis=0)
        x_label = 'Width (Time)'
    else:
        # H维度：对W取平均，得到(H,)
        static_pred_curve = static_pred.mean(axis=1)
        dynamic_pred_curve = dynamic_pred.mean(axis=1)
        static_gt_curve = static_gt.mean(axis=1)
        dynamic_gt_curve = dynamic_gt.mean(axis=1)
        x_label = 'Height (Time)'
    
    # 计算变化率（一阶差分）
    static_pred_diff = np.abs(np.diff(static_pred_curve))
    dynamic_pred_diff = np.abs(np.diff(dynamic_pred_curve))
    static_gt_diff = np.abs(np.diff(static_gt_curve))
    dynamic_gt_diff = np.abs(np.diff(dynamic_gt_curve))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    
    # 静态分量的时间曲线
    axes[0, 0].plot(static_pred_curve, label='Predicted', linewidth=2)
    axes[0, 0].plot(static_gt_curve, label='Ground Truth', linewidth=2, alpha=0.7)
    axes[0, 0].set_title('Static Component - Temporal Profile')
    axes[0, 0].set_xlabel(x_label)
    axes[0, 0].set_ylabel('Magnitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 静态分量的变化率
    axes[0, 1].plot(static_pred_diff, label='Predicted', linewidth=2)
    axes[0, 1].plot(static_gt_diff, label='Ground Truth', linewidth=2, alpha=0.7)
    axes[0, 1].set_title('Static Component - Variation Rate (should be LOW)')
    axes[0, 1].set_xlabel(x_label)
    axes[0, 1].set_ylabel('|Δ Magnitude|')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 动态分量的时间曲线
    axes[1, 0].plot(dynamic_pred_curve, label='Predicted', linewidth=2)
    axes[1, 0].plot(dynamic_gt_curve, label='Ground Truth', linewidth=2, alpha=0.7)
    axes[1, 0].set_title('Dynamic Component - Temporal Profile')
    axes[1, 0].set_xlabel(x_label)
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 动态分量的变化率
    axes[1, 1].plot(dynamic_pred_diff, label='Predicted', linewidth=2)
    axes[1, 1].plot(dynamic_gt_diff, label='Ground Truth', linewidth=2, alpha=0.7)
    axes[1, 1].set_title('Dynamic Component - Variation Rate (should be HIGH)')
    axes[1, 1].set_xlabel(x_label)
    axes[1, 1].set_ylabel('|Δ Magnitude|')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 添加统计信息
    static_var = np.var(static_pred_diff)
    dynamic_var = np.var(dynamic_pred_diff)
    
    fig.suptitle(f'Temporal Analysis | Static Var: {static_var:.6f} | Dynamic Var: {dynamic_var:.6f}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 转换为numpy数组
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    
    plt.close(fig)
    buf.close()
    
    return img_array


if __name__ == '__main__':
    """测试可视化函数"""
    
    # 创建假数据
    B, H, W = 4, 100, 150
    
    inputs = torch.randn(B, 2, H, W)
    
    # 静态分量（应该平滑）
    static = torch.randn(B, 2, H, 1).repeat(1, 1, 1, W)
    static += torch.randn(B, 2, H, W) * 0.1
    
    # 动态分量（应该变化大）
    dynamic = torch.randn(B, 2, H, W)
    
    pred = {
        'static': static,
        'dynamic': dynamic,
        'total': static + dynamic
    }
    
    target = {
        'static': static + torch.randn_like(static) * 0.1,
        'dynamic': dynamic + torch.randn_like(dynamic) * 0.1,
        'target': static + dynamic + torch.randn_like(static) * 0.1
    }
    
    print("Testing visualization functions...")
    
    # 测试对比网格
    print("1. Creating comparison grid (magnitude)...")
    grid_mag = create_comparison_grid(inputs, pred, target, is_baseline=False, num_samples=2, mode='magnitude')
    print(f"   Output shape: {grid_mag.shape}")
    
    print("2. Creating comparison grid (phase)...")
    grid_phase = create_comparison_grid(inputs, pred, target, is_baseline=False, num_samples=2, mode='phase')
    print(f"   Output shape: {grid_phase.shape}")
    
    # 测试误差直方图
    print("3. Creating error histogram...")
    hist = create_error_histogram(pred, target, is_baseline=False)
    print(f"   Output shape: {hist.shape}")
    
    # 测试时间变化图
    print("4. Creating temporal variation plot...")
    temporal = create_temporal_variation_plot(pred, target, is_baseline=False, dim=-1)
    if temporal is not None:
        print(f"   Output shape: {temporal.shape}")
    
    print("\n✅ All visualization tests passed!")
