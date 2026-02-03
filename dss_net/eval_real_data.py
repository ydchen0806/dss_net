"""
评估真实抚仙湖数据
- 加载DSS-Net模型
- 处理真实数据并进行去噪
- 生成可视化结果
"""

import os
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from datetime import datetime
import argparse

# 导入模型
from model import UNetDecomposer


def load_model(checkpoint_path, device):
    """加载训练好的DSS-Net模型"""
    print(f"📦 Loading model from: {checkpoint_path}")
    
    # 模型配置 (与训练时一致)
    model = UNetDecomposer(
        in_channels=2,
        base_channels=64,
        depth=4,
        norm_type="batch",
        dropout=0.1,
        use_attention=True
    )
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"   ✅ Model loaded (epoch {checkpoint['epoch']}, val_loss: {checkpoint['best_val_loss']:.6f})")
    return model


def complex_to_tensor(complex_data):
    """将复数数组转换为 [2, H, W] 张量 (real, imag)"""
    real = np.real(complex_data)
    imag = np.imag(complex_data)
    tensor = np.stack([real, imag], axis=0)  # (2, H, W)
    return torch.from_numpy(tensor).float()


def tensor_to_complex(tensor):
    """将 [2, H, W] 张量转换回复数数组"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    real = tensor[0]
    imag = tensor[1]
    return real + 1j * imag


def normalize_power(tensor):
    """功率归一化"""
    power = torch.sqrt((tensor ** 2).mean())
    normalized = tensor / (power + 1e-8)
    return normalized, power


def denormalize(tensor, scale):
    """反归一化"""
    return tensor * scale


def pad_to_shape(tensor, target_h, target_w):
    """将输入零填充到目标尺寸"""
    _, h, w = tensor.shape
    pad_h = target_h - h
    pad_w = target_w - w
    
    # (left, right, top, bottom)
    padding = (0, pad_w, 0, pad_h)
    return F.pad(tensor, padding, mode='constant', value=0), (h, w)


def crop_to_original(tensor, original_h, original_w):
    """裁剪回原始尺寸"""
    return tensor[:, :original_h, :original_w]


def compute_nmse(pred, target):
    """计算NMSE (dB)"""
    mse = np.mean(np.abs(pred - target) ** 2)
    signal_power = np.mean(np.abs(target) ** 2)
    nmse = 10 * np.log10(mse / (signal_power + 1e-10))
    return nmse


@torch.no_grad()
def process_real_data(model, data_path, device, save_dir, target_shape=(100, 150)):
    """
    处理单个真实数据文件
    
    Args:
        model: DSS-Net模型
        data_path: .mat文件路径
        device: 计算设备
        save_dir: 保存目录
        target_shape: 模型期望的输入尺寸 (H, W)
    
    Returns:
        results: 处理结果字典
    """
    print(f"\n📄 Processing: {os.path.basename(data_path)}")
    
    # 加载数据
    data = sio.loadmat(data_path)
    est_h = data['est_h']  # (100, 120) complex
    original_shape = est_h.shape
    print(f"   Input shape: {original_shape}")
    
    # 转换为张量
    input_tensor = complex_to_tensor(est_h)  # (2, 100, 120)
    
    # 填充到目标尺寸
    input_padded, (orig_h, orig_w) = pad_to_shape(input_tensor, target_shape[0], target_shape[1])
    print(f"   Padded shape: {input_padded.shape}")
    
    # 归一化
    input_norm, scale = normalize_power(input_padded)
    
    # 添加batch维度并移到设备
    input_batch = input_norm.unsqueeze(0).to(device)  # (1, 2, 100, 150)
    
    # 模型推理
    output = model(input_batch)
    
    # 提取结果
    pred_static = output['static'][0].cpu()  # (2, 100, 150)
    pred_dynamic = output['dynamic'][0].cpu()
    pred_total = output['total'][0].cpu()
    
    # 反归一化
    pred_static = denormalize(pred_static, scale)
    pred_dynamic = denormalize(pred_dynamic, scale)
    pred_total = denormalize(pred_total, scale)
    
    # 裁剪回原始尺寸
    pred_static = crop_to_original(pred_static, orig_h, orig_w)
    pred_dynamic = crop_to_original(pred_dynamic, orig_h, orig_w)
    pred_total = crop_to_original(pred_total, orig_h, orig_w)
    
    # 转换回复数
    pred_static_complex = tensor_to_complex(pred_static)
    pred_dynamic_complex = tensor_to_complex(pred_dynamic)
    pred_total_complex = tensor_to_complex(pred_total)
    
    print(f"   Output shape: {pred_total_complex.shape}")
    
    # 保存结果
    results = {
        'input': est_h,
        'pred_static': pred_static_complex,
        'pred_dynamic': pred_dynamic_complex,
        'pred_total': pred_total_complex,
        'original_shape': original_shape
    }
    
    # 生成可视化
    visualize_results(results, save_dir, os.path.basename(data_path).replace('.mat', ''))
    
    return results


def visualize_results(results, save_dir, prefix):
    """可视化去噪结果"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    input_h = results['input']
    pred_static = results['pred_static']
    pred_dynamic = results['pred_dynamic']
    pred_total = results['pred_total']
    
    # 1. 幅度图对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 输入 (含噪声)
    ax = axes[0, 0]
    im = ax.imshow(np.abs(input_h), aspect='auto', cmap='viridis')
    ax.set_title('Input (Noisy Channel)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Delay (samples)')
    ax.set_ylabel('Time (OFDM symbols)')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 去噪后总信道
    ax = axes[0, 1]
    im = ax.imshow(np.abs(pred_total), aspect='auto', cmap='viridis')
    ax.set_title('DSS-Net Output (Denoised)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Delay (samples)')
    ax.set_ylabel('Time (OFDM symbols)')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 静态分量
    ax = axes[1, 0]
    im = ax.imshow(np.abs(pred_static), aspect='auto', cmap='viridis')
    ax.set_title('Static Component', fontsize=12, fontweight='bold')
    ax.set_xlabel('Delay (samples)')
    ax.set_ylabel('Time (OFDM symbols)')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 动态分量
    ax = axes[1, 1]
    im = ax.imshow(np.abs(pred_dynamic), aspect='auto', cmap='viridis')
    ax.set_title('Dynamic Component', fontsize=12, fontweight='bold')
    ax.set_xlabel('Delay (samples)')
    ax.set_ylabel('Time (OFDM symbols)')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle(f'DSS-Net Channel Decomposition: {prefix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f'{prefix}_magnitude.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # 2. 时间切片对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    time_indices = [0, input_h.shape[0]//3, 2*input_h.shape[0]//3, input_h.shape[0]-1]
    
    for idx, t in enumerate(time_indices):
        ax = axes[idx // 2, idx % 2]
        ax.plot(np.abs(input_h[t, :]), 'b-', alpha=0.5, label='Input (Noisy)', linewidth=1)
        ax.plot(np.abs(pred_total[t, :]), 'r-', label='Denoised', linewidth=1.5)
        ax.plot(np.abs(pred_static[t, :]), 'g--', alpha=0.7, label='Static', linewidth=1)
        ax.plot(np.abs(pred_dynamic[t, :]), 'm--', alpha=0.7, label='Dynamic', linewidth=1)
        ax.set_title(f'Time Slice t={t}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Delay (samples)')
        ax.set_ylabel('Magnitude')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.suptitle(f'Channel Impulse Response Comparison: {prefix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f'{prefix}_time_slices.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # 3. 延迟切片对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    delay_indices = [0, input_h.shape[1]//4, input_h.shape[1]//2, 3*input_h.shape[1]//4]
    
    for idx, d in enumerate(delay_indices):
        ax = axes[idx // 2, idx % 2]
        ax.plot(np.abs(input_h[:, d]), 'b-', alpha=0.5, label='Input (Noisy)', linewidth=1)
        ax.plot(np.abs(pred_total[:, d]), 'r-', label='Denoised', linewidth=1.5)
        ax.plot(np.abs(pred_static[:, d]), 'g--', alpha=0.7, label='Static', linewidth=1)
        ax.plot(np.abs(pred_dynamic[:, d]), 'm--', alpha=0.7, label='Dynamic', linewidth=1)
        ax.set_title(f'Delay Slice d={d}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (OFDM symbols)')
        ax.set_ylabel('Magnitude')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.suptitle(f'Temporal Variation Comparison: {prefix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f'{prefix}_delay_slices.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # 4. 功率谱分析
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 输入功率分布
    ax = axes[0]
    power_input = np.abs(input_h) ** 2
    ax.hist(power_input.flatten(), bins=50, alpha=0.7, color='blue', density=True)
    ax.set_title('Input Power Distribution', fontsize=11, fontweight='bold')
    ax.set_xlabel('Power')
    ax.set_ylabel('Density')
    ax.set_yscale('log')
    
    # 静态分量功率
    ax = axes[1]
    power_static = np.abs(pred_static) ** 2
    ax.hist(power_static.flatten(), bins=50, alpha=0.7, color='green', density=True)
    ax.set_title('Static Power Distribution', fontsize=11, fontweight='bold')
    ax.set_xlabel('Power')
    ax.set_ylabel('Density')
    ax.set_yscale('log')
    
    # 动态分量功率
    ax = axes[2]
    power_dynamic = np.abs(pred_dynamic) ** 2
    ax.hist(power_dynamic.flatten(), bins=50, alpha=0.7, color='magenta', density=True)
    ax.set_title('Dynamic Power Distribution', fontsize=11, fontweight='bold')
    ax.set_xlabel('Power')
    ax.set_ylabel('Density')
    ax.set_yscale('log')
    
    plt.suptitle(f'Power Analysis: {prefix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f'{prefix}_power_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Visualizations saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate DSS-Net on Real Fuxian Lake Data')
    parser.add_argument('--checkpoint', type=str, 
                        default='/LSEM/user/chenyinda/code/signal_dy_static/dss_net/results_20251104_092511/full/Ablation2_FullImproved_UNetDecomposer_20251104_092515/checkpoints/best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, 
                        default='/LSEM/user/chenyinda/code/signal_dy_static/sea_trial_data',
                        help='Path to real data directory')
    parser.add_argument('--output_dir', type=str, 
                        default='/LSEM/user/chenyinda/code/signal_dy_static/dss_net/real_data_results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("🌊 DSS-Net Real Data Evaluation - Fuxian Lake Sea Trial")
    print("="*70)
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Data Dir: {args.data_dir}")
    print(f"   Output Dir: {args.output_dir}")
    print(f"   Device: {device}")
    print("="*70)
    
    # 加载模型
    model = load_model(args.checkpoint, device)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有数据文件
    data_files = sorted(Path(args.data_dir).glob('*.mat'))
    print(f"\n📁 Found {len(data_files)} data files")
    
    # 处理每个文件
    all_results = []
    for data_path in data_files:
        try:
            results = process_real_data(
                model=model,
                data_path=str(data_path),
                device=device,
                save_dir=output_dir,
                target_shape=(100, 150)
            )
            all_results.append({
                'file': data_path.name,
                'results': results
            })
        except Exception as e:
            print(f"   ❌ Error processing {data_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 生成汇总报告
    print("\n" + "="*70)
    print("📋 SUMMARY")
    print("="*70)
    
    for r in all_results:
        filename = r['file']
        results = r['results']
        
        # 计算一些统计量
        input_power = np.mean(np.abs(results['input']) ** 2)
        static_power = np.mean(np.abs(results['pred_static']) ** 2)
        dynamic_power = np.mean(np.abs(results['pred_dynamic']) ** 2)
        total_power = np.mean(np.abs(results['pred_total']) ** 2)
        
        static_ratio = static_power / (total_power + 1e-10) * 100
        dynamic_ratio = dynamic_power / (total_power + 1e-10) * 100
        
        print(f"\n📄 {filename}")
        print(f"   Input Power: {input_power:.6f}")
        print(f"   Output Power: {total_power:.6f}")
        print(f"   Static/Total: {static_ratio:.1f}%")
        print(f"   Dynamic/Total: {dynamic_ratio:.1f}%")
    
    # 保存结果到.mat文件
    save_path = output_dir / 'all_results.mat'
    save_data = {}
    for i, r in enumerate(all_results):
        prefix = r['file'].replace('.mat', '')
        save_data[f'{prefix}_input'] = r['results']['input']
        save_data[f'{prefix}_static'] = r['results']['pred_static']
        save_data[f'{prefix}_dynamic'] = r['results']['pred_dynamic']
        save_data[f'{prefix}_total'] = r['results']['pred_total']
    
    sio.savemat(str(save_path), save_data)
    print(f"\n💾 Results saved to: {save_path}")
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

