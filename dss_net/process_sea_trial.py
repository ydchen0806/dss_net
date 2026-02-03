"""
处理真实海试数据并生成去噪前后对比
- 输出去噪前后的mat文件
- 生成可视化对比图
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

# 添加模型路径
sys.path.insert(0, str(Path(__file__).parent))
from model import UNetDecomposer


def load_model(checkpoint_path, device):
    """加载训练好的DSS-Net模型"""
    print(f"📦 Loading model from: {checkpoint_path}")
    
    model = UNetDecomposer(
        in_channels=2,
        base_channels=64,
        depth=4,
        norm_type="batch",
        dropout=0.1,
        use_attention=True
    )
    
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
    tensor = np.stack([real, imag], axis=0)
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
    padding = (0, pad_w, 0, pad_h)
    return F.pad(tensor, padding, mode='constant', value=0), (h, w)


def crop_to_original(tensor, original_h, original_w):
    """裁剪回原始尺寸"""
    return tensor[:, :original_h, :original_w]


@torch.no_grad()
def process_single_file(model, data_path, device, target_shape=(100, 150)):
    """处理单个数据文件"""
    print(f"\n📄 Processing: {os.path.basename(data_path)}")
    
    # 加载数据
    data = sio.loadmat(data_path)
    est_h = data['est_h']  # (100, 120) complex
    original_shape = est_h.shape
    print(f"   Input shape: {original_shape}")
    
    # 转换为张量
    input_tensor = complex_to_tensor(est_h)
    
    # 填充到目标尺寸
    input_padded, (orig_h, orig_w) = pad_to_shape(input_tensor, target_shape[0], target_shape[1])
    
    # 归一化
    input_norm, scale = normalize_power(input_padded)
    
    # 添加batch维度并移到设备
    input_batch = input_norm.unsqueeze(0).to(device)
    
    # 模型推理
    output = model(input_batch)
    
    # 提取结果
    pred_static = output['static'][0].cpu()
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
    
    return {
        'input': est_h,  # 原始输入（去噪前）
        'denoised': pred_total_complex,  # 去噪后
        'static': pred_static_complex,  # 静态分量
        'dynamic': pred_dynamic_complex,  # 动态分量
    }


def save_comparison_mat(results, save_dir, prefix):
    """保存去噪前后的mat文件"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存格式与输入一致
    mat_data = {
        'est_h_original': results['input'],      # 去噪前
        'est_h_denoised': results['denoised'],   # 去噪后
        'est_h_static': results['static'],       # 静态分量
        'est_h_dynamic': results['dynamic'],     # 动态分量
    }
    
    save_path = save_dir / f'{prefix}_processed.mat'
    sio.savemat(str(save_path), mat_data)
    print(f"   💾 Saved: {save_path}")
    return save_path


def create_comparison_plot(results, save_dir, prefix, fig_idx):
    """创建单个可视化对比图"""
    save_dir = Path(save_dir)
    
    input_h = results['input']
    denoised = results['denoised']
    static = results['static']
    dynamic = results['dynamic']
    
    # 设置绘图风格
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['axes.labelsize'] = 10
    
    # 图1：幅度对比 (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    vmax = np.max(np.abs(input_h)) * 0.8
    
    # 去噪前
    ax = axes[0, 0]
    im = ax.imshow(np.abs(input_h), aspect='auto', cmap='jet', vmin=0, vmax=vmax)
    ax.set_title('Before Denoising (Original)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Delay (samples)')
    ax.set_ylabel('Time (OFDM symbols)')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 去噪后
    ax = axes[0, 1]
    im = ax.imshow(np.abs(denoised), aspect='auto', cmap='jet', vmin=0, vmax=vmax)
    ax.set_title('After Denoising (DSS-Net)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Delay (samples)')
    ax.set_ylabel('Time (OFDM symbols)')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 静态分量
    ax = axes[1, 0]
    im = ax.imshow(np.abs(static), aspect='auto', cmap='jet', vmin=0, vmax=vmax)
    ax.set_title('Static Component', fontsize=12, fontweight='bold')
    ax.set_xlabel('Delay (samples)')
    ax.set_ylabel('Time (OFDM symbols)')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 动态分量
    ax = axes[1, 1]
    im = ax.imshow(np.abs(dynamic), aspect='auto', cmap='jet', vmin=0, vmax=vmax)
    ax.set_title('Dynamic Component', fontsize=12, fontweight='bold')
    ax.set_xlabel('Delay (samples)')
    ax.set_ylabel('Time (OFDM symbols)')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle(f'DSS-Net Denoising Comparison: {prefix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f'compare_{fig_idx:02d}_{prefix}_magnitude.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return 1


def create_slice_comparison(results, save_dir, prefix, fig_idx_base):
    """创建时间/延迟切片对比"""
    save_dir = Path(save_dir)
    
    input_h = results['input']
    denoised = results['denoised']
    
    n_time, n_delay = input_h.shape
    
    # 图2：时间切片对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    time_indices = [0, n_time//3, 2*n_time//3, n_time-1]
    
    for idx, t in enumerate(time_indices):
        ax = axes[idx // 2, idx % 2]
        ax.plot(np.abs(input_h[t, :]), 'b-', alpha=0.7, label='Before Denoising', linewidth=1.5)
        ax.plot(np.abs(denoised[t, :]), 'r-', label='After Denoising', linewidth=1.5)
        ax.set_title(f'Time Slice t={t}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Delay (samples)')
        ax.set_ylabel('Magnitude')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.suptitle(f'Channel Impulse Response Comparison: {prefix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f'compare_{fig_idx_base:02d}_{prefix}_time_slices.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 图3：延迟切片对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    delay_indices = [0, n_delay//4, n_delay//2, 3*n_delay//4]
    
    for idx, d in enumerate(delay_indices):
        ax = axes[idx // 2, idx % 2]
        ax.plot(np.abs(input_h[:, d]), 'b-', alpha=0.7, label='Before Denoising', linewidth=1.5)
        ax.plot(np.abs(denoised[:, d]), 'r-', label='After Denoising', linewidth=1.5)
        ax.set_title(f'Delay Slice d={d}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (OFDM symbols)')
        ax.set_ylabel('Magnitude')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.suptitle(f'Temporal Variation Comparison: {prefix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f'compare_{fig_idx_base+1:02d}_{prefix}_delay_slices.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return 2


def create_side_by_side_comparison(results, save_dir, prefix, fig_idx):
    """创建并排对比图"""
    save_dir = Path(save_dir)
    
    input_h = results['input']
    denoised = results['denoised']
    
    # 计算差异
    diff = np.abs(input_h) - np.abs(denoised)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    vmax = np.max(np.abs(input_h)) * 0.8
    
    # 去噪前
    ax = axes[0]
    im = ax.imshow(np.abs(input_h), aspect='auto', cmap='jet', vmin=0, vmax=vmax)
    ax.set_title('Before Denoising', fontsize=12, fontweight='bold')
    ax.set_xlabel('Delay (samples)')
    ax.set_ylabel('Time (OFDM symbols)')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 去噪后
    ax = axes[1]
    im = ax.imshow(np.abs(denoised), aspect='auto', cmap='jet', vmin=0, vmax=vmax)
    ax.set_title('After Denoising', fontsize=12, fontweight='bold')
    ax.set_xlabel('Delay (samples)')
    ax.set_ylabel('Time (OFDM symbols)')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 差异
    ax = axes[2]
    diff_max = np.max(np.abs(diff))
    im = ax.imshow(diff, aspect='auto', cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
    ax.set_title('Difference (Removed Noise)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Delay (samples)')
    ax.set_ylabel('Time (OFDM symbols)')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle(f'Side-by-Side Comparison: {prefix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f'compare_{fig_idx:02d}_{prefix}_side_by_side.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return 1


def create_power_comparison(results, save_dir, prefix, fig_idx):
    """创建功率谱对比"""
    save_dir = Path(save_dir)
    
    input_h = results['input']
    denoised = results['denoised']
    
    # 平均功率延迟谱
    pdp_input = np.mean(np.abs(input_h) ** 2, axis=0)
    pdp_denoised = np.mean(np.abs(denoised) ** 2, axis=0)
    
    # 平均功率时间谱
    ptp_input = np.mean(np.abs(input_h) ** 2, axis=1)
    ptp_denoised = np.mean(np.abs(denoised) ** 2, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 功率延迟谱
    ax = axes[0]
    ax.plot(10*np.log10(pdp_input + 1e-10), 'b-', label='Before Denoising', linewidth=1.5)
    ax.plot(10*np.log10(pdp_denoised + 1e-10), 'r-', label='After Denoising', linewidth=1.5)
    ax.set_title('Power Delay Profile (PDP)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Delay (samples)')
    ax.set_ylabel('Power (dB)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 功率时间谱
    ax = axes[1]
    ax.plot(10*np.log10(ptp_input + 1e-10), 'b-', label='Before Denoising', linewidth=1.5)
    ax.plot(10*np.log10(ptp_denoised + 1e-10), 'r-', label='After Denoising', linewidth=1.5)
    ax.set_title('Power Time Profile (PTP)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (OFDM symbols)')
    ax.set_ylabel('Power (dB)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.suptitle(f'Power Profile Comparison: {prefix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f'compare_{fig_idx:02d}_{prefix}_power_profile.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return 1


def main():
    # 配置
    checkpoint_path = '/LSEM/user/chenyinda/code/signal_dy_static/dss_net/results_20251104_092511/full/Ablation2_FullImproved_UNetDecomposer_20251104_092515/checkpoints/best.pth'
    data_dir = Path('/LSEM/user/chenyinda/code/signal_dy_static/sea_trial_data')
    save_dir = Path('/LSEM/user/chenyinda/code/signal_dy_static/sea_trial_data/compare')
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("🌊 DSS-Net Sea Trial Data Processing")
    print("="*70)
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Data Dir: {data_dir}")
    print(f"   Save Dir: {save_dir}")
    print(f"   Device: {device}")
    print("="*70)
    
    # 创建保存目录
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    model = load_model(checkpoint_path, device)
    
    # 获取数据文件
    data_files = sorted(data_dir.glob('*.mat'))
    data_files = [f for f in data_files if 'processed' not in f.name]  # 排除已处理的文件
    print(f"\n📁 Found {len(data_files)} data files to process")
    
    # 处理每个文件
    all_results = {}
    fig_idx = 1
    
    for data_path in data_files:
        prefix = data_path.stem  # e.g., "484_5m_01_LS"
        print(f"\n{'='*60}")
        
        # 处理数据
        results = process_single_file(model, str(data_path), device)
        all_results[prefix] = results
        
        # 保存mat文件
        save_comparison_mat(results, save_dir, prefix)
        
        # 生成可视化
        # 1. 幅度对比图
        create_comparison_plot(results, save_dir, prefix, fig_idx)
        fig_idx += 1
        
        # 2. 并排对比图
        create_side_by_side_comparison(results, save_dir, prefix, fig_idx)
        fig_idx += 1
        
        # 3. 功率谱对比
        create_power_comparison(results, save_dir, prefix, fig_idx)
        fig_idx += 1
        
        # 4. 切片对比
        create_slice_comparison(results, save_dir, prefix, fig_idx)
        fig_idx += 2
    
    # 汇总统计
    print("\n" + "="*70)
    print("📊 PROCESSING SUMMARY")
    print("="*70)
    
    summary_data = []
    for prefix, results in all_results.items():
        input_h = results['input']
        denoised = results['denoised']
        static = results['static']
        dynamic = results['dynamic']
        
        # 计算统计量
        input_power = np.mean(np.abs(input_h) ** 2)
        denoised_power = np.mean(np.abs(denoised) ** 2)
        static_power = np.mean(np.abs(static) ** 2)
        dynamic_power = np.mean(np.abs(dynamic) ** 2)
        
        # 功率减少（客观指标，不假设去除的是噪声）
        input_power_dB = 10 * np.log10(input_power + 1e-10)
        denoised_power_dB = 10 * np.log10(denoised_power + 1e-10)
        power_reduction = input_power_dB - denoised_power_dB
        
        static_ratio = static_power / (denoised_power + 1e-10) * 100
        dynamic_ratio = dynamic_power / (denoised_power + 1e-10) * 100
        
        print(f"\n📄 {prefix}")
        print(f"   Input Power:     {input_power_dB:.2f} dB")
        print(f"   Output Power:    {denoised_power_dB:.2f} dB")
        print(f"   Power Reduction: {power_reduction:.2f} dB")
        print(f"   Static Ratio:    {static_ratio:.1f}%")
        print(f"   Dynamic Ratio:   {dynamic_ratio:.1f}%")
        
        summary_data.append({
            'file': prefix,
            'input_power_dB': input_power_dB,
            'output_power_dB': denoised_power_dB,
            'power_reduction_dB': power_reduction,
            'static_ratio': static_ratio,
            'dynamic_ratio': dynamic_ratio
        })
    
    # 保存汇总结果
    summary_mat = {
        'files': [s['file'] for s in summary_data],
        'input_power_dB': np.array([s['input_power_dB'] for s in summary_data]),
        'output_power_dB': np.array([s['output_power_dB'] for s in summary_data]),
        'power_reduction_dB': np.array([s['power_reduction_dB'] for s in summary_data]),
        'static_ratio': np.array([s['static_ratio'] for s in summary_data]),
        'dynamic_ratio': np.array([s['dynamic_ratio'] for s in summary_data])
    }
    sio.savemat(str(save_dir / 'summary.mat'), summary_mat)
    
    print("\n" + "="*70)
    print(f"✅ PROCESSING COMPLETE!")
    print(f"   📁 Results saved to: {save_dir}")
    print(f"   📊 Generated {fig_idx-1} comparison figures")
    print("="*70 + "\n")
    
    # 列出生成的文件
    print("📂 Generated files:")
    for f in sorted(save_dir.glob('*')):
        print(f"   {f.name}")


if __name__ == '__main__':
    main()

