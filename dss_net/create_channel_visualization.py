#!/usr/bin/env python
"""
创建高质量的信道可视化对比图
用于论文展示
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# 导入自定义模块
sys.path.insert(0, '/LSEM/user/chenyinda/code/signal_dy_static/1104')
from model import UNetDecomposer, UNetBaseline
from dataset import create_dataloaders

def setup_paper_style():
    """设置论文级别样式"""
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    rcParams['font.size'] = 11
    rcParams['axes.titlesize'] = 12
    rcParams['axes.labelsize'] = 11
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 9
    rcParams['figure.titlesize'] = 14
    rcParams['axes.linewidth'] = 1.5
    rcParams['xtick.major.width'] = 1.2
    rcParams['ytick.major.width'] = 1.2
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'
    rcParams['mathtext.fontset'] = 'stix'


def load_model(checkpoint_path, model_type='decomposer', device='cuda'):
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model_type == 'decomposer':
        model = UNetDecomposer(
            in_channels=2,
            out_channels=2,
            features=[64, 128, 256, 512],
            use_attention=True
        )
    else:
        model = UNetBaseline(
            in_channels=2,
            out_channels=2,
            features=[64, 128, 256, 512]
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def create_single_sample_visualization(sample_data, models_dict, output_path, sample_idx=0):
    """创建单样本的高质量可视化"""
    setup_paper_style()
    
    noisy = sample_data['noisy']
    clean = sample_data['clean']
    static_gt = sample_data.get('static', None)
    dynamic_gt = sample_data.get('dynamic', None)
    
    # 模型数量
    n_models = len(models_dict)
    
    # 创建图形 - 更紧凑的布局
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.25, wspace=0.15)
    
    # 颜色映射
    cmap = 'RdBu_r'
    
    # 计算全局范围
    vmin = min(noisy.min(), clean.min())
    vmax = max(noisy.max(), clean.max())
    
    # 第一行：输入和Ground Truth
    # (a) Noisy Input
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(noisy, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax1.set_title('(a) Noisy Input', fontweight='bold')
    ax1.set_xlabel('Time Slot')
    ax1.set_ylabel('Delay Tap')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)
    
    # (b) Clean Ground Truth
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(clean, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax2.set_title('(b) Clean (Ground Truth)', fontweight='bold')
    ax2.set_xlabel('Time Slot')
    ax2.set_ylabel('Delay Tap')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax)
    
    # (c) Static Ground Truth
    if static_gt is not None:
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(static_gt, cmap=cmap, aspect='auto')
        ax3.set_title('(c) Static Component (GT)', fontweight='bold')
        ax3.set_xlabel('Time Slot')
        ax3.set_ylabel('Delay Tap')
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im3, cax=cax)
    
    # (d) Dynamic Ground Truth
    if dynamic_gt is not None:
        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.imshow(dynamic_gt, cmap=cmap, aspect='auto')
        ax4.set_title('(d) Dynamic Component (GT)', fontweight='bold')
        ax4.set_xlabel('Time Slot')
        ax4.set_ylabel('Delay Tap')
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im4, cax=cax)
    
    # 第二行和第三行：模型输出
    model_names = list(models_dict.keys())
    for idx, (name, outputs) in enumerate(models_dict.items()):
        row = 1 + idx // 4
        col = idx % 4
        
        ax = fig.add_subplot(gs[row, col])
        
        # 计算NMSE
        pred = outputs['total']
        nmse = 10 * np.log10(np.mean((pred - clean)**2) / np.mean(clean**2))
        
        im = ax.imshow(pred, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(f'({chr(101+idx)}) {name}\nNMSE: {nmse:.2f} dB', fontweight='bold', fontsize=10)
        ax.set_xlabel('Time Slot')
        ax.set_ylabel('Delay Tap')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    
    plt.suptitle(f'Channel Denoising Comparison (Sample {sample_idx})', fontsize=14, fontweight='bold', y=1.02)
    
    # 保存
    for fmt in ['pdf', 'png', 'eps']:
        fig.savefig(f'{output_path}_sample{sample_idx}.{fmt}', format=fmt, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}_sample{sample_idx}.[pdf/png/eps]")


def create_error_map_visualization(sample_data, models_dict, output_path, sample_idx=0):
    """创建误差图可视化"""
    setup_paper_style()
    
    clean = sample_data['clean']
    
    n_models = len(models_dict)
    n_cols = min(4, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, outputs) in enumerate(models_dict.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        pred = outputs['total']
        error = np.abs(pred - clean)
        
        im = ax.imshow(error, cmap='hot', aspect='auto')
        nmse = 10 * np.log10(np.mean((pred - clean)**2) / np.mean(clean**2))
        ax.set_title(f'{name}\nNMSE: {nmse:.2f} dB', fontweight='bold', fontsize=10)
        ax.set_xlabel('Time Slot')
        ax.set_ylabel('Delay Tap')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='|Error|')
    
    # 隐藏多余的子图
    for idx in range(n_models, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Absolute Error Maps (Sample {sample_idx})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    for fmt in ['pdf', 'png', 'eps']:
        fig.savefig(f'{output_path}_error_sample{sample_idx}.{fmt}', format=fmt, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}_error_sample{sample_idx}.[pdf/png/eps]")


def create_decomposition_visualization(sample_data, model_outputs, output_path, sample_idx=0):
    """创建分解可视化 - 展示静态和动态分量"""
    setup_paper_style()
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.2)
    
    noisy = sample_data['noisy']
    clean = sample_data['clean']
    static_gt = sample_data.get('static', np.zeros_like(clean))
    dynamic_gt = sample_data.get('dynamic', np.zeros_like(clean))
    
    pred_static = model_outputs.get('static', np.zeros_like(clean))
    pred_dynamic = model_outputs.get('dynamic', np.zeros_like(clean))
    pred_total = model_outputs['total']
    
    cmap = 'RdBu_r'
    
    # 第一行：输入和总体输出
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(noisy, cmap=cmap, aspect='auto')
    ax1.set_title('(a) Noisy Input', fontweight='bold')
    ax1.set_xlabel('Time Slot')
    ax1.set_ylabel('Delay Tap')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(clean, cmap=cmap, aspect='auto')
    ax2.set_title('(b) Clean (GT)', fontweight='bold')
    ax2.set_xlabel('Time Slot')
    ax2.set_ylabel('Delay Tap')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(pred_total, cmap=cmap, aspect='auto')
    nmse_total = 10 * np.log10(np.mean((pred_total - clean)**2) / np.mean(clean**2))
    ax3.set_title(f'(c) DSS-Net Output\nNMSE: {nmse_total:.2f} dB', fontweight='bold')
    ax3.set_xlabel('Time Slot')
    ax3.set_ylabel('Delay Tap')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    ax4 = fig.add_subplot(gs[0, 3])
    error = np.abs(pred_total - clean)
    im4 = ax4.imshow(error, cmap='hot', aspect='auto')
    ax4.set_title('(d) Absolute Error', fontweight='bold')
    ax4.set_xlabel('Time Slot')
    ax4.set_ylabel('Delay Tap')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # 第二行：静态分量
    ax5 = fig.add_subplot(gs[1, 0:2])
    im5 = ax5.imshow(static_gt, cmap=cmap, aspect='auto')
    ax5.set_title('(e) Static Component (Ground Truth)', fontweight='bold')
    ax5.set_xlabel('Time Slot')
    ax5.set_ylabel('Delay Tap')
    plt.colorbar(im5, ax=ax5, fraction=0.023, pad=0.04)
    
    ax6 = fig.add_subplot(gs[1, 2:4])
    im6 = ax6.imshow(pred_static, cmap=cmap, aspect='auto')
    nmse_static = 10 * np.log10(np.mean((pred_static - static_gt)**2) / (np.mean(static_gt**2) + 1e-10))
    ax6.set_title(f'(f) Static Component (Predicted)\nNMSE: {nmse_static:.2f} dB', fontweight='bold')
    ax6.set_xlabel('Time Slot')
    ax6.set_ylabel('Delay Tap')
    plt.colorbar(im6, ax=ax6, fraction=0.023, pad=0.04)
    
    # 第三行：动态分量
    ax7 = fig.add_subplot(gs[2, 0:2])
    im7 = ax7.imshow(dynamic_gt, cmap=cmap, aspect='auto')
    ax7.set_title('(g) Dynamic Component (Ground Truth)', fontweight='bold')
    ax7.set_xlabel('Time Slot')
    ax7.set_ylabel('Delay Tap')
    plt.colorbar(im7, ax=ax7, fraction=0.023, pad=0.04)
    
    ax8 = fig.add_subplot(gs[2, 2:4])
    im8 = ax8.imshow(pred_dynamic, cmap=cmap, aspect='auto')
    nmse_dynamic = 10 * np.log10(np.mean((pred_dynamic - dynamic_gt)**2) / (np.mean(dynamic_gt**2) + 1e-10))
    ax8.set_title(f'(h) Dynamic Component (Predicted)\nNMSE: {nmse_dynamic:.2f} dB', fontweight='bold')
    ax8.set_xlabel('Time Slot')
    ax8.set_ylabel('Delay Tap')
    plt.colorbar(im8, ax=ax8, fraction=0.023, pad=0.04)
    
    plt.suptitle(f'DSS-Net Dynamic-Static Decomposition (Sample {sample_idx})', fontsize=14, fontweight='bold', y=1.02)
    
    for fmt in ['pdf', 'png', 'eps']:
        fig.savefig(f'{output_path}_decomposition_sample{sample_idx}.{fmt}', format=fmt, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}_decomposition_sample{sample_idx}.[pdf/png/eps]")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_samples', type=int, default=3)
    args = parser.parse_args()
    
    base_dir = args.base_dir
    output_dir = args.output_dir or os.path.join(base_dir, 'paper_visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    print("Loading test data...")
    config = {
        'data': {
            'data_dir': '/LSEM/user/chenyinda/code/signal_xiaoyu',
            'pattern': '2D_channel_for_DSS/Tx_all_2D_*.mat',
            'batch_size': 1,
            'num_workers': 0,
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'normalize_method': 'power',
            'complex_representation': 'real_imag',
        }
    }
    _, _, test_loader = create_dataloaders(config)
    
    # 加载模型
    print("Loading models...")
    models = {}
    
    # 加载最佳配置模型 (no_reg)
    best_ckpt = os.path.join(base_dir, 'no_reg', 'Ablation7_NoRegularization_UNetDecomposer_20251104_092515', 'checkpoints', 'best_model.pt')
    if os.path.exists(best_ckpt):
        models['DSS-Net (Best)'] = load_model(best_ckpt, 'decomposer', device)
    
    # 加载full配置
    full_ckpt = os.path.join(base_dir, 'full', 'Ablation2_FullImproved_UNetDecomposer_20251104_092515', 'checkpoints', 'best_model.pt')
    if os.path.exists(full_ckpt):
        models['DSS-Net (Full)'] = load_model(full_ckpt, 'decomposer', device)
    
    # 加载no_attention配置
    no_att_ckpt = os.path.join(base_dir, 'no_attention', 'Ablation4_NoAttention_UNetDecomposer_20251104_092514', 'checkpoints', 'best_model.pt')
    if os.path.exists(no_att_ckpt):
        models['w/o Attention'] = load_model(no_att_ckpt, 'decomposer', device)
    
    print(f"Loaded {len(models)} models")
    
    # 生成可视化
    print("Generating visualizations...")
    
    for sample_idx, batch in enumerate(test_loader):
        if sample_idx >= args.num_samples:
            break
        
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        static_gt = batch.get('static', torch.zeros_like(clean)).to(device)
        dynamic_gt = batch.get('dynamic', torch.zeros_like(clean)).to(device)
        
        sample_data = {
            'noisy': noisy[0, 0].cpu().numpy(),
            'clean': clean[0, 0].cpu().numpy(),
            'static': static_gt[0, 0].cpu().numpy() if static_gt is not None else None,
            'dynamic': dynamic_gt[0, 0].cpu().numpy() if dynamic_gt is not None else None,
        }
        
        models_outputs = {}
        best_outputs = None
        
        with torch.no_grad():
            for name, model in models.items():
                outputs = model(noisy)
                if isinstance(outputs, tuple):
                    static_pred, dynamic_pred = outputs
                    total_pred = static_pred + dynamic_pred
                    models_outputs[name] = {
                        'static': static_pred[0, 0].cpu().numpy(),
                        'dynamic': dynamic_pred[0, 0].cpu().numpy(),
                        'total': total_pred[0, 0].cpu().numpy(),
                    }
                else:
                    models_outputs[name] = {
                        'total': outputs[0, 0].cpu().numpy(),
                    }
                
                if 'Best' in name:
                    best_outputs = models_outputs[name]
        
        # 生成可视化
        output_prefix = os.path.join(output_dir, 'fig_channel')
        create_single_sample_visualization(sample_data, models_outputs, output_prefix, sample_idx)
        create_error_map_visualization(sample_data, models_outputs, output_prefix, sample_idx)
        
        if best_outputs and 'static' in best_outputs:
            create_decomposition_visualization(sample_data, best_outputs, output_prefix, sample_idx)
    
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()

