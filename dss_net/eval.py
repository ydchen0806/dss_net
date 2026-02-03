"""
自动评估所有实验结果
- 自动扫描实验目录
- 加载最佳模型
- 在测试集上评估
- 生成可视化结果保存到 visual/best/
- 生成汇总报告
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import gc
matplotlib.use('Agg')

# 导入模型和数据集
from model import UNetDecomposer, UNetBaseline
from dataset import create_dataloaders
from loss import ChannelDecompositionLoss


def find_all_experiments(base_dir):
    """
    查找所有实验目录
    
    Returns:
        list: 包含有效实验目录的Path对象列表
    """
    base_path = Path(base_dir)
    experiments = []
    
    if not base_path.exists():
        print(f"❌ Directory not found: {base_dir}")
        return experiments
    
    print(f"🔍 Scanning directory: {base_path}")
    
    for exp_dir in base_path.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # 检查是否有checkpoints目录和best.pth
        checkpoint_dir = exp_dir / "checkpoints"
        best_checkpoint = checkpoint_dir / "best.pth"
        
        if best_checkpoint.exists():
            experiments.append(exp_dir)
            print(f"   ✅ Found: {exp_dir.name}")
        else:
            print(f"   ⚠️  Skipped (no best.pth): {exp_dir.name}")
    
    return sorted(experiments)


def load_experiment_info(exp_dir):
    """加载实验信息"""
    config_path = exp_dir / "config.yaml"
    info_path = exp_dir / "experiment_info.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    info = {}
    if info_path.exists():
        with open(info_path, 'r') as f:
            info = yaml.safe_load(f)
    
    return config, info


def load_model(config, checkpoint_path, device):
    """加载模型"""
    model_name = config['model']['name']
    is_ablation = config['model'].get('ablation', {}).get('enabled', False)
    
    # 构建模型
    if is_ablation or model_name == 'UNetBaseline':
        model = UNetBaseline(
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['in_channels'],
            base_channels=config['model']['base_channels'],
            depth=config['model']['depth'],
            norm_type=config['model']['norm_type'],
            dropout=config['model']['dropout']
        )
        is_baseline = True
    else:
        model = UNetDecomposer(
            in_channels=config['model']['in_channels'],
            base_channels=config['model']['base_channels'],
            depth=config['model']['depth'],
            norm_type=config['model']['norm_type'],
            dropout=config['model']['dropout']
        )
        is_baseline = False
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, is_baseline, checkpoint


def visualize_channels(input_data, gt_static, gt_dynamic, gt_total, pred_data, 
                       is_baseline, save_dir, sample_idx=0, max_channels=4):
    """
    可视化信道数据
    
    Args:
        input_data: [B, C, H, W] 输入
        gt_static: [B, C, H, W] GT静态
        gt_dynamic: [B, C, H, W] GT动态
        gt_total: [B, C, H, W] GT总和
        pred_data: dict或tensor 预测结果
        is_baseline: 是否为baseline模型
        save_dir: 保存目录
        sample_idx: 批次中的样本索引
        max_channels: 最多可视化的信道数
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 🔧 检查预测结果是否有效
    if pred_data is None:
        print("      ⚠️  Warning: pred_data is None, skipping visualization")
        return 0
    
    # 转换为numpy
    def to_numpy(x):
        if isinstance(x, dict):
            return {k: v[sample_idx].cpu().numpy() if v is not None else None 
                    for k, v in x.items()}
        return x[sample_idx].cpu().numpy()
    
    input_np = to_numpy(input_data)
    gt_static_np = to_numpy(gt_static)
    gt_dynamic_np = to_numpy(gt_dynamic)
    gt_total_np = to_numpy(gt_total)
    
    if is_baseline:
        pred_total_np = to_numpy(pred_data)
    else:
        pred_dict = to_numpy(pred_data)
        pred_static_np = pred_dict.get('static')
        pred_dynamic_np = pred_dict.get('dynamic')
        pred_total_np = pred_dict.get('total')
        
        # 检查是否有None
        if pred_static_np is None or pred_dynamic_np is None or pred_total_np is None:
            print("      ⚠️  Warning: Some predictions are None, skipping visualization")
            return 0
    
    # 获取数据形状 [C, H, W]
    num_channels = input_np.shape[0]
    num_channels = min(num_channels, max_channels)
    
    # 为每个信道创建可视化
    for ch_idx in range(num_channels):
        if is_baseline:
            # Baseline: 只显示输入、GT总和、预测总和
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # 提取单个信道
            input_ch = input_np[ch_idx]  # [H, W]
            gt_total_ch = gt_total_np[ch_idx]
            pred_total_ch = pred_total_np[ch_idx]
            
            # 显示实部和虚部
            for idx, (data, title) in enumerate([
                (input_ch, 'Input'),
                (gt_total_ch, 'GT Total'),
                (pred_total_ch, 'Pred Total')
            ]):
                ax = axes[idx]
                
                # 如果是复数，分别显示实部和虚部
                if np.iscomplexobj(data):
                    # 显示幅度
                    magnitude = np.abs(data)
                    im = ax.imshow(magnitude, cmap='viridis', aspect='auto')
                    plt.colorbar(im, ax=ax, fraction=0.046)
                    ax.set_title(f'{title} - Magnitude', fontsize=12, fontweight='bold')
                else:
                    im = ax.imshow(data, cmap='viridis', aspect='auto')
                    plt.colorbar(im, ax=ax, fraction=0.046)
                    ax.set_title(title, fontsize=12, fontweight='bold')
                
                ax.axis('off')
            
            plt.suptitle(f'Channel {ch_idx} - Baseline Model', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_dir / f'channel_{ch_idx:02d}_baseline.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        else:
            # Decomposition: 显示静态、动态、总和
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            
            # 提取单个信道
            input_ch = input_np[ch_idx]
            gt_static_ch = gt_static_np[ch_idx]
            gt_dynamic_ch = gt_dynamic_np[ch_idx]
            gt_total_ch = gt_total_np[ch_idx]
            pred_static_ch = pred_static_np[ch_idx]
            pred_dynamic_ch = pred_dynamic_np[ch_idx]
            pred_total_ch = pred_total_np[ch_idx]
            
            # 第一行：输入和GT
            row1_data = [
                (input_ch, 'Input'),
                (gt_static_ch, 'GT Static'),
                (gt_dynamic_ch, 'GT Dynamic'),
                (gt_total_ch, 'GT Total')
            ]
            
            for idx, (data, title) in enumerate(row1_data):
                ax = axes[0, idx]
                
                if np.iscomplexobj(data):
                    magnitude = np.abs(data)
                    im = ax.imshow(magnitude, cmap='viridis', aspect='auto')
                    plt.colorbar(im, ax=ax, fraction=0.046)
                    ax.set_title(f'{title} - Magnitude', fontsize=11, fontweight='bold')
                else:
                    im = ax.imshow(data, cmap='viridis', aspect='auto')
                    plt.colorbar(im, ax=ax, fraction=0.046)
                    ax.set_title(title, fontsize=11, fontweight='bold')
                
                ax.axis('off')
            
            # 第二行：预测结果和误差
            row2_data = [
                (pred_static_ch, 'Pred Static'),
                (pred_dynamic_ch, 'Pred Dynamic'),
                (pred_total_ch, 'Pred Total'),
                (np.abs(gt_total_ch - pred_total_ch) if np.iscomplexobj(gt_total_ch) 
                 else np.abs(gt_total_ch - pred_total_ch), 'Absolute Error')
            ]
            
            for idx, (data, title) in enumerate(row2_data):
                ax = axes[1, idx]
                
                if np.iscomplexobj(data):
                    magnitude = np.abs(data)
                    im = ax.imshow(magnitude, cmap='viridis', aspect='auto')
                    plt.colorbar(im, ax=ax, fraction=0.046)
                    ax.set_title(f'{title} - Magnitude', fontsize=11, fontweight='bold')
                else:
                    # 误差使用不同的colormap
                    cmap = 'hot' if 'Error' in title else 'viridis'
                    im = ax.imshow(data, cmap=cmap, aspect='auto')
                    plt.colorbar(im, ax=ax, fraction=0.046)
                    ax.set_title(title, fontsize=11, fontweight='bold')
                
                ax.axis('off')
            
            plt.suptitle(f'Channel {ch_idx} - Decomposition Model', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_dir / f'channel_{ch_idx:02d}_decomposition.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    return num_channels


def visualize_metrics_comparison(results_list, save_path):
    """绘制所有实验的指标对比图"""
    """绘制所有实验的指标对比图"""
    df = pd.DataFrame(results_list)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. Total NMSE对比
    ax = axes[0, 0]
    df_sorted = df.sort_values('test_total_nmse_db')
    colors = ['#2ecc71' if t == 'Decomposition' else '#e74c3c' 
              for t in df_sorted['model_type']]
    bars = ax.barh(range(len(df_sorted)), df_sorted['test_total_nmse_db'], color=colors)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['experiment_name'], fontsize=9)
    ax.set_xlabel('Total NMSE (dB)', fontsize=11, fontweight='bold')
    ax.set_title('Total NMSE Comparison (Lower is Better)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_xaxis()
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, df_sorted['test_total_nmse_db'])):
        ax.text(val - 0.5, i, f'{val:.2f}', va='center', fontsize=8)
    
    # 2. Static vs Dynamic NMSE (仅分解模型)
    ax = axes[0, 1]
    decomp_df = df[df['model_type'] == 'Decomposition']
    if not decomp_df.empty:
        x = np.arange(len(decomp_df))
        width = 0.35
        bars1 = ax.bar(x - width/2, decomp_df['test_static_nmse_db'], width, 
                       label='Static', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, decomp_df['test_dynamic_nmse_db'], width, 
                       label='Dynamic', color='#e67e22', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(decomp_df['experiment_name'], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('NMSE (dB)', fontsize=11, fontweight='bold')
        ax.set_title('Static vs Dynamic NMSE (Decomposition Models)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=7)
    else:
        ax.text(0.5, 0.5, 'No Decomposition Models', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    # 3. Validation Loss vs Test Loss
    ax = axes[1, 0]
    for model_type in df['model_type'].unique():
        mask = df['model_type'] == model_type
        color = '#2ecc71' if model_type == 'Decomposition' else '#e74c3c'
        ax.scatter(df[mask]['best_val_loss'], df[mask]['test_total_loss'], 
                  s=150, alpha=0.6, label=model_type, color=color)
    
    for i, row in df.iterrows():
        ax.annotate(row['experiment_name'], 
                   (row['best_val_loss'], row['test_total_loss']),
                   fontsize=7, alpha=0.7, xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Best Validation Loss', fontsize=11, fontweight='bold')
    ax.set_ylabel('Test Loss', fontsize=11, fontweight='bold')
    ax.set_title('Validation vs Test Loss', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    # 4. Training Epochs
    ax = axes[1, 1]
    colors = ['#2ecc71' if t == 'Decomposition' else '#e74c3c' 
              for t in df['model_type']]
    bars = ax.bar(range(len(df)), df['best_epoch'], color=colors, alpha=0.7)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['experiment_name'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Best Epoch', fontsize=11, fontweight='bold')
    ax.set_title('Training Epochs to Best Model', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def evaluate_model(model, test_loader, criterion, is_baseline, device, 
                   save_dir=None, vis_sample_indices=None):
    """
    在测试集上评估模型并生成可视化
    
    Args:
        vis_sample_indices: 要可视化的样本索引列表，格式为 [(batch_idx, sample_in_batch), ...]
    """
    model.eval()
    
    total_loss = 0
    metrics = {
        'static_mse': 0,
        'dynamic_mse': 0,
        'total_mse': 0,
        'static_temporal': 0,
        'dynamic_temporal': 0,
        'static_nmse_db': 0,
        'dynamic_nmse_db': 0,
        'total_nmse_db': 0
    }
    
    num_batches = 0
    vis_count = 0
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating", leave=False)):
        inputs = batch['input'].to(device, non_blocking=True)
        target_static = batch['static'].to(device, non_blocking=True)
        target_dynamic = batch['dynamic'].to(device, non_blocking=True)
        target_total = batch['target'].to(device, non_blocking=True)
        
        # 前向传播
        pred = model(inputs)
        
        # 🔧 调试信息（仅第一个batch）
        if batch_idx == 0 and is_baseline:
            print(f"      🔍 Debug: Baseline prediction type: {type(pred)}")
            if pred is not None:
                if isinstance(pred, dict):
                    print(f"      🔍 Debug: Prediction keys: {pred.keys()}")
                else:
                    print(f"      🔍 Debug: Prediction shape: {pred.shape}")
        
        # 计算损失
        losses = criterion(
            pred,
            {
                'static': target_static,
                'dynamic': target_dynamic,
                'target': target_total
            },
            is_baseline=is_baseline
        )
        
        # 🔧 调试信息（仅第一个batch）
        if batch_idx == 0:
            print(f"      🔍 Debug: Loss keys: {losses.keys()}")
        
        # 累积指标
        total_loss += losses['total_loss'].item()
        for key in metrics:
            if key in losses:
                metrics[key] += losses[key].item()
        
        num_batches += 1
        
        # 🆕 只可视化指定的样本
        if save_dir and vis_sample_indices is not None:
            current_batch_size = inputs.shape[0]  # 获取当前batch的实际大小
            for vis_idx, (target_batch_idx, sample_in_batch) in enumerate(vis_sample_indices):
                if batch_idx == target_batch_idx and vis_count < len(vis_sample_indices):
                    # 检查样本索引是否在当前batch范围内
                    if sample_in_batch >= current_batch_size:
                        print(f"      ⚠️  Skipping sample {vis_idx}: index {sample_in_batch} exceeds batch size {current_batch_size}")
                        continue
                    try:
                        vis_dir = save_dir / f"sample_{vis_idx:02d}"
                        num_ch = visualize_channels(
                            inputs, target_static, target_dynamic, target_total,
                            pred, is_baseline, vis_dir, sample_idx=sample_in_batch
                        )
                        if vis_count == 0 and num_ch > 0:
                            print(f"      📊 Visualizing {len(vis_sample_indices)} random samples (fixed seed for consistency)")
                        vis_count += 1
                    except Exception as e:
                        print(f"      ⚠️  Visualization error for sample {vis_idx}: {e}")
    
    # 计算平均值
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in metrics.items()}
    avg_metrics['total_loss'] = avg_loss
    
    return avg_metrics


def format_results_table(results_list):
    """格式化结果为表格"""
    df = pd.DataFrame(results_list)
    
    # 选择关键列用于终端显示
    columns = [
        'experiment_name',
        'model_type',
        'learning_rate',
        'batch_size',
        'best_epoch',
        'best_val_loss',
        'test_total_nmse_db',
    ]
    
    # 如果有decomposition模型，添加静态/动态指标
    if 'test_static_nmse_db' in df.columns:
        columns.extend([
            'test_static_nmse_db',
            'test_dynamic_nmse_db',
        ])
    
    # 过滤存在的列
    columns = [c for c in columns if c in df.columns]
    df_display = df[columns]
    
    return df_display


def main():
    parser = argparse.ArgumentParser(description='Evaluate all experiments')
    parser.add_argument('--exp_dir', type=str, 
                        default='/LSEM/user/chenyinda/code/signal_dy_static/experiments1017',
                        help='Base experiments directory')
    parser.add_argument('--output', type=str, default='evaluation_results.csv',
                        help='Output CSV file for results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for evaluation')
    parser.add_argument('--num_vis_samples', type=int, default=10,
                        help='Number of random samples to visualize per experiment (default: 10)')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*100}")
    print(f"🔍 Automatic Experiment Evaluation System")
    print(f"{'='*100}")
    print(f"   Base Directory: {args.exp_dir}")
    print(f"   Device: {device}")
    print(f"   Random Samples: {args.num_vis_samples} (fixed seed=42 for consistency)")
    print(f"{'='*100}\n")
    
    # 查找所有实验
    experiments = find_all_experiments(args.exp_dir)
    
    if not experiments:
        print("\n❌ No valid experiments found!")
        return
    
    print(f"\n✅ Found {len(experiments)} valid experiments\n")
    
    # 🆕 预加载数据（如果所有实验使用相同的数据配置）
    print(f"{'='*100}")
    print(f"📦 Data Loading Strategy")
    print(f"{'='*100}")
    
    shared_test_loader = None
    vis_sample_indices = None  # 🆕 初始化
    data_configs = []
    
    # 检查所有实验的数据配置是否相同
    for exp_dir in experiments:
        try:
            config, _ = load_experiment_info(exp_dir)
            data_configs.append(config.get('data', {}))
        except:
            data_configs.append(None)
    
    # 判断是否可以共享数据
    can_share_data = all(dc == data_configs[0] for dc in data_configs if dc is not None)
    
    if can_share_data and data_configs[0] is not None:
        print("✅ All experiments use the same data configuration")
        print("💡 Loading test data ONCE (will be shared across all experiments)")
        print("   This saves ~90% of data loading time!\n")
        
        try:
            # 使用第一个实验的配置加载数据
            first_config, _ = load_experiment_info(experiments[0])
            _, _, shared_test_loader = create_dataloaders(
                first_config,
                rank=0,
                world_size=1,
                use_ddp=False
            )
            print(f"\n✅ Test data loaded successfully!")
            print(f"   Batches: {len(shared_test_loader)}")
            
            # 🆕 生成随机可视化样本索引（固定种子）
            np.random.seed(42)  # 固定种子确保所有实验可视化相同样本
            batch_size = first_config.get('data', {}).get('batch_size', 
                                                          first_config.get('training', {}).get('batch_size', 16))
            total_samples = len(shared_test_loader.dataset) if hasattr(shared_test_loader, 'dataset') else len(shared_test_loader) * batch_size
            num_vis_samples = min(args.num_vis_samples, total_samples)
            
            # 随机选择样本（全局索引）
            random_sample_indices = np.random.choice(total_samples, size=num_vis_samples, replace=False)
            random_sample_indices = sorted(random_sample_indices)  # 排序以便按顺序访问
            
            # 转换为 (batch_idx, sample_in_batch) 格式
            vis_sample_indices = []
            for global_idx in random_sample_indices:
                batch_idx = global_idx // batch_size
                sample_in_batch = global_idx % batch_size
                vis_sample_indices.append((batch_idx, sample_in_batch))
            
            print(f"   🎲 Selected {num_vis_samples} random samples for visualization (seed=42)")
            print(f"   📍 Sample indices: {random_sample_indices[:5]}{'...' if num_vis_samples > 5 else ''}")
            print(f"   This dataset will be reused for all {len(experiments)} experiments.\n")
            
        except Exception as e:
            print(f"\n⚠️  Failed to preload data: {e}")
            print(f"   Will load data separately for each experiment.\n")
            shared_test_loader = None
            vis_sample_indices = None  # 🆕 确保初始化
    else:
        print("⚠️  Experiments have different data configurations")
        print("   Will load data separately for each experiment.\n")
        vis_sample_indices = None  # 🆕 确保初始化
    
    print(f"{'='*100}\n")
    
    # 评估每个实验
    results_list = []
    
    for idx, exp_dir in enumerate(experiments, 1):
        print(f"\n{'='*100}")
        print(f"📦 [{idx}/{len(experiments)}] Processing: {exp_dir.name}")
        print(f"{'='*100}")
        
        try:
            # 加载配置和信息
            config, info = load_experiment_info(exp_dir)
            
            print(f"   📋 Experiment: {info.get('experiment_name', 'N/A')}")
            print(f"   🏗️  Model: {info.get('model_name', 'N/A')}")
            print(f"   🔬 Is Ablation: {info.get('is_ablation', 'N/A')}")
            print(f"   ⏱️  Temporal Enabled: {info.get('temporal_enabled', 'N/A')}")
            print(f"   📚 Learning Rate: {config.get('training', {}).get('learning_rate', 'N/A')}")
            print(f"   📦 Batch Size: {config.get('data', {}).get('batch_size', config.get('training', {}).get('batch_size', 'N/A'))}")
            print(f"   🔧 Optimizer: {config.get('training', {}).get('optimizer', 'N/A')}")
            
            # 加载最佳模型
            checkpoint_path = exp_dir / "checkpoints" / "best.pth"
            print(f"   💾 Checkpoint: {checkpoint_path}")
            model, is_baseline, checkpoint = load_model(config, checkpoint_path, device)
            
            best_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            
            print(f"   🏆 Best Epoch: {best_epoch}")
            print(f"   📊 Best Val Loss: {best_val_loss:.6f}")
            
            # 创建数据加载器（只在没有共享数据时才加载）
            if shared_test_loader is None:
                print("\n   📦 Loading test data for this experiment...")
                _, _, test_loader = create_dataloaders(
                    config, 
                    rank=0, 
                    world_size=1, 
                    use_ddp=False
                )
                print(f"   ✅ Loaded {len(test_loader)} batches")
                
                # 生成该实验的随机样本索引
                np.random.seed(42)
                batch_size = config.get('data', {}).get('batch_size', 
                                                        config.get('training', {}).get('batch_size', 16))
                total_samples = len(test_loader.dataset) if hasattr(test_loader, 'dataset') else len(test_loader) * batch_size
                num_vis = min(args.num_vis_samples, total_samples)
                random_indices = np.random.choice(total_samples, size=num_vis, replace=False)
                random_indices = sorted(random_indices)
                local_vis_indices = [(idx // batch_size, idx % batch_size) for idx in random_indices]
                print(f"   🎲 Selected {num_vis} random samples (seed=42)")
            else:
                print("\n   ♻️  Using pre-loaded shared test data")
                print(f"   ⚡ Skipping data loading (saves ~2-3 minutes per experiment!)")
                test_loader = shared_test_loader
                local_vis_indices = vis_sample_indices
            
            # 创建损失函数
            criterion = ChannelDecompositionLoss(config)
            
            # 创建可视化目录：实验目录/visual/best/
            vis_dir = exp_dir / "visual" / "best"
            vis_dir.mkdir(parents=True, exist_ok=True)
            
            # 评估
            print(f"   🔄 Evaluating on test set...")
            test_metrics = evaluate_model(
                model, test_loader, criterion, is_baseline, device,
                save_dir=vis_dir, vis_sample_indices=local_vis_indices
            )
            
            # 打印测试结果
            print(f"\n   {'─'*80}")
            print(f"   📊 Test Results:")
            print(f"   {'─'*80}")
            
            # 🔧 调试：打印所有可用的键
            # print(f"   🔍 Debug: Available metrics: {test_metrics.keys()}")
            
            print(f"      Total Loss: {test_metrics['total_loss']:.6f}")
            print(f"      Total NMSE: {test_metrics['total_nmse_db']:.2f} dB")
            
            if not is_baseline:
                print(f"      Static NMSE: {test_metrics['static_nmse_db']:.2f} dB")
                print(f"      Dynamic NMSE: {test_metrics['dynamic_nmse_db']:.2f} dB")
                
                temporal_enabled = config['loss'].get('temporal_correlation', {}).get('enabled', True)
                if temporal_enabled:
                    print(f"      Static Temporal: {test_metrics['static_temporal']:.6f}")
                    print(f"      Dynamic Temporal: {test_metrics['dynamic_temporal']:.6f}")
            
            # 收集结果
            # 安全地获取配置参数
            training_config = config.get('training', {})
            data_config = config.get('data', {})
            model_config = config.get('model', {})
            loss_config = config.get('loss', {})
            loss_weights = loss_config.get('weights', {})
            scheduler_config = training_config.get('scheduler', {})
            
            # 兼容batch_size在不同位置的情况
            batch_size = data_config.get('batch_size', training_config.get('batch_size', 'N/A'))
            
            result = {
                'experiment_name': info.get('experiment_name', exp_dir.name),
                'model_name': info.get('model_name', 'N/A'),
                'model_type': 'Baseline' if is_baseline else 'Decomposition',
                'is_ablation': info.get('is_ablation', False),
                'temporal_enabled': info.get('temporal_enabled', True),
                'timestamp': info.get('timestamp', 'N/A'),
                
                # 超参数设置
                'learning_rate': training_config.get('learning_rate', 'N/A'),
                'optimizer': training_config.get('optimizer', 'N/A'),
                'batch_size': batch_size,
                'base_channels': model_config.get('base_channels', 'N/A'),
                'depth': model_config.get('depth', 'N/A'),
                'dropout': model_config.get('dropout', 'N/A'),
                'norm_type': model_config.get('norm_type', 'N/A'),
                'scheduler_type': scheduler_config.get('type', 'N/A'),
                'weight_decay': training_config.get('weight_decay', 'N/A'),
                'gradient_clip': training_config.get('gradient_clip', 'None'),
                
                # 损失函数权重
                'loss_static_weight': loss_weights.get('static', 'N/A'),
                'loss_dynamic_weight': loss_weights.get('dynamic', 'N/A'),
                'loss_total_weight': loss_weights.get('total', 'N/A'),
                'loss_temporal_weight': loss_weights.get('temporal_correlation', 'N/A'),
                
                # 训练结果
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss,
                
                # 测试结果 (注意键名对应)
                'test_total_loss': test_metrics['total_loss'],
                'test_total_nmse_db': test_metrics['total_nmse_db'],
                'test_total_mse': test_metrics['total_mse'],
                
                # 路径信息
                'checkpoint_path': str(checkpoint_path),
                'output_dir': str(exp_dir),
                'vis_dir': str(vis_dir)
            }
            
            if not is_baseline:
                result.update({
                    'test_static_nmse_db': test_metrics['static_nmse_db'],
                    'test_dynamic_nmse_db': test_metrics['dynamic_nmse_db'],
                    'test_static_mse': test_metrics['static_mse'],
                    'test_dynamic_mse': test_metrics['dynamic_mse'],
                    'test_static_temporal': test_metrics['static_temporal'],
                    'test_dynamic_temporal': test_metrics['dynamic_temporal'],
                })
            
            results_list.append(result)
            print(f"\n   ✅ Evaluation completed")
            print(f"   💾 Visualizations saved to: {vis_dir}")
            
        except Exception as e:
            print(f"\n   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n\n{'='*100}")
    print(f"📋 SUMMARY REPORT")
    print(f"{'='*100}\n")
    
    # 清理共享的数据加载器
    if shared_test_loader is not None:
        del shared_test_loader
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if results_list:
        df = format_results_table(results_list)
        
        # 打印表格
        print(df.to_string(index=False))
        print()
        
        # 保存到CSV（保存在基础目录）
        output_path = Path(args.exp_dir) / args.output
        pd.DataFrame(results_list).to_csv(output_path, index=False)
        print(f"✅ Results CSV saved to: {output_path}")
        
        # 生成对比图（保存在基础目录）
        comparison_plot_path = Path(args.exp_dir) / "all_experiments_comparison.png"
        visualize_metrics_comparison(results_list, comparison_plot_path)
        print(f"✅ Comparison plot saved to: {comparison_plot_path}")
        
        # 找出最佳模型
        print(f"\n{'='*100}")
        print(f"🏆 BEST MODELS")
        print(f"{'='*100}\n")
        
        df_sorted = pd.DataFrame(results_list).sort_values('test_total_nmse_db')
        best = df_sorted.iloc[0]
        
        print(f"🥇 Best Overall Model:")
        print(f"   Name: {best['experiment_name']}")
        print(f"   Type: {best['model_type']}")
        print(f"   Test NMSE: {best['test_total_nmse_db']:.2f} dB")
        print(f"   Val Loss: {best['best_val_loss']:.6f}")
        print(f"   Best Epoch: {best['best_epoch']}")
        print(f"   Learning Rate: {best.get('learning_rate', 'N/A')}")
        print(f"   Batch Size: {best.get('batch_size', 'N/A')}")
        print(f"   Checkpoint: {best.get('checkpoint_path', 'N/A')}")
        print(f"   Visualizations: {best.get('vis_dir', 'N/A')}")
        
        # 如果有decomposition模型，也显示其性能
        decomp_models = df_sorted[df_sorted['model_type'] == 'Decomposition']
        if not decomp_models.empty:
            best_decomp = decomp_models.iloc[0]
            print(f"\n🥈 Best Decomposition Model:")
            print(f"   Name: {best_decomp['experiment_name']}")
            print(f"   Test NMSE: {best_decomp['test_total_nmse_db']:.2f} dB")
            
            static_nmse = best_decomp.get('test_static_nmse_db', None)
            dynamic_nmse = best_decomp.get('test_dynamic_nmse_db', None)
            if static_nmse is not None and isinstance(static_nmse, (int, float)):
                print(f"   Static NMSE: {static_nmse:.2f} dB")
            if dynamic_nmse is not None and isinstance(dynamic_nmse, (int, float)):
                print(f"   Dynamic NMSE: {dynamic_nmse:.2f} dB")
            
            print(f"   Temporal Enabled: {best_decomp.get('temporal_enabled', 'N/A')}")
            print(f"   Checkpoint: {best_decomp.get('checkpoint_path', 'N/A')}")
            print(f"   Visualizations: {best_decomp.get('vis_dir', 'N/A')}")
        
        # 显示baseline模型（如果有）
        baseline_models = df_sorted[df_sorted['model_type'] == 'Baseline']
        if not baseline_models.empty:
            best_baseline = baseline_models.iloc[0]
            print(f"\n🥉 Baseline Model:")
            print(f"   Name: {best_baseline['experiment_name']}")
            print(f"   Test NMSE: {best_baseline['test_total_nmse_db']:.2f} dB")
            print(f"   Checkpoint: {best_baseline.get('checkpoint_path', 'N/A')}")
            print(f"   Visualizations: {best_baseline.get('vis_dir', 'N/A')}")
    
    else:
        print("❌ No experiments were successfully evaluated")
    
    print(f"\n{'='*100}")
    print(f"✅ EVALUATION COMPLETE!")
    print(f"{'='*100}\n")


if __name__ == '__main__':
    main()