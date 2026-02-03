#!/usr/bin/env python
"""
论文级别的完整测评脚本
1. 在测试集上评估所有模型
2. 生成详细的指标报告
3. 生成高质量的可视化图片
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from typing import Dict, List, Tuple, Optional
import warnings
import gc

warnings.filterwarnings('ignore')

# 导入自定义模块
from model import UNetDecomposer, UNetBaseline
from dataset import create_dataloaders


# ============================================================================
# 样式设置
# ============================================================================
def setup_paper_style():
    """设置论文级别的matplotlib样式"""
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    rcParams['font.size'] = 12
    rcParams['axes.labelsize'] = 14
    rcParams['axes.titlesize'] = 16
    rcParams['xtick.labelsize'] = 12
    rcParams['ytick.labelsize'] = 12
    rcParams['legend.fontsize'] = 11
    rcParams['figure.titlesize'] = 18
    rcParams['axes.linewidth'] = 1.5
    rcParams['xtick.major.width'] = 1.2
    rcParams['ytick.major.width'] = 1.2
    rcParams['axes.grid'] = True
    rcParams['grid.alpha'] = 0.3
    rcParams['grid.linewidth'] = 0.8
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'
    rcParams['mathtext.fontset'] = 'stix'


# ============================================================================
# 指标计算
# ============================================================================
class MetricsCalculator:
    """评估指标计算器"""
    
    @staticmethod
    def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
        return torch.mean((pred - target) ** 2).item()
    
    @staticmethod
    def nmse(pred: torch.Tensor, target: torch.Tensor) -> float:
        """归一化均方误差 (dB)"""
        mse = torch.mean((pred - target) ** 2)
        power = torch.mean(target ** 2)
        nmse_linear = mse / (power + 1e-10)
        return 10 * torch.log10(nmse_linear + 1e-10).item()
    
    @staticmethod
    def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        return (20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)).item()
    
    @staticmethod
    def cosine_similarity(pred: torch.Tensor, target: torch.Tensor) -> float:
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        dot_product = torch.dot(pred_flat, target_flat)
        pred_norm = torch.norm(pred_flat)
        target_norm = torch.norm(target_flat)
        return (dot_product / (pred_norm * target_norm + 1e-10)).item()


# ============================================================================
# 模型加载
# ============================================================================
def load_model(checkpoint_path: Path, device: torch.device):
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model_name = config['model']['name']
    is_ablation = config['model'].get('ablation', {}).get('enabled', False)
    is_baseline = (model_name == 'UNetBaseline' or is_ablation)
    
    if is_baseline:
        model = UNetBaseline(
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['in_channels'],
            base_channels=config['model']['base_channels'],
            depth=config['model']['depth'],
            norm_type=config['model']['norm_type'],
            dropout=config['model']['dropout']
        )
    else:
        use_attention = config['model'].get('use_attention', False)
        model = UNetDecomposer(
            in_channels=config['model']['in_channels'],
            base_channels=config['model']['base_channels'],
            depth=config['model']['depth'],
            norm_type=config['model']['norm_type'],
            dropout=config['model']['dropout'],
            use_attention=use_attention
        )
    
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model, config, is_baseline, checkpoint


# ============================================================================
# 可视化函数
# ============================================================================
def complex_to_magnitude(tensor: torch.Tensor) -> torch.Tensor:
    """将复数表示的张量转换为幅度"""
    real = tensor[:, 0, :, :]
    imag = tensor[:, 1, :, :]
    magnitude = torch.sqrt(real**2 + imag**2)
    return magnitude


def visualize_sample(input_data, pred, target, is_baseline, save_path, sample_idx=0):
    """
    可视化单个样本的预测结果
    """
    setup_paper_style()
    
    # 转换为numpy
    input_mag = complex_to_magnitude(input_data[sample_idx:sample_idx+1]).squeeze().cpu().numpy()
    
    if is_baseline:
        pred_total = complex_to_magnitude(pred[sample_idx:sample_idx+1]).squeeze().cpu().numpy()
        gt_total = complex_to_magnitude(target['target'][sample_idx:sample_idx+1]).squeeze().cpu().numpy()
        error = np.abs(pred_total - gt_total)
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        im0 = axes[0].imshow(input_mag, cmap='viridis', aspect='auto')
        axes[0].set_title('Input', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        
        im1 = axes[1].imshow(pred_total, cmap='viridis', aspect='auto')
        axes[1].set_title('Prediction', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        im2 = axes[2].imshow(gt_total, cmap='viridis', aspect='auto')
        axes[2].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
        
        im3 = axes[3].imshow(error, cmap='hot', aspect='auto')
        axes[3].set_title('Absolute Error', fontsize=14, fontweight='bold')
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046)
        
    else:
        # 分解模型
        pred_static = complex_to_magnitude(pred['static'][sample_idx:sample_idx+1]).squeeze().cpu().numpy()
        pred_dynamic = complex_to_magnitude(pred['dynamic'][sample_idx:sample_idx+1]).squeeze().cpu().numpy()
        pred_total = pred_static + pred_dynamic
        
        gt_static = complex_to_magnitude(target['static'][sample_idx:sample_idx+1]).squeeze().cpu().numpy()
        gt_dynamic = complex_to_magnitude(target['dynamic'][sample_idx:sample_idx+1]).squeeze().cpu().numpy()
        gt_total = complex_to_magnitude(target['target'][sample_idx:sample_idx+1]).squeeze().cpu().numpy()
        
        error_static = np.abs(pred_static - gt_static)
        error_dynamic = np.abs(pred_dynamic - gt_dynamic)
        error_total = np.abs(pred_total - gt_total)
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        # 第一行：输入和静态分量
        im = axes[0, 0].imshow(input_mag, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Input', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(im, ax=axes[0, 0], fraction=0.046)
        
        im = axes[0, 1].imshow(pred_static, cmap='viridis', aspect='auto')
        axes[0, 1].set_title('Pred Static', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
        
        im = axes[0, 2].imshow(gt_static, cmap='viridis', aspect='auto')
        axes[0, 2].set_title('GT Static', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
        
        im = axes[0, 3].imshow(error_static, cmap='hot', aspect='auto')
        axes[0, 3].set_title('Static Error', fontsize=12, fontweight='bold')
        axes[0, 3].axis('off')
        plt.colorbar(im, ax=axes[0, 3], fraction=0.046)
        
        # 第二行：动态分量
        axes[1, 0].axis('off')
        
        im = axes[1, 1].imshow(pred_dynamic, cmap='viridis', aspect='auto')
        axes[1, 1].set_title('Pred Dynamic', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
        
        im = axes[1, 2].imshow(gt_dynamic, cmap='viridis', aspect='auto')
        axes[1, 2].set_title('GT Dynamic', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
        
        im = axes[1, 3].imshow(error_dynamic, cmap='hot', aspect='auto')
        axes[1, 3].set_title('Dynamic Error', fontsize=12, fontweight='bold')
        axes[1, 3].axis('off')
        plt.colorbar(im, ax=axes[1, 3], fraction=0.046)
        
        # 第三行：总和
        axes[2, 0].axis('off')
        
        im = axes[2, 1].imshow(pred_total, cmap='viridis', aspect='auto')
        axes[2, 1].set_title('Pred Total', fontsize=12, fontweight='bold')
        axes[2, 1].axis('off')
        plt.colorbar(im, ax=axes[2, 1], fraction=0.046)
        
        im = axes[2, 2].imshow(gt_total, cmap='viridis', aspect='auto')
        axes[2, 2].set_title('GT Total', fontsize=12, fontweight='bold')
        axes[2, 2].axis('off')
        plt.colorbar(im, ax=axes[2, 2], fraction=0.046)
        
        im = axes[2, 3].imshow(error_total, cmap='hot', aspect='auto')
        axes[2, 3].set_title('Total Error', fontsize=12, fontweight='bold')
        axes[2, 3].axis('off')
        plt.colorbar(im, ax=axes[2, 3], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_comparison_grid(all_results: List[Dict], sample_idx: int, output_dir: Path):
    """
    创建多模型对比网格图
    """
    setup_paper_style()
    
    n_models = len(all_results)
    fig, axes = plt.subplots(n_models, 5, figsize=(20, 4 * n_models))
    
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(all_results):
        exp_name = result['name']
        pred = result['predictions'][sample_idx]
        target = result['targets'][sample_idx]
        is_baseline = result['is_baseline']
        
        # 获取数据
        if is_baseline:
            pred_total = complex_to_magnitude(pred.unsqueeze(0)).squeeze().cpu().numpy()
        else:
            pred_static = complex_to_magnitude(pred['static'].unsqueeze(0)).squeeze().cpu().numpy()
            pred_dynamic = complex_to_magnitude(pred['dynamic'].unsqueeze(0)).squeeze().cpu().numpy()
            pred_total = pred_static + pred_dynamic
        
        gt_total = complex_to_magnitude(target['target'].unsqueeze(0)).squeeze().cpu().numpy()
        error = np.abs(pred_total - gt_total)
        
        # 绘制
        im = axes[i, 0].imshow(pred_total, cmap='viridis', aspect='auto')
        axes[i, 0].set_title(f'{exp_name}\nPrediction', fontsize=11, fontweight='bold')
        axes[i, 0].axis('off')
        
        im = axes[i, 1].imshow(gt_total, cmap='viridis', aspect='auto')
        axes[i, 1].set_title('Ground Truth', fontsize=11, fontweight='bold')
        axes[i, 1].axis('off')
        
        im = axes[i, 2].imshow(error, cmap='hot', aspect='auto')
        axes[i, 2].set_title('Error', fontsize=11, fontweight='bold')
        axes[i, 2].axis('off')
        
        if not is_baseline:
            im = axes[i, 3].imshow(pred_static, cmap='viridis', aspect='auto')
            axes[i, 3].set_title('Static', fontsize=11, fontweight='bold')
            axes[i, 3].axis('off')
            
            im = axes[i, 4].imshow(pred_dynamic, cmap='viridis', aspect='auto')
            axes[i, 4].set_title('Dynamic', fontsize=11, fontweight='bold')
            axes[i, 4].axis('off')
        else:
            axes[i, 3].axis('off')
            axes[i, 4].axis('off')
    
    plt.suptitle(f'Model Comparison - Sample {sample_idx}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = output_dir / f'comparison_sample_{sample_idx:03d}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


# ============================================================================
# 评估函数
# ============================================================================
@torch.no_grad()
def evaluate_model(model, test_loader, is_baseline, device, num_vis_samples=5):
    """在测试集上评估模型"""
    model.eval()
    
    all_metrics = {
        'static_mse': [],
        'dynamic_mse': [],
        'total_mse': [],
        'static_nmse_db': [],
        'dynamic_nmse_db': [],
        'total_nmse_db': [],
        'static_psnr': [],
        'dynamic_psnr': [],
        'total_psnr': [],
        'static_cosine': [],
        'dynamic_cosine': [],
        'total_cosine': [],
    }
    
    predictions = []
    targets = []
    inputs = []
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        input_data = batch['input'].to(device)
        target_static = batch['static'].to(device)
        target_dynamic = batch['dynamic'].to(device)
        target_total = batch['target'].to(device)
        
        # 前向传播
        pred = model(input_data)
        
        # 解析输出
        if is_baseline:
            pred_total = pred
            pred_static = None
            pred_dynamic = None
        else:
            if isinstance(pred, dict):
                pred_static = pred['static']
                pred_dynamic = pred['dynamic']
                pred_total = pred_static + pred_dynamic
            else:
                pred_static, pred_dynamic = pred
                pred_total = pred_static + pred_dynamic
        
        # 计算指标
        all_metrics['total_mse'].append(MetricsCalculator.mse(pred_total, target_total))
        all_metrics['total_nmse_db'].append(MetricsCalculator.nmse(pred_total, target_total))
        all_metrics['total_psnr'].append(MetricsCalculator.psnr(pred_total, target_total))
        all_metrics['total_cosine'].append(MetricsCalculator.cosine_similarity(pred_total, target_total))
        
        if not is_baseline and pred_static is not None:
            all_metrics['static_mse'].append(MetricsCalculator.mse(pred_static, target_static))
            all_metrics['static_nmse_db'].append(MetricsCalculator.nmse(pred_static, target_static))
            all_metrics['static_psnr'].append(MetricsCalculator.psnr(pred_static, target_static))
            all_metrics['static_cosine'].append(MetricsCalculator.cosine_similarity(pred_static, target_static))
            
            all_metrics['dynamic_mse'].append(MetricsCalculator.mse(pred_dynamic, target_dynamic))
            all_metrics['dynamic_nmse_db'].append(MetricsCalculator.nmse(pred_dynamic, target_dynamic))
            all_metrics['dynamic_psnr'].append(MetricsCalculator.psnr(pred_dynamic, target_dynamic))
            all_metrics['dynamic_cosine'].append(MetricsCalculator.cosine_similarity(pred_dynamic, target_dynamic))
        
        # 保存用于可视化的样本
        if len(predictions) < num_vis_samples:
            for i in range(min(input_data.size(0), num_vis_samples - len(predictions))):
                inputs.append(input_data[i].cpu())
                if is_baseline:
                    predictions.append(pred_total[i].cpu())
                else:
                    predictions.append({
                        'static': pred_static[i].cpu(),
                        'dynamic': pred_dynamic[i].cpu(),
                        'total': pred_total[i].cpu()
                    })
                targets.append({
                    'static': target_static[i].cpu(),
                    'dynamic': target_dynamic[i].cpu(),
                    'target': target_total[i].cpu()
                })
    
    # 计算平均值
    avg_metrics = {}
    for key, values in all_metrics.items():
        if values:
            avg_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    return avg_metrics, predictions, targets, inputs


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='论文级别的完整测评')
    parser.add_argument('--base_dir', type=str,
                       default='/LSEM/user/chenyinda/code/signal_dy_static/1104/results_20251104_092511',
                       help='实验结果目录')
    parser.add_argument('--output_dir', type=str,
                       default='/LSEM/user/chenyinda/code/signal_dy_static/1104/results_20251104_092511/paper_eval',
                       help='评估结果输出目录')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='使用的设备')
    parser.add_argument('--num_vis_samples', type=int, default=10,
                       help='可视化样本数量')
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*80)
    print("📊 论文级别完整测评")
    print("="*80)
    print(f"📁 实验目录: {base_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🖥️  设备: {device}")
    print("="*80 + "\n")
    
    # 查找所有实验
    experiments = []
    for category_dir in base_dir.iterdir():
        if not category_dir.is_dir():
            continue
        if category_dir.name in ['ablation_analysis', 'paper_figures', 'paper_eval']:
            continue
        
        for exp_dir in category_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            checkpoint_path = exp_dir / 'checkpoints' / 'best.pth'
            if checkpoint_path.exists():
                experiments.append({
                    'category': category_dir.name,
                    'name': exp_dir.name,
                    'path': exp_dir,
                    'checkpoint': checkpoint_path
                })
    
    print(f"✅ 找到 {len(experiments)} 个实验\n")
    
    if not experiments:
        print("❌ 未找到任何实验")
        return
    
    # 创建共享的数据加载器
    print("📦 加载测试数据...")
    first_checkpoint = torch.load(experiments[0]['checkpoint'], map_location='cpu')
    first_config = first_checkpoint['config']
    _, _, test_loader = create_dataloaders(first_config, rank=0, world_size=1, use_ddp=False)
    print(f"✅ 测试集大小: {len(test_loader.dataset)}\n")
    
    # 评估所有模型
    all_results = []
    
    for idx, exp in enumerate(experiments):
        print(f"\n{'='*80}")
        print(f"📦 [{idx+1}/{len(experiments)}] 评估: {exp['category']}/{exp['name']}")
        print(f"{'='*80}")
        
        try:
            # 加载模型
            model, config, is_baseline, checkpoint = load_model(exp['checkpoint'], device)
            
            print(f"   模型类型: {'Baseline' if is_baseline else 'Decomposition'}")
            print(f"   最佳Epoch: {checkpoint['epoch']}")
            print(f"   验证Loss: {checkpoint['best_val_loss']:.6f}")
            
            # 评估
            metrics, predictions, targets, inputs = evaluate_model(
                model, test_loader, is_baseline, device, args.num_vis_samples
            )
            
            # 打印结果
            print(f"\n   📊 测试集结果:")
            print(f"      Total NMSE: {metrics['total_nmse_db']['mean']:.2f} ± {metrics['total_nmse_db']['std']:.2f} dB")
            print(f"      Total PSNR: {metrics['total_psnr']['mean']:.2f} ± {metrics['total_psnr']['std']:.2f} dB")
            
            if not is_baseline and 'static_nmse_db' in metrics:
                print(f"      Static NMSE: {metrics['static_nmse_db']['mean']:.2f} ± {metrics['static_nmse_db']['std']:.2f} dB")
                print(f"      Dynamic NMSE: {metrics['dynamic_nmse_db']['mean']:.2f} ± {metrics['dynamic_nmse_db']['std']:.2f} dB")
            
            # 保存结果
            result = {
                'category': exp['category'],
                'name': exp['name'],
                'is_baseline': is_baseline,
                'val_loss': checkpoint['best_val_loss'],
                'best_epoch': checkpoint['epoch'],
                'metrics': metrics,
                'predictions': predictions,
                'targets': targets,
                'inputs': inputs
            }
            all_results.append(result)
            
            # 创建可视化目录
            vis_dir = output_dir / exp['category'] / exp['name']
            vis_dir.mkdir(parents=True, exist_ok=True)
            
            # 可视化样本
            print(f"\n   📊 生成可视化...")
            for i in range(min(5, len(predictions))):
                save_path = vis_dir / f'sample_{i:03d}.png'
                if is_baseline:
                    pred_for_vis = predictions[i].unsqueeze(0)
                else:
                    pred_for_vis = {k: v.unsqueeze(0) for k, v in predictions[i].items()}
                target_for_vis = {k: v.unsqueeze(0) for k, v in targets[i].items()}
                
                visualize_sample(inputs[i].unsqueeze(0), pred_for_vis, target_for_vis, 
                               is_baseline, save_path, sample_idx=0)
            
            print(f"   ✅ 可视化保存到: {vis_dir}")
            
            # 保存指标
            metrics_file = vis_dir / 'test_metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # 清理GPU内存
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ 评估失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成汇总报告
    print(f"\n\n{'='*80}")
    print("📋 生成汇总报告")
    print("="*80 + "\n")
    
    summary_data = []
    for result in all_results:
        entry = {
            'Category': result['category'],
            'Experiment': result['name'],
            'Type': 'Baseline' if result['is_baseline'] else 'Decomposition',
            'Val Loss': result['val_loss'],
            'Best Epoch': result['best_epoch'],
            'Test Total NMSE (dB)': result['metrics']['total_nmse_db']['mean'],
            'Test Total PSNR (dB)': result['metrics']['total_psnr']['mean'],
        }
        
        if not result['is_baseline'] and 'static_nmse_db' in result['metrics']:
            entry['Test Static NMSE (dB)'] = result['metrics']['static_nmse_db']['mean']
            entry['Test Dynamic NMSE (dB)'] = result['metrics']['dynamic_nmse_db']['mean']
        
        summary_data.append(entry)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Test Total NMSE (dB)')
    
    # 保存汇总
    summary_csv = output_dir / 'test_evaluation_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"✅ 汇总CSV: {summary_csv}")
    
    # 打印汇总表格
    print("\n📊 测试集评估结果:")
    print(summary_df.to_string(index=False))
    
    # 生成对比可视化
    print(f"\n📊 生成模型对比图...")
    for sample_idx in range(min(3, args.num_vis_samples)):
        comparison_results = []
        for result in all_results[:6]:  # 最多显示6个模型
            comparison_results.append({
                'name': result['category'],
                'predictions': result['predictions'],
                'targets': result['targets'],
                'is_baseline': result['is_baseline']
            })
        
        if len(comparison_results) > 1:
            save_path = visualize_comparison_grid(comparison_results, sample_idx, output_dir)
            print(f"   ✅ 保存: {save_path}")
    
    print(f"\n\n{'='*80}")
    print("✅ 评估完成！")
    print("="*80)
    print(f"\n📁 结果保存位置: {output_dir}")
    print(f"\n🏆 最佳模型: {summary_df.iloc[0]['Category']}/{summary_df.iloc[0]['Experiment']}")
    print(f"   Test NMSE: {summary_df.iloc[0]['Test Total NMSE (dB)']:.2f} dB")
    print()


if __name__ == '__main__':
    main()

