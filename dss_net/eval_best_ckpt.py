"""
推理和评估脚本（修复版 - 正确处理不同模型架构）
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# 导入自定义模块
from dataset import create_dataloaders
from model import UNetDecomposer, UNetBaseline
from visualization import create_comparison_grid


class MetricsCalculator:
    """评估指标计算器"""
    
    @staticmethod
    def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
        """均方误差"""
        return torch.mean((pred - target) ** 2).item()
    
    @staticmethod
    def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
        """平均绝对误差"""
        return torch.mean(torch.abs(pred - target)).item()
    
    @staticmethod
    def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
        """均方根误差"""
        return torch.sqrt(torch.mean((pred - target) ** 2)).item()
    
    @staticmethod
    def nmse(pred: torch.Tensor, target: torch.Tensor) -> float:
        """归一化均方误差 (dB)"""
        mse = torch.mean((pred - target) ** 2)
        power = torch.mean(target ** 2)
        nmse_linear = mse / (power + 1e-10)
        return 10 * torch.log10(nmse_linear + 1e-10).item()
    
    @staticmethod
    def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
        """峰值信噪比"""
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        return (20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)).item()
    
    @staticmethod
    def snr(pred: torch.Tensor, target: torch.Tensor) -> float:
        """信噪比 (dB)"""
        signal_power = torch.mean(target ** 2)
        noise_power = torch.mean((pred - target) ** 2)
        return 10 * torch.log10(signal_power / (noise_power + 1e-10)).item()
    
    @staticmethod
    def cosine_similarity(pred: torch.Tensor, target: torch.Tensor) -> float:
        """余弦相似度"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        dot_product = torch.dot(pred_flat, target_flat)
        pred_norm = torch.norm(pred_flat)
        target_norm = torch.norm(target_flat)
        
        return (dot_product / (pred_norm * target_norm + 1e-10)).item()
    
    @staticmethod
    def correlation_coefficient(pred: torch.Tensor, target: torch.Tensor) -> float:
        """相关系数"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        pred_mean = torch.mean(pred_flat)
        target_mean = torch.mean(target_flat)
        
        numerator = torch.sum((pred_flat - pred_mean) * (target_flat - target_mean))
        denominator = torch.sqrt(
            torch.sum((pred_flat - pred_mean) ** 2) * 
            torch.sum((target_flat - target_mean) ** 2)
        )
        
        return (numerator / (denominator + 1e-10)).item()
    
    @staticmethod
    def relative_error(pred: torch.Tensor, target: torch.Tensor) -> float:
        """相对误差 (%)"""
        error = torch.norm(pred - target)
        norm = torch.norm(target)
        return (error / (norm + 1e-10) * 100).item()
    
    @staticmethod
    def calculate_all_metrics(pred: torch.Tensor, target: torch.Tensor, 
                              prefix: str = '') -> Dict[str, float]:
        """计算所有指标"""
        metrics = {
            f'{prefix}mse': MetricsCalculator.mse(pred, target),
            f'{prefix}mae': MetricsCalculator.mae(pred, target),
            f'{prefix}rmse': MetricsCalculator.rmse(pred, target),
            f'{prefix}nmse_db': MetricsCalculator.nmse(pred, target),
            f'{prefix}psnr': MetricsCalculator.psnr(pred, target),
            f'{prefix}snr': MetricsCalculator.snr(pred, target),
            f'{prefix}cosine_sim': MetricsCalculator.cosine_similarity(pred, target),
            f'{prefix}correlation': MetricsCalculator.correlation_coefficient(pred, target),
            f'{prefix}relative_error': MetricsCalculator.relative_error(pred, target),
        }
        return metrics


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda', 
                 test_loader = None):
        """
        Args:
            checkpoint_path: checkpoint文件路径
            device: 设备 ('cuda' 或 'cpu')
            test_loader: 预先加载的测试数据加载器（可选）
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 🆕 加载checkpoint（使用自己的config，不用共享config）
        print(f"\n📦 Loading checkpoint: {self.checkpoint_path}")
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # 🆕 使用checkpoint自己的config
        self.config = self.checkpoint['config']
        
        # 确定模型类型
        model_name = self.config['model']['name']
        is_ablation = self.config['model'].get('ablation', {}).get('enabled', False)
        self.is_baseline = (model_name == 'UNetBaseline' or is_ablation)
        
        print(f"   Model type: {'Baseline' if self.is_baseline else 'Decomposition'}")
        print(f"   Epoch: {self.checkpoint['epoch']}")
        print(f"   Best val loss: {self.checkpoint['best_val_loss']:.6f}")
        
        # 构建模型
        self.model = self._build_model()
        self.model.eval()
        
        # 使用提供的数据加载器或创建新的
        if test_loader is not None:
            print(f"\n📊 Using shared test dataloader...")
            self.test_loader = test_loader
        else:
            print(f"\n📊 Creating test dataloader...")
            _, _, self.test_loader = create_dataloaders(
                self.config, 
                rank=0, 
                world_size=1, 
                use_ddp=False
            )
        
        print(f"   Test samples: {len(self.test_loader.dataset)}")
        print(f"   Batch size: {self.test_loader.batch_size}")
        
    def _build_model(self) -> nn.Module:
        """构建模型"""
        if self.is_baseline:
            model = UNetBaseline(
                in_channels=self.config['model']['in_channels'],
                out_channels=self.config['model']['in_channels'],
                base_channels=self.config['model']['base_channels'],
                depth=self.config['model']['depth'],
                norm_type=self.config['model']['norm_type'],
                dropout=self.config['model']['dropout']
            )
        else:
            # 🆕 从config中获取use_attention参数
            use_attention = self.config['model'].get('use_attention', False)
            
            model = UNetDecomposer(
                in_channels=self.config['model']['in_channels'],
                base_channels=self.config['model']['base_channels'],
                depth=self.config['model']['depth'],
                norm_type=self.config['model']['norm_type'],
                dropout=self.config['model']['dropout'],
                use_attention=use_attention
            )
        
        # 加载权重
        state_dict = self.checkpoint['model_state_dict']
        
        # 处理DDP保存的模型（去掉module.前缀）
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        
        return model
    
    def _parse_model_output(self, pred):
        """
        解析模型输出，处理不同的返回格式
        
        Returns:
            (pred_static, pred_dynamic, pred_total) 或 (None, None, pred_total)
        """
        # 情况1: 字典格式 {'static': ..., 'dynamic': ...}
        if isinstance(pred, dict):
            if 'static' in pred and 'dynamic' in pred:
                pred_static = pred['static']
                pred_dynamic = pred['dynamic']
                pred_total = pred_static + pred_dynamic
                return pred_static, pred_dynamic, pred_total
            elif 'output' in pred:
                return None, None, pred['output']
            else:
                # 尝试找到第一个tensor
                for v in pred.values():
                    if isinstance(v, torch.Tensor):
                        return None, None, v
                raise ValueError(f"Cannot find tensor in dict output: {pred.keys()}")
        
        # 情况2: 元组格式 (static, dynamic)
        elif isinstance(pred, tuple):
            if len(pred) == 2:
                pred_static, pred_dynamic = pred
                pred_total = pred_static + pred_dynamic
                return pred_static, pred_dynamic, pred_total
            elif len(pred) == 1:
                return None, None, pred[0]
            else:
                raise ValueError(f"Unexpected tuple length: {len(pred)}")
        
        # 情况3: 直接返回tensor
        elif isinstance(pred, torch.Tensor):
            return None, None, pred
        
        else:
            raise ValueError(f"Unexpected model output type: {type(pred)}")
    
    @torch.no_grad()
    def evaluate(self) -> Dict:
        """在测试集上评估"""
        print(f"\n{'='*80}")
        print("🔍 Evaluating on test set...")
        print(f"{'='*80}\n")
        
        all_metrics = {
            'total': [],
            'static': [] if not self.is_baseline else None,
            'dynamic': [] if not self.is_baseline else None,
        }
        
        # 用于保存样本（用于可视化）
        sample_inputs = []
        sample_preds = []
        sample_targets = []
        
        for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
            inputs = batch['input'].to(self.device)
            target_static = batch['static'].to(self.device)
            target_dynamic = batch['dynamic'].to(self.device)
            target_total = batch['target'].to(self.device)
            
            # 前向传播
            pred = self.model(inputs)
            
            # 解析模型输出
            pred_static, pred_dynamic, pred_total = self._parse_model_output(pred)
            
            if self.is_baseline:
                # 基线模型：只有总信号
                metrics_total = MetricsCalculator.calculate_all_metrics(
                    pred_total, target_total, prefix='total_'
                )
                all_metrics['total'].append(metrics_total)
                
            else:
                # 分解模型：有静态、动态和总信号
                if pred_static is not None and pred_dynamic is not None:
                    # 计算静态分量指标
                    metrics_static = MetricsCalculator.calculate_all_metrics(
                        pred_static, target_static, prefix='static_'
                    )
                    all_metrics['static'].append(metrics_static)
                    
                    # 计算动态分量指标
                    metrics_dynamic = MetricsCalculator.calculate_all_metrics(
                        pred_dynamic, target_dynamic, prefix='dynamic_'
                    )
                    all_metrics['dynamic'].append(metrics_dynamic)
                
                # 计算总信号指标
                metrics_total = MetricsCalculator.calculate_all_metrics(
                    pred_total, target_total, prefix='total_'
                )
                all_metrics['total'].append(metrics_total)
            
            # 🆕 保存前几个batch用于可视化（统一使用字典格式）
            if batch_idx < 2:
                sample_inputs.append(inputs.cpu())
                
                # 统一转换为字典格式
                if pred_static is not None and pred_dynamic is not None:
                    sample_preds.append({
                        'static': pred_static.cpu(),
                        'dynamic': pred_dynamic.cpu(),
                        'total': pred_total.cpu()
                    })
                else:
                    sample_preds.append({
                        'total': pred_total.cpu()
                    })
                
                sample_targets.append({
                    'static': target_static.cpu(),
                    'dynamic': target_dynamic.cpu(),
                    'target': target_total.cpu()
                })
        
        # 计算平均指标
        avg_metrics = {}
        
        for component, metrics_list in all_metrics.items():
            if metrics_list is None or len(metrics_list) == 0:
                continue
            
            # 转换为DataFrame
            df = pd.DataFrame(metrics_list)
            
            # 计算统计量
            avg_metrics[component] = {
                'mean': df.mean().to_dict(),
                'std': df.std().to_dict(),
                'min': df.min().to_dict(),
                'max': df.max().to_dict(),
            }
        
        # 保存样本
        self.sample_inputs = torch.cat(sample_inputs, dim=0) if sample_inputs else None
        self.sample_preds = sample_preds
        self.sample_targets = sample_targets
        
        return avg_metrics
    
    def print_metrics(self, metrics: Dict):
        """打印评估指标"""
        print(f"\n{'='*80}")
        print("📊 Evaluation Results")
        print(f"{'='*80}\n")
        
        for component, stats in metrics.items():
            print(f"\n{'='*80}")
            print(f"📈 {component.upper()} Component Metrics")
            print(f"{'='*80}")
            
            try:
                # 检查stats结构
                if not stats or not isinstance(stats, dict):
                    print(f"⚠️  No data available for {component}")
                    continue
                
                # 检查是否有必要的键
                required_keys = ['mean', 'std', 'min', 'max']
                if not all(key in stats for key in required_keys):
                    print(f"⚠️  Incomplete statistics for {component}")
                    print(f"   Available keys: {list(stats.keys())}")
                    continue
                
                # 检查是否有数据
                if not stats['mean']:
                    print(f"⚠️  No metrics data for {component}")
                    continue
                
                # 创建DataFrame
                df = pd.DataFrame(stats)
                
                # 重新排列列的顺序（如果列存在的话）
                available_cols = [col for col in ['mean', 'std', 'min', 'max'] if col in df.columns]
                if available_cols:
                    df = df[available_cols]
                
                print(df.to_string())
                print()
                
            except Exception as e:
                print(f"❌ Error printing metrics for {component}: {e}")
                import traceback
                traceback.print_exc()
    
    def save_metrics(self, metrics: Dict, output_dir: Path):
        """保存评估指标"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存为JSON
        json_file = output_dir / 'test_metrics.json'
        with open(json_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Metrics saved to: {json_file}")
        
        # 保存为CSV
        for component, stats in metrics.items():
            if not stats:
                continue
            csv_file = output_dir / f'test_metrics_{component}.csv'
            df = pd.DataFrame(stats)
            df.to_csv(csv_file)
            print(f"✅ {component} metrics saved to: {csv_file}")
        
        # 创建汇总表格
        summary_data = []
        for component, stats in metrics.items():
            if not stats or 'mean' not in stats:
                continue
            for metric_name, value in stats['mean'].items():
                summary_data.append({
                    'Component': component,
                    'Metric': metric_name,
                    'Mean': value,
                    'Std': stats['std'].get(metric_name, np.nan),
                    'Min': stats['min'].get(metric_name, np.nan),
                    'Max': stats['max'].get(metric_name, np.nan)
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = output_dir / 'test_metrics_summary.csv'
            summary_df.to_csv(summary_file, index=False)
            print(f"✅ Summary saved to: {summary_file}")
    
    def create_visualizations(self, output_dir: Path):
        """创建可视化图表"""
        print(f"\n{'='*80}")
        print("📊 Creating Visualizations...")
        print(f"{'='*80}\n")
        
        if self.sample_inputs is None or len(self.sample_preds) == 0:
            print("⚠️  No samples available for visualization")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建对比图
        print("   - Creating comparison grids...")
        try:
            for mode in ['magnitude', 'phase']:
                for batch_idx in range(min(2, len(self.sample_preds))):
                    inputs = self.sample_inputs[batch_idx:batch_idx+1]
                    pred_dict = self.sample_preds[batch_idx]
                    target = {
                        'static': self.sample_targets[batch_idx]['static'][0:1],
                        'dynamic': self.sample_targets[batch_idx]['dynamic'][0:1],
                        'target': self.sample_targets[batch_idx]['target'][0:1]
                    }
                    
                    # 🆕 根据is_baseline决定pred格式
                    if self.is_baseline:
                        # 基线模型
                        pred = pred_dict['total'][0:1]
                    else:
                        # 分解模型：传递字典（包含static和dynamic）
                        pred = {
                            'static': pred_dict['static'][0:1],
                            'dynamic': pred_dict['dynamic'][0:1]
                        }
                    
                    grid_img = create_comparison_grid(
                        inputs, pred, target,
                        is_baseline=self.is_baseline,
                        num_samples=1,
                        mode=mode
                    )
                    
                    plt.figure(figsize=(20, 12))
                    plt.imshow(grid_img)
                    plt.axis('off')
                    plt.tight_layout()
                    
                    save_path = output_dir / f'comparison_{mode}_batch{batch_idx}.png'
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    print(f"      ✅ Saved: {save_path.name}")
        
        except Exception as e:
            print(f"      ⚠️  Failed to create comparison grids: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n✅ Visualizations created!")


def evaluate_single_checkpoint(checkpoint_path: str, output_dir: str, 
                               device: str = 'cuda', test_loader=None):
    """评估单个checkpoint"""
    print(f"\n{'='*80}")
    print(f"🔬 Evaluating Checkpoint")
    print(f"{'='*80}")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Output: {output_dir}")
    print(f"{'='*80}\n")
    
    # 创建评估器（不传递config_override，让它使用自己的config）
    evaluator = ModelEvaluator(checkpoint_path, device, test_loader)
    
    # 评估
    metrics = evaluator.evaluate()
    
    # 打印指标
    evaluator.print_metrics(metrics)
    
    # 保存指标
    output_path = Path(output_dir)
    evaluator.save_metrics(metrics, output_path)
    
    # 创建可视化
    evaluator.create_visualizations(output_path / 'visualizations')
    
    print(f"\n{'='*80}")
    print("✅ Evaluation Complete!")
    print(f"{'='*80}")
    print(f"📁 Results saved to: {output_path}")
    print(f"{'='*80}\n")
    
    return metrics


def batch_evaluate_checkpoints(base_dir: str, pattern: str = 'Ablation',
                               output_dir: str = None, device: str = 'cuda'):
    """批量评估多个checkpoint（共享数据加载器）"""
    print(f"\n{'='*80}")
    print(f"🔬 Batch Evaluation")
    print(f"{'='*80}")
    print(f"   Base directory: {base_dir}")
    print(f"   Pattern: {pattern}")
    print(f"{'='*80}\n")
    
    base_path = Path(base_dir)
    
    # 设置默认输出目录
    if output_dir is None:
        output_dir = base_path / 'eval_result'
        print(f"💡 Using default output directory: {output_dir}\n")
    
    # 查找所有checkpoint
    checkpoints = []
    for category_dir in base_path.iterdir():
        if not category_dir.is_dir():
            continue
        
        for exp_dir in category_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            if pattern in exp_dir.name:
                checkpoint_path = exp_dir / 'checkpoints' / 'best.pth'
                if checkpoint_path.exists():
                    checkpoints.append((exp_dir, checkpoint_path))
    
    print(f"📊 Found {len(checkpoints)} checkpoints to evaluate\n")
    
    if not checkpoints:
        print("❌ No checkpoints found!")
        return
    
    # 创建共享的数据加载器（只加载一次）
    print(f"\n{'='*80}")
    print("📦 Creating shared test dataloader (loading once for all models)...")
    print(f"{'='*80}\n")
    
    # 从第一个checkpoint加载配置
    first_checkpoint = torch.load(checkpoints[0][1], map_location='cpu')
    shared_config = first_checkpoint['config']
    
    _, _, shared_test_loader = create_dataloaders(
        shared_config,
        rank=0,
        world_size=1,
        use_ddp=False
    )
    
    print(f"\n✅ Shared dataloader created!")
    print(f"   Test samples: {len(shared_test_loader.dataset)}")
    print(f"   This dataloader will be reused for all {len(checkpoints)} models\n")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory created: {output_path}\n")
    
    # 评估每个checkpoint
    all_results = []
    
    for idx, (exp_dir, checkpoint_path) in enumerate(checkpoints):
        print(f"\n{'='*80}")
        print(f"📦 Evaluating [{idx+1}/{len(checkpoints)}]: {exp_dir.parent.name}/{exp_dir.name}")
        print(f"{'='*80}\n")
        
        try:
            # 创建该实验的输出目录
            exp_output = output_path / exp_dir.parent.name / exp_dir.name
            
            # 🆕 不传递config_override，让每个模型使用自己的config
            metrics = evaluate_single_checkpoint(
                str(checkpoint_path),
                str(exp_output),
                device,
                test_loader=shared_test_loader  # 只共享数据加载器
            )
            
            # 记录结果
            result = {
                'category': exp_dir.parent.name,
                'experiment': exp_dir.name,
                'checkpoint': str(checkpoint_path),
            }
            
            # 添加指标
            for component, stats in metrics.items():
                if not stats or 'mean' not in stats:
                    continue
                for metric_name, value in stats['mean'].items():
                    result[f'{component}_{metric_name}_mean'] = value
                    result[f'{component}_{metric_name}_std'] = stats['std'].get(metric_name, np.nan)
            
            all_results.append(result)
            
            print(f"✅ Successfully evaluated: {exp_dir.name}\n")
        
        except Exception as e:
            print(f"❌ Failed to evaluate {exp_dir.name}: {e}\n")
            import traceback
            traceback.print_exc()
    
    # 保存汇总结果
    if all_results:
        summary_df = pd.DataFrame(all_results)
        
        # 排序：按total_nmse_db_mean升序
        if 'total_total_nmse_db_mean' in summary_df.columns:
            summary_df = summary_df.sort_values('total_total_nmse_db_mean')
        
        summary_file = output_path / 'evaluation_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\n✅ Summary saved to: {summary_file}")
        
        # 保存为JSON
        summary_json = output_path / 'evaluation_summary.json'
        with open(summary_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"✅ JSON summary saved to: {summary_json}")
        
        # 打印简要统计
        print(f"\n{'='*80}")
        print("📊 Quick Summary")
        print(f"{'='*80}")
        if 'total_total_nmse_db_mean' in summary_df.columns:
            print(f"\n🏆 Best Model:")
            best_row = summary_df.iloc[0]
            print(f"   {best_row['category']}/{best_row['experiment']}")
            print(f"   Total NMSE: {best_row['total_total_nmse_db_mean']:.2f} dB")
            
            print(f"\n📈 Top 5 Models:")
            for i, (idx, row) in enumerate(summary_df.head(5).iterrows()):
                print(f"   {i+1}. {row['category']}/{row['experiment'][:40]}")
                print(f"      Total NMSE: {row['total_total_nmse_db_mean']:.2f} dB")
    
    print(f"\n{'='*80}")
    print("✅ Batch Evaluation Complete!")
    print(f"{'='*80}")
    print(f"📁 Results saved to: {output_path}")
    print(f"📊 Data was loaded only ONCE and reused for all {len(checkpoints)} models")
    print(f"   (Each model used its own config for correct architecture)")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Trained Models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['single', 'batch'],
        default='batch',
        help='Evaluation mode'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file (for single mode)'
    )
    
    parser.add_argument(
        '--base_dir',
        type=str,
        default='/LSEM/user/chenyinda/code/signal_dy_static/1104/results_20251103_180526',
        help='Base directory containing experiments (for batch mode)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='Ablation',
        help='Pattern to match experiment directories (for batch mode)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for evaluation results (default: {base_dir}/eval_result)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    # 打印配置信息
    print(f"\n{'='*80}")
    print("⚙️  Configuration")
    print(f"{'='*80}")
    print(f"   Mode: {args.mode}")
    if args.mode == 'single':
        print(f"   Checkpoint: {args.checkpoint}")
    else:
        print(f"   Base directory: {args.base_dir}")
        print(f"   Pattern: {args.pattern}")
    if args.output_dir:
        print(f"   Output directory: {args.output_dir}")
    else:
        if args.mode == 'batch':
            print(f"   Output directory: {args.base_dir}/eval_result (default)")
    print(f"   Device: {args.device}")
    print(f"{'='*80}\n")
    
    if args.mode == 'single':
        if not args.checkpoint:
            print("❌ Error: --checkpoint is required for single mode")
            print("💡 Example: python eval_best_ckpt.py --mode single --checkpoint /path/to/checkpoint/best.pth")
            return
        
        # 单个模式的默认输出目录
        if args.output_dir is None:
            checkpoint_path = Path(args.checkpoint)
            args.output_dir = checkpoint_path.parent.parent / 'evaluation'
        
        evaluate_single_checkpoint(
            args.checkpoint,
            args.output_dir,
            args.device
        )
    
    elif args.mode == 'batch':
        # 检查base_dir是否存在
        if not Path(args.base_dir).exists():
            print(f"❌ Error: Base directory does not exist: {args.base_dir}")
            print("💡 Please specify a valid --base_dir")
            return
        
        # 批量评估
        batch_evaluate_checkpoints(
            args.base_dir,
            args.pattern,
            args.output_dir,
            args.device
        )


if __name__ == '__main__':
    main()