"""
消融实验结果收集和分析脚本（增强版）
自动读取所有消融实验的结果并生成对比报告
支持提取checkpoint中的所有可用信息
"""

import os
import torch
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import argparse
import json
from collections import defaultdict


def parse_args():
    """
    解析命令行参数
    
    Returns:
        args: 参数命名空间
    """
    parser = argparse.ArgumentParser(
        description='消融实验结果收集和分析脚本（增强版）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 路径参数
    parser.add_argument(
        '--base_dir',
        type=str,
        default='/LSEM/user/chenyinda/code/signal_dy_static/1104/results_20251104_092511',
        help='实验基础目录（读取checkpoint的位置）'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/LSEM/user/chenyinda/code/signal_dy_static/1104/results_20251104_092511/ablation_analysis',
        help='分析结果输出目录'
    )
    
    parser.add_argument(
        '--csv_file',
        type=str,
        default='ablation_results.csv',
        help='CSV结果文件名'
    )
    
    parser.add_argument(
        '--detailed_csv_file',
        type=str,
        default='ablation_results_detailed.csv',
        help='详细CSV结果文件名（包含所有提取的信息）'
    )
    
    parser.add_argument(
        '--latex_file',
        type=str,
        default='ablation_table.tex',
        help='LaTeX表格文件名'
    )
    
    parser.add_argument(
        '--checkpoint_keys_file',
        type=str,
        default='checkpoint_keys.json',
        help='Checkpoint keys信息保存文件'
    )
    
    # 实验匹配参数
    parser.add_argument(
        '--pattern',
        type=str,
        default='Ablation',
        help='实验目录名称匹配模式（在第二层目录中匹配）'
    )
    
    parser.add_argument(
        '--exclude_dirs',
        type=str,
        nargs='*',
        default=['ablation_analysis', '.ipynb_checkpoints', '__pycache__'],
        help='要排除的第一层目录名称列表'
    )
    
    parser.add_argument(
        '--checkpoint_name',
        type=str,
        default='best.pth',
        help='checkpoint文件名'
    )
    
    # 可视化参数
    parser.add_argument(
        '--figure_dpi',
        type=int,
        default=300,
        help='图表分辨率'
    )
    
    parser.add_argument(
        '--figure_width',
        type=float,
        default=12.0,
        help='图表宽度（英寸）'
    )
    
    parser.add_argument(
        '--figure_height',
        type=float,
        default=6.0,
        help='图表高度（英寸）'
    )
    
    parser.add_argument(
        '--style',
        type=str,
        default='seaborn-v0_8-darkgrid',
        choices=['seaborn-v0_8-darkgrid', 'seaborn-v0_8-whitegrid', 'ggplot', 'default'],
        help='matplotlib绘图风格'
    )
    
    # 其他参数
    parser.add_argument(
        '--float_format',
        type=str,
        default='%.6f',
        help='浮点数格式化字符串'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细输出信息'
    )
    
    parser.add_argument(
        '--explore_first',
        action='store_true',
        help='先探索第一个checkpoint的结构，然后询问是否继续'
    )
    
    args = parser.parse_args()
    return args


def explore_nested_dict(data: Any, prefix: str = '', max_depth: int = 5, current_depth: int = 0) -> Dict[str, Any]:
    """
    递归探索嵌套字典/对象的结构
    
    Args:
        data: 要探索的数据
        prefix: 键的前缀
        max_depth: 最大递归深度
        current_depth: 当前递归深度
    
    Returns:
        结构信息字典
    """
    info = {}
    
    if current_depth >= max_depth:
        return {prefix: f"<max depth reached, type: {type(data).__name__}>"}
    
    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, (dict, list, tuple)):
                info.update(explore_nested_dict(value, full_key, max_depth, current_depth + 1))
            elif isinstance(value, torch.Tensor):
                info[full_key] = f"Tensor(shape={list(value.shape)}, dtype={value.dtype})"
            elif isinstance(value, (int, float, str, bool, type(None))):
                info[full_key] = f"{type(value).__name__}: {value}"
            else:
                info[full_key] = f"<{type(value).__name__}>"
    
    elif isinstance(data, (list, tuple)):
        info[prefix] = f"{type(data).__name__}(len={len(data)})"
        if len(data) > 0 and current_depth < max_depth - 1:
            # 只探索第一个元素作为示例
            info.update(explore_nested_dict(data[0], f"{prefix}[0]", max_depth, current_depth + 1))
    
    elif isinstance(data, torch.Tensor):
        info[prefix] = f"Tensor(shape={list(data.shape)}, dtype={data.dtype})"
    
    else:
        info[prefix] = f"<{type(data).__name__}>"
    
    return info


def explore_checkpoint_structure(checkpoint_path: Path) -> Dict[str, Any]:
    """
    探索checkpoint的完整结构
    
    Args:
        checkpoint_path: checkpoint文件路径
    
    Returns:
        结构信息字典
    """
    print(f"\n{'='*80}")
    print(f"🔍 探索Checkpoint结构: {checkpoint_path.name}")
    print(f"{'='*80}\n")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 顶层keys
        print("📋 顶层Keys:")
        for key in checkpoint.keys():
            value = checkpoint[key]
            if isinstance(value, torch.Tensor):
                print(f"   - {key}: Tensor(shape={list(value.shape)}, dtype={value.dtype})")
            elif isinstance(value, dict):
                print(f"   - {key}: dict (len={len(value)})")
            elif isinstance(value, (list, tuple)):
                print(f"   - {key}: {type(value).__name__} (len={len(value)})")
            else:
                print(f"   - {key}: {type(value).__name__} = {value}")
        
        # 详细结构
        print(f"\n📊 详细结构:")
        structure = explore_nested_dict(checkpoint)
        
        for key, value in sorted(structure.items()):
            print(f"   {key}: {value}")
        
        print(f"\n{'='*80}\n")
        
        return structure
    
    except Exception as e:
        print(f"❌ 探索失败: {e}")
        return {}


def extract_value_from_nested(data: Any, key_path: str, default: Any = None) -> Any:
    """
    从嵌套字典中提取值
    
    Args:
        data: 数据字典
        key_path: 键路径，例如 "optimizer.lr"
        default: 默认值
    
    Returns:
        提取的值
    """
    keys = key_path.split('.')
    current = data
    
    try:
        for key in keys:
            if isinstance(current, dict):
                current = current[key]
            else:
                return default
        return current
    except (KeyError, TypeError, IndexError):
        return default


def load_checkpoint_metrics(checkpoint_path: Path, verbose: bool = False, all_keys: set = None) -> Dict:
    """
    从checkpoint文件加载所有可用指标（增强版）
    支持 val_metrics 的完整展开与类型安全提取
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 收集顶层keys
        if all_keys is not None:
            all_keys.update(checkpoint.keys())

        if verbose:
            print(f"\n📋 Checkpoint Keys: {list(checkpoint.keys())}")

        metrics = {
            'epoch': checkpoint.get('epoch', 0),
            'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        }

        # 优化 val_metrics 解析
        if 'val_metrics' in checkpoint:
            val_metrics_data = checkpoint['val_metrics']

            if isinstance(val_metrics_data, dict):
                for k, v in val_metrics_data.items():
                    if isinstance(v, torch.Tensor):
                        if v.numel() == 1:
                            metrics[f'val_metrics_{k}'] = v.item()
                        else:
                            metrics[f'val_metrics_{k}'] = v.mean().item()
                    elif isinstance(v, (int, float)):
                        metrics[f'val_metrics_{k}'] = v
                    elif isinstance(v, (list, tuple)):
                        if all(isinstance(x, (int, float)) for x in v):
                            metrics[f'val_metrics_{k}'] = float(np.mean(v))
                        else:
                            metrics[f'val_metrics_{k}_len'] = len(v)
                    else:
                        metrics[f'val_metrics_{k}_type'] = type(v).__name__

            elif isinstance(val_metrics_data, torch.Tensor):
                metrics['val_metrics_tensor_shape'] = list(val_metrics_data.shape)
            else:
                metrics['val_metrics_type'] = type(val_metrics_data).__name__

            if all_keys is not None:
                for subk in (val_metrics_data.keys() if isinstance(val_metrics_data, dict) else []):
                    all_keys.add(f"val_metrics.{subk}")

        # 兼容旧字段如 best_val_metrics
        if 'best_val_metrics' in checkpoint and isinstance(checkpoint['best_val_metrics'], dict):
            for k, v in checkpoint['best_val_metrics'].items():
                try:
                    metrics[f'best_val_metrics_{k}'] = float(v)
                except Exception:
                    metrics[f'best_val_metrics_{k}_type'] = type(v).__name__

        # 统计模型参数数量
        if 'model_state_dict' in checkpoint and isinstance(checkpoint['model_state_dict'], dict):
            params = checkpoint['model_state_dict']
            metrics['model_total_params'] = sum(
                p.numel() for p in params.values() if isinstance(p, torch.Tensor)
            )
            metrics['model_param_count'] = len(params)

        # 补充基础配置标志
        metrics['config_available'] = 'config' in checkpoint
        metrics['optimizer_available'] = 'optimizer_state_dict' in checkpoint
        metrics['scheduler_available'] = 'scheduler_state_dict' in checkpoint

        if verbose:
            print(f"   ✅ 提取到 {len(metrics)} 个指标:")
            for k, v in sorted(metrics.items()):
                print(f"      - {k}: {v}")

        return metrics

    except Exception as e:
        print(f"   ⚠️  加载checkpoint失败: {checkpoint_path}")
        print(f"      错误: {e}")
        return None


def find_experiment_dirs(base_dir: Path, pattern: str = 'Ablation', 
                         exclude_dirs: List[str] = None, verbose: bool = False) -> List[Path]:
    """
    查找所有消融实验目录（两层结构）
    路径格式: base_dir/category_dir/Ablation*/checkpoints/best.pth
    
    Args:
        base_dir: 实验基础目录
        pattern: 第二层目录名称匹配模式
        exclude_dirs: 要排除的第一层目录名称列表
        verbose: 是否显示详细信息
    
    Returns:
        实验目录列表
    """
    exp_dirs = []
    
    if exclude_dirs is None:
        exclude_dirs = []
    
    if not base_dir.exists():
        print(f"❌ 基础目录不存在: {base_dir}")
        return exp_dirs
    
    if verbose:
        print(f"\n🔍 开始查找实验目录...")
        print(f"   基础目录: {base_dir}")
        print(f"   匹配模式: '{pattern}'")
        print(f"   排除目录: {exclude_dirs}")
    
    # 遍历第一层目录
    for category_dir in base_dir.iterdir():
        if not category_dir.is_dir():
            if verbose:
                print(f"   ⏭️  跳过非目录: {category_dir.name}")
            continue
        
        if category_dir.name in exclude_dirs:
            if verbose:
                print(f"   🚫 排除目录: {category_dir.name}")
            continue
        
        if verbose:
            print(f"\n   📂 检查类别目录: {category_dir.name}")
        
        # 遍历第二层目录
        for exp_dir in category_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            if pattern in exp_dir.name:
                checkpoint_path = exp_dir / 'checkpoints' / 'best.pth'
                if checkpoint_path.exists():
                    exp_dirs.append(exp_dir)
                    if verbose:
                        print(f"      ✅ 找到实验: {exp_dir.name}")
                else:
                    if verbose:
                        print(f"      ⚠️  {exp_dir.name} 缺少 checkpoints/best.pth")
            else:
                if verbose:
                    print(f"      ⏭️  {exp_dir.name} 不匹配模式 '{pattern}'")
    
    if verbose:
        print(f"\n📊 总共找到 {len(exp_dirs)} 个有效实验目录\n")
    
    return sorted(exp_dirs)


def extract_experiment_info(exp_dir: Path, verbose: bool = False) -> Dict:
    """
    从实验目录提取信息
    
    Args:
        exp_dir: 实验目录路径
        verbose: 是否显示详细信息
    
    Returns:
        info: 实验信息字典
    """
    info = {
        'name': exp_dir.name,
        'category': exp_dir.parent.name,
        'path': str(exp_dir),
    }
    
    # 读取experiment_info.yaml
    info_file = exp_dir / 'experiment_info.yaml'
    if info_file.exists():
        try:
            with open(info_file, 'r', encoding='utf-8') as f:
                exp_info = yaml.safe_load(f)
                info.update(exp_info)
                if verbose:
                    print(f"   ✅ 读取experiment_info.yaml")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  读取experiment_info.yaml失败: {e}")
    
    # 读取config.yaml
    config_file = exp_dir / 'config.yaml'
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                info['config'] = config
                if verbose:
                    print(f"   ✅ 读取config.yaml")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  读取config.yaml失败: {e}")
    
    return info


def collect_all_results(args) -> tuple:
    """
    收集所有消融实验的结果
    
    Args:
        args: 命令行参数
    
    Returns:
        (results_df, detailed_df, all_keys): 基础结果DataFrame、详细DataFrame和所有遇到的keys
    """
    base_path = Path(args.base_dir)
    
    if not base_path.exists():
        print(f"❌ 实验目录不存在: {args.base_dir}")
        return None, None, None
    
    # 查找所有实验
    exp_dirs = find_experiment_dirs(base_path, args.pattern, args.exclude_dirs, args.verbose)
    
    if not exp_dirs:
        print(f"⚠️  未找到任何实验目录")
        print(f"   匹配模式: '{args.pattern}'")
        print(f"   排除目录: {args.exclude_dirs}")
        print(f"\n💡 提示: 请检查目录结构是否为: base_dir/*/Ablation*/checkpoints/best.pth")
        return None, None, None
    
    print(f"\n📊 找到 {len(exp_dirs)} 个实验目录:")
    for exp in exp_dirs:
        print(f"   - {exp.parent.name}/{exp.name}")
    print()
    
    # 如果启用explore_first，先探索第一个checkpoint
    if args.explore_first and len(exp_dirs) > 0:
        first_checkpoint = exp_dirs[0] / 'checkpoints' / args.checkpoint_name
        if first_checkpoint.exists():
            explore_checkpoint_structure(first_checkpoint)
            
            response = input("📝 是否继续收集所有实验结果? (y/n): ").strip().lower()
            if response != 'y':
                print("❌ 用户取消操作")
                return None, None, None
    
    # 收集所有遇到的keys
    all_keys = set()
    
    # 收集结果
    results = []
    detailed_results = []
    
    for exp_dir in exp_dirs:
        print(f"🔍 处理实验: {exp_dir.parent.name}/{exp_dir.name}")
        
        # 提取实验信息
        info = extract_experiment_info(exp_dir, args.verbose)
        
        # 查找best checkpoint
        checkpoint_dir = exp_dir / 'checkpoints'
        best_ckpt = checkpoint_dir / args.checkpoint_name
        
        if not best_ckpt.exists():
            print(f"   ⚠️  未找到{args.checkpoint_name}，跳过")
            continue
        
        # 加载指标
        metrics = load_checkpoint_metrics(best_ckpt, args.verbose, all_keys)
        
        if metrics is None:
            continue
        
        # 组合基础结果
        result = {
            'Category': info.get('category', 'Unknown'),
            'Experiment': info.get('experiment_name', exp_dir.name),
            'Model': info.get('model_name', 'Unknown'),
            'Is_Ablation': info.get('is_ablation', False),
            'Temporal': info.get('temporal_enabled', True),
            'Epoch': metrics.get('epoch', 0),
            'Best_Val_Loss': metrics.get('best_val_loss', float('inf')),
        }
        
        # 从config提取额外信息
        if 'config' in info:
            config = info['config']
            result.update({
                'Use_Attention': config.get('model', {}).get('use_attention', False),
                'Separation_Weight': config.get('loss', {}).get('separation_weight', 0),
                'Dynamic_MSE_Weight': config.get('loss', {}).get('weights', {}).get('dynamic_mse', 0),
                'Learning_Rate': config.get('training', {}).get('learning_rate', 0),
            })
        
        results.append(result)
        
        # 组合详细结果（包含所有提取的metrics）
        detailed_result = result.copy()
        detailed_result.update(metrics)
        detailed_results.append(detailed_result)
        
        print(f"   ✅ 已添加结果 (Loss: {result['Best_Val_Loss']:.6f}, 提取指标数: {len(metrics)})")
    
    if not results:
        print("❌ 未收集到任何有效结果，请检查实验目录")
        return None, None, None
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    detailed_df = pd.DataFrame(detailed_results)
    
    # 排序
    df = df.sort_values('Best_Val_Loss')
    detailed_df = detailed_df.sort_values('Best_Val_Loss')
    
    return df, detailed_df, all_keys


def create_comparison_plots(df: pd.DataFrame, args):
    """
    创建对比可视化图表
    
    Args:
        df: 结果DataFrame
        args: 命令行参数
    """
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 设置绘图风格
    plt.style.use(args.style)
    sns.set_palette("husl")
    
    # 1. Loss对比条形图
    fig, ax = plt.subplots(figsize=(args.figure_width, args.figure_height))
    
    labels = [f"{row['Category']}\n{row['Experiment'].replace('Ablation', 'Abl')[:20]}" 
              for _, row in df.iterrows()]
    losses = df['Best_Val_Loss'].values
    
    # 颜色映射
    color_map = {
        'full': 'green',
        'baseunet': 'orange',
        'baseline': 'orange',
    }
    
    colors = []
    for category in df['Category'].values:
        category_lower = category.lower()
        color = 'skyblue'
        for key, val in color_map.items():
            if key in category_lower:
                color = val
                break
        colors.append(color)
    
    bars = ax.bar(range(len(labels)), losses, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study - Loss Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, loss) in enumerate(zip(bars, losses)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}',
                ha='center', va='bottom', fontsize=7, rotation=0)
    
    plt.tight_layout()
    loss_comparison_file = output_path / 'loss_comparison.png'
    plt.savefig(loss_comparison_file, dpi=args.figure_dpi, bbox_inches='tight')
    print(f"✅ 保存图表: {loss_comparison_file}")
    plt.close()
    
    # 2. 按类别分组的Loss对比图
    fig, ax = plt.subplots(figsize=(args.figure_width, args.figure_height))
    
    categories = df['Category'].unique()
    x_pos = 0
    xticks = []
    xticklabels = []
    
    for category in categories:
        category_df = df[df['Category'] == category]
        n_exps = len(category_df)
        
        positions = range(x_pos, x_pos + n_exps)
        losses = category_df['Best_Val_Loss'].values
        
        color = color_map.get(category.lower(), 'skyblue')
        bars = ax.bar(positions, losses, color=color, alpha=0.7, 
                      edgecolor='black', label=category)
        
        for pos, loss in zip(positions, losses):
            ax.text(pos, loss, f'{loss:.4f}',
                   ha='center', va='bottom', fontsize=7)
        
        xticks.extend(positions)
        xticklabels.extend([exp.replace('Ablation', 'Abl')[:15] 
                           for exp in category_df['Experiment'].values])
        
        x_pos += n_exps + 1
    
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study - Loss by Category', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    category_comparison_file = output_path / 'loss_by_category.png'
    plt.savefig(category_comparison_file, dpi=args.figure_dpi, bbox_inches='tight')
    print(f"✅ 保存图表: {category_comparison_file}")
    plt.close()
    
    # 3. 如果有更多指标，创建额外的对比图
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    metric_columns = [col for col in numeric_columns 
                     if col not in ['Epoch', 'Is_Ablation']]
    
    if len(metric_columns) > 2:
        # 创建多指标对比图
        n_metrics = min(len(metric_columns), 6)  # 最多显示6个指标
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metric_columns[:n_metrics]):
            ax = axes[idx]
            
            # 过滤掉NaN值
            valid_data = df[df[metric].notna()]
            if len(valid_data) == 0:
                continue
            
            labels = [f"{row['Category'][:10]}\n{row['Experiment'][:15]}" 
                     for _, row in valid_data.iterrows()]
            values = valid_data[metric].values
            
            ax.bar(range(len(labels)), values, alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
            ax.set_ylabel(metric, fontsize=10)
            ax.set_title(f'{metric} Comparison', fontsize=11)
            ax.grid(axis='y', alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        multi_metric_file = output_path / 'multi_metric_comparison.png'
        plt.savefig(multi_metric_file, dpi=args.figure_dpi, bbox_inches='tight')
        print(f"✅ 保存图表: {multi_metric_file}")
        plt.close()
    
    print(f"\n📊 所有图表已保存到: {output_path}")


def generate_latex_table(df: pd.DataFrame, args):
    """
    生成LaTeX格式的表格
    
    Args:
        df: 结果DataFrame
        args: 命令行参数
    """
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 简化列名
    columns_to_include = ['Category', 'Experiment', 'Model', 'Best_Val_Loss', 'Epoch']
    columns_to_include = [col for col in columns_to_include if col in df.columns]
    
    df_latex = df[columns_to_include].copy()
    
    # 重命名列
    column_rename = {
        'Category': 'Category',
        'Experiment': 'Configuration',
        'Model': 'Model',
        'Best_Val_Loss': 'Loss',
        'Epoch': 'Epoch'
    }
    df_latex.columns = [column_rename.get(col, col) for col in df_latex.columns]
    
    # 生成LaTeX代码
    latex_code = df_latex.to_latex(
        index=False,
        float_format=args.float_format,
        caption='Ablation Study Results',
        label='tab:ablation',
        escape=False
    )
    
    # 保存
    latex_file = output_path / args.latex_file
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print(f"\n✅ LaTeX表格已保存: {latex_file}")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    print("\n" + "="*80)
    print("🔬 消融实验结果收集与分析（增强版）")
    print("="*80)
    print(f"📁 实验目录: {args.base_dir}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"🔍 匹配模式: '{args.pattern}' (在第二层目录中匹配)")
    print(f"🚫 排除目录: {args.exclude_dirs}")
    print(f"📂 目录结构: base_dir/*/Ablation*/checkpoints/best.pth")
    print("="*80 + "\n")
    
    # 1. 收集结果
    df, detailed_df, all_keys = collect_all_results(args)
    
    if df is None or len(df) == 0:
        print("\n❌ 未能收集到有效结果，请检查:")
        print("   1. 实验目录路径是否正确")
        print("   2. 目录结构是否为: base_dir/category/Ablation*/checkpoints/best.pth")
        print("   3. best.pth 文件是否存在")
        return
    
    # 2. 打印checkpoint keys信息
    print("\n" + "="*80)
    print("🔑 所有Checkpoint中发现的Keys")
    print("="*80)
    for key in sorted(all_keys):
        print(f"   - {key}")
    print("="*80 + "\n")
    
    # 保存keys信息
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    keys_file = output_path / args.checkpoint_keys_file
    with open(keys_file, 'w', encoding='utf-8') as f:
        json.dump({
            'all_keys': sorted(list(all_keys)),
            'num_keys': len(all_keys)
        }, f, indent=2)
    print(f"✅ Checkpoint keys信息已保存: {keys_file}\n")
    
    # 3. 打印基础结果
    print("\n" + "="*80)
    print("📊 实验结果汇总（基础指标）")
    print("="*80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    # 4. 打印详细结果预览
    print("\n" + "="*80)
    print("📊 详细结果预览（包含所有提取的指标）")
    print("="*80)
    print(f"总列数: {len(detailed_df.columns)}")
    print(f"列名: {list(detailed_df.columns)}")
    print("\n前3行数据:")
    print(detailed_df.head(3).to_string(index=False))
    print("="*80 + "\n")
    
    # 5. 保存CSV
    csv_file = output_path / args.csv_file
    df.to_csv(csv_file, index=False)
    print(f"✅ 基础结果已保存: {csv_file}")
    
    detailed_csv_file = output_path / args.detailed_csv_file
    detailed_df.to_csv(detailed_csv_file, index=False)
    print(f"✅ 详细结果已保存: {detailed_csv_file}\n")
    
    # 6. 创建可视化
    print("📊 生成对比图表...")
    create_comparison_plots(df, args)
    
    # 7. 生成LaTeX表格
    print("\n📝 生成LaTeX表格...")
    generate_latex_table(df, args)
    
    # 8. 统计分析
    print("\n" + "="*80)
    print("📈 统计分析")
    print("="*80)
    
    # 按类别统计
    print("\n📊 按类别统计:")
    for category in df['Category'].unique():
        category_df = df[df['Category'] == category]
        print(f"\n   【{category}】")
        print(f"   - 实验数量: {len(category_df)}")
        print(f"   - 最佳Loss: {category_df['Best_Val_Loss'].min():.6f}")
        print(f"   - 平均Loss: {category_df['Best_Val_Loss'].mean():.6f}")
        print(f"   - 最差Loss: {category_df['Best_Val_Loss'].max():.6f}")
    
    # 查找baseline和full模型
    baseline_mask = (df['Category'].str.contains('baseline', case=False, na=False) | 
                     df['Experiment'].str.contains('Baseline', case=False, na=False))
    full_mask = (df['Category'].str.contains('full', case=False, na=False) | 
                 df['Experiment'].str.contains('Full', case=False, na=False))
    
    baseline_loss = df[baseline_mask]['Best_Val_Loss'].values
    full_loss = df[full_mask]['Best_Val_Loss'].values
    
    print("\n📊 对比分析:")
    if len(baseline_loss) > 0 and len(full_loss) > 0:
        print(f"   Baseline Loss: {baseline_loss[0]:.6f}")
        print(f"   Full Model Loss: {full_loss[0]:.6f}")
        improvement = ((baseline_loss[0] - full_loss[0]) / baseline_loss[0] * 100)
        print(f"   改进幅度: {improvement:.2f}%")
    else:
        if len(baseline_loss) == 0:
            print("   ⚠️  未找到Baseline模型")
        if len(full_loss) == 0:
            print("   ⚠️  未找到Full模型")
    
    print(f"\n📊 全局统计:")
    print(f"   最佳模型: {df.iloc[0]['Category']}/{df.iloc[0]['Experiment']}")
    print(f"   最佳Loss: {df.iloc[0]['Best_Val_Loss']:.6f}")
    print(f"   最差模型: {df.iloc[-1]['Category']}/{df.iloc[-1]['Experiment']}")
    print(f"   最差Loss: {df.iloc[-1]['Best_Val_Loss']:.6f}")
    print(f"   Loss范围: [{df['Best_Val_Loss'].min():.6f}, {df['Best_Val_Loss'].max():.6f}]")
    print(f"   Loss标准差: {df['Best_Val_Loss'].std():.6f}")
    
    # 详细指标统计
    print(f"\n📊 提取的指标统计:")
    print(f"   详细结果表格列数: {len(detailed_df.columns)}")
    numeric_cols = detailed_df.select_dtypes(include=[np.number]).columns
    print(f"   数值型指标数量: {len(numeric_cols)}")
    print(f"   数值型指标列表: {list(numeric_cols)}")
    
    print("\n" + "="*80)
    print("✅ 分析完成！")
    print("="*80 + "\n")
    
    print("📁 生成的文件:")
    print(f"   - {csv_file} (基础结果)")
    print(f"   - {detailed_csv_file} (详细结果，包含所有提取的指标)")
    print(f"   - {keys_file} (checkpoint keys信息)")
    print(f"   - {output_path / args.latex_file} (LaTeX表格)")
    print(f"   - {output_path / 'loss_comparison.png'} (Loss对比图)")
    print(f"   - {output_path / 'loss_by_category.png'} (按类别Loss对比图)")
    if (output_path / 'multi_metric_comparison.png').exists():
        print(f"   - {output_path / 'multi_metric_comparison.png'} (多指标对比图)")
    print()


if __name__ == '__main__':
    main()