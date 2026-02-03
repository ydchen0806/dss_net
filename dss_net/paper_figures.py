"""
论文级别的实验结果可视化脚本
生成高质量的消融实验对比图、误差分析图等
- Times New Roman字体
- 加粗边框
- 英文标签
- 复杂子图布局
"""

import os
import sys
import yaml
import torch
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
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================================
# 全局样式设置 - 论文级别
# ============================================================================
def setup_paper_style():
    """设置论文级别的matplotlib样式"""
    # 设置Times New Roman字体
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    rcParams['font.size'] = 12
    rcParams['axes.labelsize'] = 14
    rcParams['axes.titlesize'] = 16
    rcParams['xtick.labelsize'] = 12
    rcParams['ytick.labelsize'] = 12
    rcParams['legend.fontsize'] = 11
    rcParams['figure.titlesize'] = 18
    
    # 加粗边框
    rcParams['axes.linewidth'] = 1.5
    rcParams['xtick.major.width'] = 1.2
    rcParams['ytick.major.width'] = 1.2
    rcParams['xtick.minor.width'] = 0.8
    rcParams['ytick.minor.width'] = 0.8
    
    # 其他设置
    rcParams['axes.grid'] = True
    rcParams['grid.alpha'] = 0.3
    rcParams['grid.linewidth'] = 0.8
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.pad_inches'] = 0.1
    
    # LaTeX风格的数学公式
    rcParams['mathtext.fontset'] = 'stix'


# ============================================================================
# 数据加载
# ============================================================================
def load_ablation_results(base_dir: str) -> pd.DataFrame:
    """加载消融实验结果"""
    csv_path = Path(base_dir) / 'ablation_analysis' / 'ablation_results_detailed.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df
    else:
        print(f"⚠️  未找到结果文件: {csv_path}")
        return None


def load_checkpoint_info(checkpoint_path: Path) -> Dict:
    """加载checkpoint信息"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint
    except Exception as e:
        print(f"⚠️  加载checkpoint失败: {e}")
        return None


# ============================================================================
# 图表1: 消融实验Loss对比条形图
# ============================================================================
def plot_ablation_loss_comparison(df: pd.DataFrame, output_dir: Path):
    """
    创建消融实验Loss对比条形图
    横向条形图，按Loss排序
    """
    setup_paper_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 数据准备
    df_sorted = df.sort_values('Best_Val_Loss', ascending=True)
    
    # 创建简短的实验名称
    exp_names = []
    for _, row in df_sorted.iterrows():
        name = row['Category']
        if 'full' in name.lower():
            exp_names.append('Full Model')
        elif 'baseunet' in name.lower() or 'baseline' in name.lower():
            exp_names.append('Baseline U-Net')
        elif 'no_temporal' in name.lower():
            exp_names.append('w/o Temporal')
        elif 'no_attention' in name.lower():
            exp_names.append('w/o Attention')
        elif 'no_separation' in name.lower():
            exp_names.append('w/o Separation')
        elif 'no_smooth' in name.lower():
            exp_names.append('w/o Smooth')
        elif 'no_reg' in name.lower():
            exp_names.append('w/o Regularization')
        elif 'old_weights' in name.lower():
            exp_names.append('Old Weights')
        else:
            exp_names.append(name)
    
    losses = df_sorted['Best_Val_Loss'].values
    
    # 颜色方案
    colors = []
    for name in exp_names:
        if 'Full' in name:
            colors.append('#2E86AB')  # 深蓝色 - 完整模型
        elif 'Baseline' in name:
            colors.append('#E94F37')  # 红色 - 基线
        else:
            colors.append('#A23B72')  # 紫色 - 消融变体
    
    # 绘制条形图
    y_pos = np.arange(len(exp_names))
    bars = ax.barh(y_pos, losses, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # 添加数值标签
    for i, (bar, loss) in enumerate(zip(bars, losses)):
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height()/2,
                f'{loss:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # 设置坐标轴
    ax.set_yticks(y_pos)
    ax.set_yticklabels(exp_names, fontsize=12)
    ax.set_xlabel('Validation Loss', fontsize=14, fontweight='bold')
    ax.set_title('Ablation Study: Validation Loss Comparison', fontsize=16, fontweight='bold', pad=15)
    
    # 添加网格
    ax.xaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    
    # 反转y轴使最好的在上面
    ax.invert_yaxis()
    
    # 加粗边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # 保存
    save_path = output_dir / 'fig_ablation_loss_comparison.pdf'
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    save_path_png = output_dir / 'fig_ablation_loss_comparison.png'
    plt.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存: {save_path}")
    print(f"✅ 保存: {save_path_png}")


# ============================================================================
# 图表2: 多指标雷达图/蜘蛛图
# ============================================================================
def plot_radar_chart(df: pd.DataFrame, output_dir: Path):
    """
    创建多指标雷达图对比
    """
    setup_paper_style()
    
    # 选择要对比的指标
    metrics = ['val_metrics_static_mse', 'val_metrics_dynamic_mse', 'val_metrics_total_mse']
    metric_labels = ['Static MSE', 'Dynamic MSE', 'Total MSE']
    
    # 检查是否有这些指标
    available_metrics = [m for m in metrics if m in df.columns]
    if len(available_metrics) < 2:
        print("⚠️  雷达图需要至少2个指标，跳过")
        return
    
    # 选择几个关键实验
    key_experiments = ['full', 'baseunet', 'no_temporal', 'no_attention']
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # 角度设置
    num_vars = len(available_metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    # 颜色
    colors = ['#2E86AB', '#E94F37', '#A23B72', '#F18F01']
    
    for idx, exp_key in enumerate(key_experiments):
        exp_data = df[df['Category'].str.contains(exp_key, case=False, na=False)]
        if exp_data.empty:
            continue
        
        row = exp_data.iloc[0]
        values = [row[m] if m in row and pd.notna(row[m]) else 0 for m in available_metrics]
        values += values[:1]  # 闭合
        
        # 归一化到0-1
        max_val = max(values[:-1]) if max(values[:-1]) > 0 else 1
        values_norm = [v / max_val for v in values]
        
        ax.plot(angles, values_norm, 'o-', linewidth=2, label=exp_key.title(), color=colors[idx % len(colors)])
        ax.fill(angles, values_norm, alpha=0.15, color=colors[idx % len(colors)])
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metric_labels[metrics.index(m)] for m in available_metrics], fontsize=12)
    
    ax.set_title('Multi-Metric Comparison (Radar Chart)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    save_path = output_dir / 'fig_radar_comparison.pdf'
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    save_path_png = output_dir / 'fig_radar_comparison.png'
    plt.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存: {save_path}")


# ============================================================================
# 图表3: NMSE对比柱状图（分组）
# ============================================================================
def plot_nmse_comparison(df: pd.DataFrame, output_dir: Path):
    """
    创建NMSE对比柱状图 - 分组显示Static/Dynamic/Total
    """
    setup_paper_style()
    
    # 检查是否有NMSE指标
    nmse_cols = ['val_metrics_static_nmse_db', 'val_metrics_dynamic_nmse_db', 'val_metrics_total_nmse_db']
    available_cols = [c for c in nmse_cols if c in df.columns]
    
    if not available_cols:
        print("⚠️  未找到NMSE指标，跳过")
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 过滤有效数据
    df_valid = df[df['val_metrics_total_nmse_db'].notna()].copy()
    if df_valid.empty:
        print("⚠️  没有有效的NMSE数据")
        return
    
    # 创建简短名称
    exp_names = []
    for _, row in df_valid.iterrows():
        name = row['Category']
        if 'full' in name.lower():
            exp_names.append('Full')
        elif 'baseunet' in name.lower():
            exp_names.append('Baseline')
        elif 'no_temporal' in name.lower():
            exp_names.append('w/o Temp')
        elif 'no_attention' in name.lower():
            exp_names.append('w/o Attn')
        elif 'no_separation' in name.lower():
            exp_names.append('w/o Sep')
        elif 'no_smooth' in name.lower():
            exp_names.append('w/o Smooth')
        elif 'no_reg' in name.lower():
            exp_names.append('w/o Reg')
        elif 'old_weights' in name.lower():
            exp_names.append('Old Wt')
        else:
            exp_names.append(name[:10])
    
    x = np.arange(len(exp_names))
    width = 0.25
    
    # 绘制分组柱状图
    colors = ['#3498DB', '#E74C3C', '#2ECC71']
    labels = ['Static NMSE', 'Dynamic NMSE', 'Total NMSE']
    
    for i, (col, color, label) in enumerate(zip(available_cols, colors, labels)):
        values = df_valid[col].values
        bars = ax.bar(x + (i - 1) * width, values, width, label=label, 
                     color=color, edgecolor='black', linewidth=1.2, alpha=0.85)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            if pd.notna(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 1,
                       f'{val:.1f}', ha='center', va='top', fontsize=8, 
                       fontweight='bold', color='white')
    
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('NMSE (dB)', fontsize=14, fontweight='bold')
    ax.set_title('Normalized Mean Square Error Comparison', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.9)
    
    # 添加基准线
    ax.axhline(y=-20, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Reference: -20 dB')
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    save_path = output_dir / 'fig_nmse_comparison.pdf'
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    save_path_png = output_dir / 'fig_nmse_comparison.png'
    plt.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存: {save_path}")


# ============================================================================
# 图表4: 消融实验热力图
# ============================================================================
def plot_ablation_heatmap(df: pd.DataFrame, output_dir: Path):
    """
    创建消融实验配置与性能的热力图
    """
    setup_paper_style()
    
    # 创建配置矩阵
    configs = {
        'Attention': df['Use_Attention'].astype(int) if 'Use_Attention' in df.columns else [1] * len(df),
        'Temporal': df['Temporal'].astype(int) if 'Temporal' in df.columns else [1] * len(df),
        'Separation': (df['Separation_Weight'] > 0).astype(int) if 'Separation_Weight' in df.columns else [1] * len(df),
    }
    
    # 创建简短名称
    exp_names = []
    for _, row in df.iterrows():
        name = row['Category']
        if 'full' in name.lower():
            exp_names.append('Full')
        elif 'baseunet' in name.lower():
            exp_names.append('Baseline')
        elif 'no_temporal' in name.lower():
            exp_names.append('w/o Temp')
        elif 'no_attention' in name.lower():
            exp_names.append('w/o Attn')
        elif 'no_separation' in name.lower():
            exp_names.append('w/o Sep')
        elif 'no_smooth' in name.lower():
            exp_names.append('w/o Smooth')
        elif 'no_reg' in name.lower():
            exp_names.append('w/o Reg')
        elif 'old_weights' in name.lower():
            exp_names.append('Old Wt')
        else:
            exp_names.append(name[:10])
    
    # 创建配置DataFrame
    config_df = pd.DataFrame(configs, index=exp_names)
    
    # 添加性能列
    config_df['Loss'] = df['Best_Val_Loss'].values
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
    
    # 左图：配置热力图
    ax1 = axes[0]
    config_only = config_df.drop('Loss', axis=1)
    sns.heatmap(config_only, annot=True, fmt='d', cmap='RdYlGn', 
                ax=ax1, cbar_kws={'label': 'Enabled (1) / Disabled (0)'},
                linewidths=1, linecolor='black')
    ax1.set_title('Component Configuration', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Experiment', fontsize=12, fontweight='bold')
    
    # 右图：Loss热力图
    ax2 = axes[1]
    loss_df = config_df[['Loss']]
    sns.heatmap(loss_df, annot=True, fmt='.4f', cmap='RdYlGn_r',
                ax=ax2, cbar_kws={'label': 'Validation Loss'},
                linewidths=1, linecolor='black')
    ax2.set_title('Performance', fontsize=14, fontweight='bold')
    ax2.set_ylabel('')
    
    plt.suptitle('Ablation Study: Configuration vs Performance', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = output_dir / 'fig_ablation_heatmap.pdf'
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    save_path_png = output_dir / 'fig_ablation_heatmap.png'
    plt.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存: {save_path}")


# ============================================================================
# 图表5: 训练曲线对比（如果有TensorBoard日志）
# ============================================================================
def plot_training_curves(base_dir: Path, output_dir: Path):
    """
    从TensorBoard日志中提取并绘制训练曲线
    """
    setup_paper_style()
    
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("⚠️  需要安装tensorboard来读取训练曲线")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_idx = 0
    
    # 遍历所有实验目录
    for category_dir in base_dir.iterdir():
        if not category_dir.is_dir():
            continue
        if category_dir.name in ['ablation_analysis', 'paper_figures']:
            continue
        
        for exp_dir in category_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            log_dir = exp_dir / 'logs'
            if not log_dir.exists():
                continue
            
            # 查找events文件
            event_files = list(log_dir.glob('events.out.tfevents.*'))
            if not event_files:
                continue
            
            try:
                ea = event_accumulator.EventAccumulator(str(log_dir))
                ea.Reload()
                
                # 获取标签
                exp_name = category_dir.name
                if 'full' in exp_name.lower():
                    label = 'Full Model'
                elif 'baseunet' in exp_name.lower():
                    label = 'Baseline'
                else:
                    label = exp_name.replace('no_', 'w/o ').title()
                
                # 训练Loss
                if 'train/loss' in ea.Tags()['scalars']:
                    train_loss = ea.Scalars('train/loss')
                    steps = [s.step for s in train_loss]
                    values = [s.value for s in train_loss]
                    axes[0].plot(steps, values, label=label, color=colors[color_idx], linewidth=1.5, alpha=0.8)
                
                # 验证Loss
                if 'val/loss' in ea.Tags()['scalars']:
                    val_loss = ea.Scalars('val/loss')
                    steps = [s.step for s in val_loss]
                    values = [s.value for s in val_loss]
                    axes[1].plot(steps, values, label=label, color=colors[color_idx], linewidth=1.5, alpha=0.8)
                
                color_idx = (color_idx + 1) % len(colors)
                
            except Exception as e:
                print(f"⚠️  读取日志失败 {exp_dir.name}: {e}")
                continue
    
    # 设置图表
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].set_yscale('log')
    
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    axes[1].set_title('Validation Loss Curves', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].set_yscale('log')
    
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_dir / 'fig_training_curves.pdf'
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    save_path_png = output_dir / 'fig_training_curves.png'
    plt.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存: {save_path}")


# ============================================================================
# 图表6: 综合对比表格（LaTeX格式）
# ============================================================================
def generate_latex_table(df: pd.DataFrame, output_dir: Path):
    """
    生成论文级别的LaTeX表格
    """
    # 创建简短名称
    exp_names = []
    for _, row in df.iterrows():
        name = row['Category']
        if 'full' in name.lower():
            exp_names.append('Full Model')
        elif 'baseunet' in name.lower():
            exp_names.append('Baseline U-Net')
        elif 'no_temporal' in name.lower():
            exp_names.append('w/o Temporal Constraint')
        elif 'no_attention' in name.lower():
            exp_names.append('w/o Attention Module')
        elif 'no_separation' in name.lower():
            exp_names.append('w/o Separation Loss')
        elif 'no_smooth' in name.lower():
            exp_names.append('w/o Static Smoothness')
        elif 'no_reg' in name.lower():
            exp_names.append('w/o Regularization')
        elif 'old_weights' in name.lower():
            exp_names.append('Original Weights')
        else:
            exp_names.append(name)
    
    # 创建表格数据
    table_data = []
    for i, (_, row) in enumerate(df.iterrows()):
        entry = {
            'Method': exp_names[i],
            'Val Loss': f"{row['Best_Val_Loss']:.4f}",
            'Best Epoch': int(row['Epoch']) if pd.notna(row['Epoch']) else '-',
        }
        
        # 添加NMSE指标（如果有）
        if 'val_metrics_static_nmse_db' in row and pd.notna(row['val_metrics_static_nmse_db']):
            entry['Static NMSE'] = f"{row['val_metrics_static_nmse_db']:.2f}"
        if 'val_metrics_dynamic_nmse_db' in row and pd.notna(row['val_metrics_dynamic_nmse_db']):
            entry['Dynamic NMSE'] = f"{row['val_metrics_dynamic_nmse_db']:.2f}"
        if 'val_metrics_total_nmse_db' in row and pd.notna(row['val_metrics_total_nmse_db']):
            entry['Total NMSE'] = f"{row['val_metrics_total_nmse_db']:.2f}"
        
        table_data.append(entry)
    
    table_df = pd.DataFrame(table_data)
    
    # 生成LaTeX代码
    latex_code = r"""
\begin{table}[htbp]
\centering
\caption{Ablation Study Results on Channel Decomposition Task}
\label{tab:ablation_results}
\begin{tabular}{l|c|c|ccc}
\toprule
\textbf{Method} & \textbf{Val Loss} & \textbf{Epoch} & \textbf{Static} & \textbf{Dynamic} & \textbf{Total} \\
& & & \textbf{NMSE (dB)} & \textbf{NMSE (dB)} & \textbf{NMSE (dB)} \\
\midrule
"""
    
    for entry in table_data:
        static = entry.get('Static NMSE', '-')
        dynamic = entry.get('Dynamic NMSE', '-')
        total = entry.get('Total NMSE', '-')
        
        # 高亮最佳结果
        method = entry['Method']
        if 'Full Model' in method:
            method = r'\textbf{' + method + '}'
        
        latex_code += f"{method} & {entry['Val Loss']} & {entry['Best Epoch']} & {static} & {dynamic} & {total} \\\\\n"
    
    latex_code += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    # 保存
    save_path = output_dir / 'table_ablation_results.tex'
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print(f"✅ 保存LaTeX表格: {save_path}")
    
    # 同时保存为CSV
    csv_path = output_dir / 'table_ablation_results.csv'
    table_df.to_csv(csv_path, index=False)
    print(f"✅ 保存CSV表格: {csv_path}")


# ============================================================================
# 图表7: 模型复杂度对比
# ============================================================================
def plot_complexity_comparison(df: pd.DataFrame, output_dir: Path):
    """
    创建模型复杂度与性能对比图
    """
    setup_paper_style()
    
    if 'model_total_params' not in df.columns:
        print("⚠️  未找到模型参数信息，跳过复杂度对比图")
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 创建简短名称
    exp_names = []
    for _, row in df.iterrows():
        name = row['Category']
        if 'full' in name.lower():
            exp_names.append('Full')
        elif 'baseunet' in name.lower():
            exp_names.append('Baseline')
        else:
            exp_names.append(name.replace('no_', 'w/o ').replace('_', ' ').title()[:12])
    
    # 数据
    params = df['model_total_params'].values / 1e6  # 转换为百万
    losses = df['Best_Val_Loss'].values
    
    # 颜色
    colors = []
    for name in exp_names:
        if 'Full' in name:
            colors.append('#2E86AB')
        elif 'Baseline' in name:
            colors.append('#E94F37')
        else:
            colors.append('#A23B72')
    
    # 散点图
    scatter = ax.scatter(params, losses, c=colors, s=200, edgecolors='black', 
                        linewidths=1.5, alpha=0.8, zorder=5)
    
    # 添加标签
    for i, (x, y, name) in enumerate(zip(params, losses, exp_names)):
        ax.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Parameters (M)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=14, fontweight='bold')
    ax.set_title('Model Complexity vs Performance', fontsize=16, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    save_path = output_dir / 'fig_complexity_comparison.pdf'
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    save_path_png = output_dir / 'fig_complexity_comparison.png'
    plt.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存: {save_path}")


# ============================================================================
# 图表8: 综合子图布局
# ============================================================================
def plot_comprehensive_figure(df: pd.DataFrame, output_dir: Path):
    """
    创建综合的多子图布局（适合论文的主图）
    """
    setup_paper_style()
    
    fig = plt.figure(figsize=(16, 12))
    
    # 创建网格布局
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 创建简短名称
    exp_names = []
    for _, row in df.iterrows():
        name = row['Category']
        if 'full' in name.lower():
            exp_names.append('Full')
        elif 'baseunet' in name.lower():
            exp_names.append('Baseline')
        elif 'no_temporal' in name.lower():
            exp_names.append('w/o Temp')
        elif 'no_attention' in name.lower():
            exp_names.append('w/o Attn')
        elif 'no_separation' in name.lower():
            exp_names.append('w/o Sep')
        elif 'no_smooth' in name.lower():
            exp_names.append('w/o Smooth')
        elif 'no_reg' in name.lower():
            exp_names.append('w/o Reg')
        elif 'old_weights' in name.lower():
            exp_names.append('Old Wt')
        else:
            exp_names.append(name[:8])
    
    # ============ 子图 (a): Loss对比条形图 ============
    ax1 = fig.add_subplot(gs[0, 0])
    df_sorted = df.sort_values('Best_Val_Loss')
    sorted_names = [exp_names[i] for i in df_sorted.index]
    losses = df_sorted['Best_Val_Loss'].values
    
    colors = ['#2E86AB' if 'Full' in n else '#E94F37' if 'Baseline' in n else '#A23B72' 
              for n in sorted_names]
    
    bars = ax1.barh(range(len(sorted_names)), losses, color=colors, 
                   edgecolor='black', linewidth=1.2)
    ax1.set_yticks(range(len(sorted_names)))
    ax1.set_yticklabels(sorted_names, fontsize=10)
    ax1.set_xlabel('Validation Loss', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Loss Comparison', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    ax1.xaxis.grid(True, linestyle='--', alpha=0.5)
    
    # ============ 子图 (b): NMSE对比 ============
    ax2 = fig.add_subplot(gs[0, 1:])
    
    nmse_cols = ['val_metrics_static_nmse_db', 'val_metrics_dynamic_nmse_db', 'val_metrics_total_nmse_db']
    available_cols = [c for c in nmse_cols if c in df.columns]
    
    if available_cols:
        x = np.arange(len(exp_names))
        width = 0.25
        colors_nmse = ['#3498DB', '#E74C3C', '#2ECC71']
        labels_nmse = ['Static', 'Dynamic', 'Total']
        
        for i, (col, color, label) in enumerate(zip(available_cols, colors_nmse, labels_nmse)):
            values = df[col].fillna(0).values
            ax2.bar(x + (i - 1) * width, values, width, label=label, 
                   color=color, edgecolor='black', linewidth=0.8, alpha=0.85)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=10)
        ax2.set_ylabel('NMSE (dB)', fontsize=11, fontweight='bold')
        ax2.set_title('(b) NMSE Comparison by Component', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
    
    # ============ 子图 (c): 训练epoch对比 ============
    ax3 = fig.add_subplot(gs[1, 0])
    epochs = df['Epoch'].values
    bars = ax3.bar(range(len(exp_names)), epochs, color='#9B59B6', 
                  edgecolor='black', linewidth=1.2, alpha=0.8)
    ax3.set_xticks(range(len(exp_names)))
    ax3.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=10)
    ax3.set_ylabel('Best Epoch', fontsize=11, fontweight='bold')
    ax3.set_title('(c) Convergence Speed', fontsize=13, fontweight='bold')
    ax3.yaxis.grid(True, linestyle='--', alpha=0.5)
    
    # ============ 子图 (d): 配置热力图 ============
    ax4 = fig.add_subplot(gs[1, 1])
    
    config_data = {
        'Attn': df['Use_Attention'].astype(int).values if 'Use_Attention' in df.columns else np.ones(len(df)),
        'Temp': df['Temporal'].astype(int).values if 'Temporal' in df.columns else np.ones(len(df)),
        'Sep': (df['Separation_Weight'] > 0).astype(int).values if 'Separation_Weight' in df.columns else np.ones(len(df)),
    }
    config_df = pd.DataFrame(config_data, index=exp_names)
    
    sns.heatmap(config_df, annot=True, fmt='d', cmap='RdYlGn', ax=ax4,
               cbar=False, linewidths=1, linecolor='black')
    ax4.set_title('(d) Component Configuration', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Experiment', fontsize=11)
    
    # ============ 子图 (e): 改进百分比 ============
    ax5 = fig.add_subplot(gs[1, 2])
    
    # 计算相对于baseline的改进
    baseline_loss = df[df['Category'].str.contains('baseunet', case=False, na=False)]['Best_Val_Loss'].values
    if len(baseline_loss) > 0:
        baseline_loss = baseline_loss[0]
        improvements = [(baseline_loss - loss) / baseline_loss * 100 for loss in df['Best_Val_Loss'].values]
        
        colors_imp = ['#27AE60' if imp > 0 else '#E74C3C' for imp in improvements]
        bars = ax5.barh(range(len(exp_names)), improvements, color=colors_imp,
                       edgecolor='black', linewidth=1.2, alpha=0.8)
        ax5.set_yticks(range(len(exp_names)))
        ax5.set_yticklabels(exp_names, fontsize=10)
        ax5.set_xlabel('Improvement over Baseline (%)', fontsize=11, fontweight='bold')
        ax5.set_title('(e) Relative Performance', fontsize=13, fontweight='bold')
        ax5.axvline(x=0, color='black', linewidth=1.5)
        ax5.xaxis.grid(True, linestyle='--', alpha=0.5)
    
    # 加粗所有子图边框
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
    
    plt.suptitle('Ablation Study: Comprehensive Analysis', fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    save_path = output_dir / 'fig_comprehensive_ablation.pdf'
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    save_path_png = output_dir / 'fig_comprehensive_ablation.png'
    plt.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存综合图: {save_path}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='生成论文级别的实验图表')
    parser.add_argument('--base_dir', type=str, 
                       default='/LSEM/user/chenyinda/code/signal_dy_static/1104/results_20251104_092511',
                       help='实验结果基础目录')
    parser.add_argument('--output_dir', type=str, 
                       default='/LSEM/user/chenyinda/code/signal_dy_static/1104/results_20251104_092511/paper_figures',
                       help='图表输出目录')
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("📊 论文级别图表生成器")
    print("="*80)
    print(f"📁 实验目录: {base_dir}")
    print(f"📁 输出目录: {output_dir}")
    print("="*80 + "\n")
    
    # 加载数据
    df = load_ablation_results(str(base_dir))
    if df is None or df.empty:
        print("❌ 无法加载实验结果")
        return
    
    print(f"✅ 加载了 {len(df)} 个实验结果\n")
    
    # 生成各种图表
    print("📈 生成图表...\n")
    
    # 1. Loss对比条形图
    print("1️⃣  生成Loss对比条形图...")
    plot_ablation_loss_comparison(df, output_dir)
    
    # 2. NMSE对比图
    print("2️⃣  生成NMSE对比图...")
    plot_nmse_comparison(df, output_dir)
    
    # 3. 热力图
    print("3️⃣  生成配置热力图...")
    plot_ablation_heatmap(df, output_dir)
    
    # 4. 复杂度对比
    print("4️⃣  生成复杂度对比图...")
    plot_complexity_comparison(df, output_dir)
    
    # 5. 综合图
    print("5️⃣  生成综合对比图...")
    plot_comprehensive_figure(df, output_dir)
    
    # 6. LaTeX表格
    print("6️⃣  生成LaTeX表格...")
    generate_latex_table(df, output_dir)
    
    # 7. 训练曲线（如果有日志）
    print("7️⃣  尝试生成训练曲线...")
    plot_training_curves(base_dir, output_dir)
    
    print("\n" + "="*80)
    print("✅ 所有图表生成完成！")
    print("="*80)
    print(f"\n📁 图表保存位置: {output_dir}")
    print("\n生成的文件:")
    for f in sorted(output_dir.glob('*')):
        print(f"   - {f.name}")
    print()


if __name__ == '__main__':
    main()

