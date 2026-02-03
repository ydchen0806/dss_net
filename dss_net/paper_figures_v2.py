#!/usr/bin/env python
"""
论文级别的高质量可视化脚本 V2
- 更炫酷的配色方案
- 更清晰的图像质量
- Times New Roman字体
- 加粗边框
- 专业的子图布局
- 不包含loss图
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
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from tqdm import tqdm
import argparse
import warnings
import seaborn as sns
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches

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
    rcParams['axes.titlesize'] = 14
    rcParams['axes.labelsize'] = 12
    rcParams['xtick.labelsize'] = 11
    rcParams['ytick.labelsize'] = 11
    rcParams['legend.fontsize'] = 10
    rcParams['figure.titlesize'] = 16
    
    # 加粗边框
    rcParams['axes.linewidth'] = 2.0
    rcParams['xtick.major.width'] = 1.5
    rcParams['ytick.major.width'] = 1.5
    rcParams['xtick.minor.width'] = 1.0
    rcParams['ytick.minor.width'] = 1.0
    
    # 高质量输出
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.pad_inches'] = 0.1
    
    # LaTeX风格
    rcParams['text.usetex'] = False
    rcParams['mathtext.fontset'] = 'stix'
    
    # 网格样式
    rcParams['grid.alpha'] = 0.3
    rcParams['grid.linewidth'] = 0.8

# 炫酷配色方案
COLORS = {
    'primary': '#1E3A5F',      # 深蓝色
    'secondary': '#4A90A4',    # 青蓝色
    'accent': '#E74C3C',       # 红色强调
    'success': '#27AE60',      # 绿色
    'warning': '#F39C12',      # 橙色
    'purple': '#8E44AD',       # 紫色
    'dark': '#2C3E50',         # 深灰
    'light': '#ECF0F1',        # 浅灰
}

# 渐变配色
GRADIENT_COLORS = [
    '#1a237e',  # 深蓝
    '#303f9f',  # 靛蓝
    '#3f51b5',  # 蓝色
    '#7986cb',  # 浅蓝
    '#c5cae9',  # 淡蓝
]

# 对比配色
CONTRAST_COLORS = [
    '#E74C3C',  # 红
    '#3498DB',  # 蓝
    '#2ECC71',  # 绿
    '#F39C12',  # 橙
    '#9B59B6',  # 紫
    '#1ABC9C',  # 青
    '#34495E',  # 灰
]

# 模型名称映射
MODEL_NAMES = {
    'baseunet': 'Baseline U-Net',
    'full': 'DSS-Net (Full)',
    'no_attention': 'w/o Attention',
    'no_temporal': 'w/o Temporal',
    'no_separation': 'w/o Separation',
    'old_weights': 'w/o Optimized Weights',
    'no_reg': 'w/o Regularization',
    'no_smooth': 'w/o Static Smooth',
}

# ============================================================================
# 图表生成函数
# ============================================================================

def create_ablation_bar_chart(results_df, output_dir):
    """创建消融实验柱状图 - 更炫酷版本"""
    setup_paper_style()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 准备数据
    categories = results_df['Category'].tolist()
    nmse_values = results_df['Test Total NMSE (dB)'].tolist()
    
    # 映射名称
    display_names = [MODEL_NAMES.get(cat, cat) for cat in categories]
    
    # 排序
    sorted_indices = np.argsort(nmse_values)
    display_names = [display_names[i] for i in sorted_indices]
    nmse_values = [nmse_values[i] for i in sorted_indices]
    
    # 创建渐变色条
    n_bars = len(nmse_values)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, n_bars))
    
    # 绘制柱状图
    bars = ax.barh(range(n_bars), nmse_values, color=colors, edgecolor='black', linewidth=1.5, height=0.7)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, nmse_values)):
        ax.text(val - 0.3, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f} dB', va='center', ha='right', 
                fontsize=11, fontweight='bold', color='white')
    
    # 标记最佳结果
    best_idx = np.argmin(nmse_values)
    bars[best_idx].set_edgecolor('#E74C3C')
    bars[best_idx].set_linewidth(3)
    
    ax.set_yticks(range(n_bars))
    ax.set_yticklabels(display_names, fontsize=11)
    ax.set_xlabel('Test NMSE (dB)', fontsize=13, fontweight='bold')
    ax.set_title('Ablation Study: NMSE Performance Comparison', fontsize=14, fontweight='bold', pad=15)
    
    # 添加网格
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    # 添加图例说明
    ax.axvline(x=nmse_values[best_idx], color='#E74C3C', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(nmse_values[best_idx] + 0.1, n_bars - 0.5, 'Best', color='#E74C3C', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存
    for fmt in ['pdf', 'png', 'eps']:
        fig.savefig(os.path.join(output_dir, f'fig_ablation_nmse.{fmt}'), 
                    format=fmt, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: fig_ablation_nmse.[pdf/png/eps]")


def create_component_comparison(results_df, output_dir):
    """创建组件对比图 - 静态/动态分离"""
    setup_paper_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    categories = results_df['Category'].tolist()
    static_nmse = results_df['Test Static NMSE (dB)'].tolist()
    dynamic_nmse = results_df['Test Dynamic NMSE (dB)'].tolist()
    
    display_names = [MODEL_NAMES.get(cat, cat) for cat in categories]
    
    # 按总NMSE排序
    total_nmse = results_df['Test Total NMSE (dB)'].tolist()
    sorted_indices = np.argsort(total_nmse)
    
    x = np.arange(len(categories))
    width = 0.35
    
    # 静态分量
    ax1 = axes[0]
    static_sorted = [static_nmse[i] for i in sorted_indices]
    names_sorted = [display_names[i] for i in sorted_indices]
    
    bars1 = ax1.bar(x, static_sorted, width, color='#3498DB', edgecolor='black', linewidth=1.5, label='Static Component')
    ax1.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Static NMSE (dB)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Static Component Performance', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names_sorted, rotation=45, ha='right', fontsize=9)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.5)
    
    # 添加数值标签
    for bar, val in zip(bars1, static_sorted):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 动态分量
    ax2 = axes[1]
    dynamic_sorted = [dynamic_nmse[i] for i in sorted_indices]
    
    bars2 = ax2.bar(x, dynamic_sorted, width, color='#E74C3C', edgecolor='black', linewidth=1.5, label='Dynamic Component')
    ax2.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Dynamic NMSE (dB)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Dynamic Component Performance', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names_sorted, rotation=45, ha='right', fontsize=9)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
    
    # 添加数值标签
    for bar, val in zip(bars2, dynamic_sorted):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png', 'eps']:
        fig.savefig(os.path.join(output_dir, f'fig_component_comparison.{fmt}'), 
                    format=fmt, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: fig_component_comparison.[pdf/png/eps]")


def create_radar_chart(results_df, output_dir):
    """创建雷达图 - 多维度对比"""
    setup_paper_style()
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # 选择关键配置进行对比
    key_configs = ['baseunet', 'full', 'no_attention', 'no_temporal', 'no_separation']
    
    # 指标
    metrics = ['Total NMSE', 'Static NMSE', 'Dynamic NMSE', 'PSNR', 'Val Loss']
    n_metrics = len(metrics)
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    
    for idx, config in enumerate(key_configs):
        if config == 'baseunet':
            # 基线UNet可能不在结果中，跳过或使用默认值
            continue
        
        row = results_df[results_df['Category'] == config]
        if len(row) == 0:
            continue
        
        row = row.iloc[0]
        
        # 归一化数值 (越小越好的取负值)
        values = [
            -row['Test Total NMSE (dB)'] / 30,  # 归一化
            -row['Test Static NMSE (dB)'] / 30,
            -row['Test Dynamic NMSE (dB)'] / 20,
            row['Test Total PSNR (dB)'] / 30,
            1 - row['Val Loss'] / 0.2,  # 归一化
        ]
        values += values[:1]  # 闭合
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label=MODEL_NAMES.get(config, config), 
                color=colors[idx % len(colors)], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_title('Multi-Metric Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png', 'eps']:
        fig.savefig(os.path.join(output_dir, f'fig_radar_comparison.{fmt}'), 
                    format=fmt, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: fig_radar_comparison.[pdf/png/eps]")


def create_comprehensive_figure(results_df, output_dir):
    """创建综合多子图 - 论文主图"""
    setup_paper_style()
    
    fig = plt.figure(figsize=(16, 12))
    
    # 创建网格布局
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    categories = results_df['Category'].tolist()
    total_nmse = results_df['Test Total NMSE (dB)'].tolist()
    static_nmse = results_df['Test Static NMSE (dB)'].tolist()
    dynamic_nmse = results_df['Test Dynamic NMSE (dB)'].tolist()
    psnr = results_df['Test Total PSNR (dB)'].tolist()
    
    display_names = [MODEL_NAMES.get(cat, cat) for cat in categories]
    
    # 按总NMSE排序
    sorted_indices = np.argsort(total_nmse)
    
    # ===== 子图 (a): 总NMSE对比 =====
    ax1 = fig.add_subplot(gs[0, 0])
    nmse_sorted = [total_nmse[i] for i in sorted_indices]
    names_sorted = [display_names[i] for i in sorted_indices]
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(nmse_sorted)))
    bars = ax1.barh(range(len(nmse_sorted)), nmse_sorted, color=colors, edgecolor='black', linewidth=1.5)
    
    # 标记最佳
    best_idx = 0  # 已排序，第一个最好
    bars[best_idx].set_edgecolor('#E74C3C')
    bars[best_idx].set_linewidth(3)
    
    ax1.set_yticks(range(len(nmse_sorted)))
    ax1.set_yticklabels(names_sorted, fontsize=9)
    ax1.set_xlabel('NMSE (dB)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Total NMSE Comparison', fontsize=12, fontweight='bold')
    ax1.xaxis.grid(True, linestyle='--', alpha=0.5)
    
    # ===== 子图 (b): 静态/动态分量对比 =====
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(categories))
    width = 0.35
    
    static_sorted = [static_nmse[i] for i in sorted_indices]
    dynamic_sorted = [dynamic_nmse[i] for i in sorted_indices]
    
    bars1 = ax2.bar(x - width/2, static_sorted, width, label='Static', color='#3498DB', edgecolor='black', linewidth=1)
    bars2 = ax2.bar(x + width/2, dynamic_sorted, width, label='Dynamic', color='#E74C3C', edgecolor='black', linewidth=1)
    
    ax2.set_xlabel('Configuration', fontsize=11, fontweight='bold')
    ax2.set_ylabel('NMSE (dB)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Static vs Dynamic NMSE', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([n[:8] + '...' if len(n) > 10 else n for n in names_sorted], rotation=45, ha='right', fontsize=8)
    ax2.legend(loc='upper right')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
    
    # ===== 子图 (c): PSNR对比 =====
    ax3 = fig.add_subplot(gs[0, 2])
    psnr_sorted = [psnr[i] for i in sorted_indices]
    
    colors_psnr = plt.cm.YlGn(np.linspace(0.3, 0.9, len(psnr_sorted)))
    bars = ax3.barh(range(len(psnr_sorted)), psnr_sorted, color=colors_psnr, edgecolor='black', linewidth=1.5)
    
    ax3.set_yticks(range(len(psnr_sorted)))
    ax3.set_yticklabels(names_sorted, fontsize=9)
    ax3.set_xlabel('PSNR (dB)', fontsize=11, fontweight='bold')
    ax3.set_title('(c) PSNR Comparison', fontsize=12, fontweight='bold')
    ax3.xaxis.grid(True, linestyle='--', alpha=0.5)
    
    # ===== 子图 (d): 热力图 =====
    ax4 = fig.add_subplot(gs[1, 0:2])
    
    # 创建热力图数据
    heatmap_data = np.array([
        [total_nmse[i] for i in sorted_indices],
        [static_nmse[i] for i in sorted_indices],
        [dynamic_nmse[i] for i in sorted_indices],
    ])
    
    im = ax4.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
    
    ax4.set_xticks(range(len(names_sorted)))
    ax4.set_xticklabels(names_sorted, rotation=45, ha='right', fontsize=9)
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(['Total', 'Static', 'Dynamic'], fontsize=10)
    ax4.set_title('(d) Performance Heatmap (NMSE dB)', fontsize=12, fontweight='bold')
    
    # 添加数值标注
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            text = ax4.text(j, i, f'{heatmap_data[i, j]:.1f}', ha='center', va='center', 
                           color='white' if heatmap_data[i, j] < -22 else 'black', fontsize=9, fontweight='bold')
    
    # 添加colorbar
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('NMSE (dB)', fontsize=10)
    
    # ===== 子图 (e): 改进量柱状图 =====
    ax5 = fig.add_subplot(gs[1, 2])
    
    # 计算相对于最差配置的改进
    baseline_nmse = max(total_nmse)  # 最差作为基准
    improvements = [baseline_nmse - nmse for nmse in total_nmse]
    improvements_sorted = [improvements[i] for i in sorted_indices]
    
    colors_imp = ['#2ECC71' if imp > 0 else '#E74C3C' for imp in improvements_sorted]
    bars = ax5.barh(range(len(improvements_sorted)), improvements_sorted, color=colors_imp, edgecolor='black', linewidth=1.5)
    
    ax5.set_yticks(range(len(improvements_sorted)))
    ax5.set_yticklabels(names_sorted, fontsize=9)
    ax5.set_xlabel('Improvement (dB)', fontsize=11, fontweight='bold')
    ax5.set_title('(e) Improvement over Baseline', fontsize=12, fontweight='bold')
    ax5.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax5.axvline(x=0, color='black', linewidth=1.5, linestyle='-')
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png', 'eps']:
        fig.savefig(os.path.join(output_dir, f'fig_comprehensive_ablation_v2.{fmt}'), 
                    format=fmt, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: fig_comprehensive_ablation_v2.[pdf/png/eps]")


def create_ablation_table_figure(results_df, output_dir):
    """创建消融实验表格图"""
    setup_paper_style()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    # 准备表格数据
    sorted_df = results_df.sort_values('Test Total NMSE (dB)')
    
    headers = ['Configuration', 'Total NMSE (dB)', 'Static NMSE (dB)', 'Dynamic NMSE (dB)', 'PSNR (dB)', 'Gain (dB)']
    
    # 计算增益
    baseline_nmse = sorted_df['Test Total NMSE (dB)'].max()
    
    table_data = []
    for _, row in sorted_df.iterrows():
        config_name = MODEL_NAMES.get(row['Category'], row['Category'])
        gain = baseline_nmse - row['Test Total NMSE (dB)']
        table_data.append([
            config_name,
            f"{row['Test Total NMSE (dB)']:.2f}",
            f"{row['Test Static NMSE (dB)']:.2f}",
            f"{row['Test Dynamic NMSE (dB)']:.2f}",
            f"{row['Test Total PSNR (dB)']:.2f}",
            f"+{gain:.2f}" if gain > 0 else f"{gain:.2f}"
        ])
    
    # 创建表格
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    
    # 设置表头样式
    for j, header in enumerate(headers):
        cell = table[(0, j)]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold')
    
    # 设置最佳行样式
    for j in range(len(headers)):
        cell = table[(1, j)]  # 第一行数据（已排序，最佳）
        cell.set_facecolor('#D5F5E3')
    
    ax.set_title('Ablation Study Results Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        fig.savefig(os.path.join(output_dir, f'fig_ablation_table.{fmt}'), 
                    format=fmt, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: fig_ablation_table.[pdf/png]")


def create_improvement_waterfall(results_df, output_dir):
    """创建改进瀑布图"""
    setup_paper_style()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 按照消融顺序排列
    ablation_order = ['old_weights', 'no_attention', 'no_separation', 'no_temporal', 'no_smooth', 'no_reg', 'full']
    
    ordered_data = []
    for cat in ablation_order:
        row = results_df[results_df['Category'] == cat]
        if len(row) > 0:
            ordered_data.append({
                'name': MODEL_NAMES.get(cat, cat),
                'nmse': row.iloc[0]['Test Total NMSE (dB)']
            })
    
    if len(ordered_data) < 2:
        print("Not enough data for waterfall chart")
        return
    
    names = [d['name'] for d in ordered_data]
    nmse_values = [d['nmse'] for d in ordered_data]
    
    # 计算累积改进
    baseline = nmse_values[0]
    improvements = [0]
    for i in range(1, len(nmse_values)):
        improvements.append(baseline - nmse_values[i])
    
    # 绘制瀑布图
    colors = ['#E74C3C' if imp < 0 else '#2ECC71' for imp in improvements]
    colors[0] = '#3498DB'  # 基准
    
    bars = ax.bar(range(len(names)), improvements, color=colors, edgecolor='black', linewidth=1.5)
    
    # 添加连接线
    for i in range(len(names) - 1):
        ax.plot([i + 0.4, i + 0.6], [improvements[i], improvements[i+1]], 'k--', linewidth=1, alpha=0.5)
    
    # 添加数值标签
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.1, 
                f'{imp:+.2f} dB', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Improvement over Baseline (dB)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Cumulative Improvement Analysis', fontsize=14, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='black', linewidth=1.5)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png', 'eps']:
        fig.savefig(os.path.join(output_dir, f'fig_improvement_waterfall.{fmt}'), 
                    format=fmt, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: fig_improvement_waterfall.[pdf/png/eps]")


def generate_latex_table(results_df, output_dir):
    """生成LaTeX表格"""
    sorted_df = results_df.sort_values('Test Total NMSE (dB)')
    
    baseline_nmse = sorted_df['Test Total NMSE (dB)'].max()
    
    latex_content = r"""\begin{table}[htbp]
\centering
\caption{Ablation Study Results on Test Set}
\label{tab:ablation_results}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Configuration} & \textbf{Total NMSE (dB)} & \textbf{Static NMSE (dB)} & \textbf{Dynamic NMSE (dB)} & \textbf{Gain (dB)} \\
\midrule
"""
    
    for idx, (_, row) in enumerate(sorted_df.iterrows()):
        config_name = MODEL_NAMES.get(row['Category'], row['Category'])
        gain = baseline_nmse - row['Test Total NMSE (dB)']
        
        # 最佳结果加粗
        if idx == 0:
            latex_content += f"\\textbf{{{config_name}}} & \\textbf{{{row['Test Total NMSE (dB)']:.2f}}} & \\textbf{{{row['Test Static NMSE (dB)']:.2f}}} & \\textbf{{{row['Test Dynamic NMSE (dB)']:.2f}}} & \\textbf{{+{gain:.2f}}} \\\\\n"
        else:
            latex_content += f"{config_name} & {row['Test Total NMSE (dB)']:.2f} & {row['Test Static NMSE (dB)']:.2f} & {row['Test Dynamic NMSE (dB)']:.2f} & +{gain:.2f} \\\\\n"
    
    latex_content += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(os.path.join(output_dir, 'table_ablation_results_v2.tex'), 'w') as f:
        f.write(latex_content)
    
    print(f"Saved: table_ablation_results_v2.tex")


def main():
    parser = argparse.ArgumentParser(description='Generate paper-quality figures V2')
    parser.add_argument('--base_dir', type=str, required=True, help='Base results directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for figures')
    args = parser.parse_args()
    
    base_dir = args.base_dir
    output_dir = args.output_dir or os.path.join(base_dir, 'paper_figures_v2')
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取测试评估结果
    eval_summary_path = os.path.join(base_dir, 'paper_eval', 'test_evaluation_summary.csv')
    if not os.path.exists(eval_summary_path):
        print(f"Error: {eval_summary_path} not found. Please run evaluation first.")
        return
    
    results_df = pd.read_csv(eval_summary_path)
    print(f"Loaded {len(results_df)} experiment results")
    print(results_df)
    
    # 生成所有图表
    print("\n" + "="*60)
    print("Generating paper-quality figures V2...")
    print("="*60)
    
    create_ablation_bar_chart(results_df, output_dir)
    create_component_comparison(results_df, output_dir)
    create_comprehensive_figure(results_df, output_dir)
    create_ablation_table_figure(results_df, output_dir)
    create_improvement_waterfall(results_df, output_dir)
    generate_latex_table(results_df, output_dir)
    
    print("\n" + "="*60)
    print(f"All figures saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

