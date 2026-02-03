#!/usr/bin/env python
"""
高质量论文图表生成脚本
- PDF矢量图格式
- 加粗边框
- 大字体
- Times New Roman字体
- 符合IEEE顶刊要求
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 全局样式设置 - 顶刊级别
# ============================================================================
def setup_ieee_style():
    """设置IEEE顶刊级别的matplotlib样式"""
    # 字体设置
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    rcParams['font.size'] = 14
    rcParams['axes.titlesize'] = 16
    rcParams['axes.labelsize'] = 14
    rcParams['xtick.labelsize'] = 12
    rcParams['ytick.labelsize'] = 12
    rcParams['legend.fontsize'] = 12
    rcParams['figure.titlesize'] = 18
    
    # 加粗边框
    rcParams['axes.linewidth'] = 2.5
    rcParams['xtick.major.width'] = 2.0
    rcParams['ytick.major.width'] = 2.0
    rcParams['xtick.minor.width'] = 1.5
    rcParams['ytick.minor.width'] = 1.5
    rcParams['xtick.major.size'] = 6
    rcParams['ytick.major.size'] = 6
    
    # 高质量输出
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.pad_inches'] = 0.1
    
    # 数学字体
    rcParams['mathtext.fontset'] = 'stix'
    
    # 网格样式
    rcParams['grid.alpha'] = 0.4
    rcParams['grid.linewidth'] = 1.0
    rcParams['grid.linestyle'] = '--'

# 模型名称映射
MODEL_NAMES = {
    'no_reg': 'DSS-Net (Full)',
    'no_temporal': 'w/o Temporal',
    'no_smooth': 'w/o Static Smooth',
    'full': 'w/o Physics Loss',
    'no_separation': 'w/o Separation',
    'no_attention': 'w/o Attention',
    'old_weights': 'Baseline (Equal Weights)',
}

# 专业配色方案
COLORS = {
    'best': '#1E88E5',      # 蓝色 - 最佳
    'good': '#43A047',      # 绿色 - 好
    'medium': '#FB8C00',    # 橙色 - 中等
    'poor': '#E53935',      # 红色 - 差
    'static': '#1565C0',    # 深蓝 - 静态
    'dynamic': '#C62828',   # 深红 - 动态
}


def create_ablation_nmse_figure(results_df, output_dir):
    """创建消融实验NMSE对比图"""
    setup_ieee_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 准备数据并排序
    categories = results_df['Category'].tolist()
    nmse_values = results_df['Test Total NMSE (dB)'].tolist()
    display_names = [MODEL_NAMES.get(cat, cat) for cat in categories]
    
    # 按NMSE排序（从好到差）
    sorted_indices = np.argsort(nmse_values)
    display_names = [display_names[i] for i in sorted_indices]
    nmse_values = [nmse_values[i] for i in sorted_indices]
    
    n_bars = len(nmse_values)
    y_pos = np.arange(n_bars)
    
    # 使用渐变色
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, n_bars))
    
    # 绘制水平柱状图
    bars = ax.barh(y_pos, nmse_values, height=0.65, color=colors, 
                   edgecolor='black', linewidth=2.0)
    
    # 标记最佳结果
    bars[0].set_edgecolor('#1565C0')
    bars[0].set_linewidth(3.5)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, nmse_values)):
        # 在柱子内部显示数值
        ax.text(val + 0.15, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f} dB', va='center', ha='left', 
                fontsize=12, fontweight='bold', color='black')
    
    # 设置坐标轴
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=13, fontweight='bold')
    ax.set_xlabel('Total NMSE (dB)', fontsize=14, fontweight='bold')
    ax.set_title('Ablation Study: NMSE Performance Comparison', fontsize=16, fontweight='bold', pad=15)
    
    # 添加网格
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, linewidth=1.0)
    ax.set_axisbelow(True)
    
    # 添加最佳标记线
    ax.axvline(x=nmse_values[0], color='#1565C0', linestyle='--', linewidth=2.0, alpha=0.7)
    
    # 调整边距
    ax.set_xlim(min(nmse_values) - 1, max(nmse_values) + 2)
    
    plt.tight_layout()
    
    # 保存为PDF和EPS
    fig.savefig(os.path.join(output_dir, 'fig_ablation_nmse.pdf'), format='pdf', bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig_ablation_nmse.eps'), format='eps', bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig_ablation_nmse.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✓ Saved: fig_ablation_nmse.[pdf/eps/png]")


def create_component_comparison_figure(results_df, output_dir):
    """创建静态/动态分量对比图"""
    setup_ieee_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 准备数据
    categories = results_df['Category'].tolist()
    static_nmse = results_df['Test Static NMSE (dB)'].tolist()
    dynamic_nmse = results_df['Test Dynamic NMSE (dB)'].tolist()
    total_nmse = results_df['Test Total NMSE (dB)'].tolist()
    display_names = [MODEL_NAMES.get(cat, cat) for cat in categories]
    
    # 按总NMSE排序
    sorted_indices = np.argsort(total_nmse)
    display_names = [display_names[i] for i in sorted_indices]
    static_sorted = [static_nmse[i] for i in sorted_indices]
    dynamic_sorted = [dynamic_nmse[i] for i in sorted_indices]
    
    x = np.arange(len(display_names))
    width = 0.6
    
    # ===== 静态分量 =====
    colors_static = plt.cm.Blues(np.linspace(0.4, 0.9, len(static_sorted)))
    bars1 = ax1.bar(x, static_sorted, width, color=colors_static, 
                    edgecolor='black', linewidth=2.0)
    
    # 标记最佳
    bars1[0].set_edgecolor('#1565C0')
    bars1[0].set_linewidth(3.5)
    
    ax1.set_xlabel('Model Configuration', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Static NMSE (dB)', fontsize=14, fontweight='bold')
    ax1.set_title('(a) Static Component', fontsize=15, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_names, rotation=40, ha='right', fontsize=11, fontweight='bold')
    ax1.yaxis.grid(True, linestyle='--', alpha=0.5, linewidth=1.0)
    ax1.set_axisbelow(True)
    
    # 添加数值标签
    for bar, val in zip(bars1, static_sorted):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.8, 
                f'{val:.1f}', ha='center', va='top', fontsize=10, fontweight='bold', color='white')
    
    # ===== 动态分量 =====
    colors_dynamic = plt.cm.Reds(np.linspace(0.4, 0.9, len(dynamic_sorted)))
    bars2 = ax2.bar(x, dynamic_sorted, width, color=colors_dynamic, 
                    edgecolor='black', linewidth=2.0)
    
    # 标记最佳
    bars2[0].set_edgecolor('#C62828')
    bars2[0].set_linewidth(3.5)
    
    ax2.set_xlabel('Model Configuration', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Dynamic NMSE (dB)', fontsize=14, fontweight='bold')
    ax2.set_title('(b) Dynamic Component', fontsize=15, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_names, rotation=40, ha='right', fontsize=11, fontweight='bold')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5, linewidth=1.0)
    ax2.set_axisbelow(True)
    
    # 添加数值标签
    for bar, val in zip(bars2, dynamic_sorted):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.5, 
                f'{val:.1f}', ha='center', va='top', fontsize=10, fontweight='bold', color='white')
    
    plt.tight_layout()
    
    # 保存
    fig.savefig(os.path.join(output_dir, 'fig_component_comparison.pdf'), format='pdf', bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig_component_comparison.eps'), format='eps', bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig_component_comparison.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✓ Saved: fig_component_comparison.[pdf/eps/png]")


def create_improvement_waterfall_figure(results_df, output_dir):
    """创建改进瀑布图"""
    setup_ieee_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 按照消融顺序排列（从基准到最佳）
    ablation_order = ['old_weights', 'no_attention', 'no_separation', 'full', 'no_smooth', 'no_temporal', 'no_reg']
    
    ordered_data = []
    for cat in ablation_order:
        row = results_df[results_df['Category'] == cat]
        if len(row) > 0:
            ordered_data.append({
                'name': MODEL_NAMES.get(cat, cat),
                'nmse': row.iloc[0]['Test Total NMSE (dB)'],
                'category': cat
            })
    
    if len(ordered_data) < 2:
        print("Not enough data for waterfall chart")
        return
    
    names = [d['name'] for d in ordered_data]
    nmse_values = [d['nmse'] for d in ordered_data]
    
    # 计算相对于基准的改进
    baseline = nmse_values[0]
    improvements = [baseline - nmse for nmse in nmse_values]
    
    x = np.arange(len(names))
    
    # 颜色：基准灰色，改进绿色，最佳蓝色
    colors = []
    for i, imp in enumerate(improvements):
        if i == 0:
            colors.append('#607D8B')  # 基准灰色
        elif i == len(improvements) - 1:
            colors.append('#1565C0')  # 最佳蓝色
        else:
            colors.append('#43A047')  # 改进绿色
    
    # 绘制柱状图
    bars = ax.bar(x, improvements, width=0.65, color=colors, 
                  edgecolor='black', linewidth=2.0)
    
    # 最佳结果加粗边框
    bars[-1].set_edgecolor('#0D47A1')
    bars[-1].set_linewidth(3.5)
    
    # 添加数值标签
    for i, (bar, imp, nmse) in enumerate(zip(bars, improvements, nmse_values)):
        height = bar.get_height()
        # 显示改进量
        if imp > 0:
            label = f'+{imp:.2f} dB'
        else:
            label = f'{imp:.2f} dB'
        
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.15, 
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 在柱子下方显示实际NMSE
        ax.text(bar.get_x() + bar.get_width()/2, -0.3, 
                f'({nmse:.1f})', ha='center', va='top', fontsize=9, color='#555555')
    
    # 设置坐标轴
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement over Baseline (dB)', fontsize=14, fontweight='bold')
    ax.set_title('Ablation Study: Cumulative Improvement Analysis', fontsize=16, fontweight='bold', pad=15)
    
    # 添加网格和基准线
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, linewidth=1.0)
    ax.axhline(y=0, color='black', linewidth=2.0)
    ax.set_axisbelow(True)
    
    # 调整y轴范围
    ax.set_ylim(-0.8, max(improvements) + 1)
    
    plt.tight_layout()
    
    # 保存
    fig.savefig(os.path.join(output_dir, 'fig_improvement_waterfall.pdf'), format='pdf', bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig_improvement_waterfall.eps'), format='eps', bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig_improvement_waterfall.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✓ Saved: fig_improvement_waterfall.[pdf/eps/png]")


def create_gamma_distribution_figure(output_dir):
    """创建Gamma分布示意图"""
    setup_ieee_style()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.linspace(0, 8, 500)
    
    # 不同参数的Gamma分布
    from scipy import stats
    
    params = [
        (1, 0.5, '#E53935', 'α=1, β=0.5'),
        (2, 0.5, '#1E88E5', 'α=2, β=0.5'),
        (3, 0.5, '#43A047', 'α=3, β=0.5'),
        (2, 1.0, '#FB8C00', 'α=2, β=1.0'),
    ]
    
    for alpha, beta, color, label in params:
        y = stats.gamma.pdf(x, a=alpha, scale=1/beta)
        ax.plot(x, y, linewidth=3.0, color=color, label=label)
    
    ax.set_xlabel('Amplitude', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
    ax.set_title('Gamma Distribution of Dynamic Multipath Amplitude', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=1.0)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    
    # 保存
    fig.savefig(os.path.join(output_dir, 'Gamma.pdf'), format='pdf', bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'Gamma.eps'), format='eps', bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'Gamma.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✓ Saved: Gamma.[pdf/eps/png]")


def create_comprehensive_ablation_figure(results_df, output_dir):
    """创建综合消融实验图"""
    setup_ieee_style()
    
    # 增大图像尺寸，增加子图间距
    fig = plt.figure(figsize=(18, 12))
    
    # 准备数据
    categories = results_df['Category'].tolist()
    total_nmse = results_df['Test Total NMSE (dB)'].tolist()
    static_nmse = results_df['Test Static NMSE (dB)'].tolist()
    dynamic_nmse = results_df['Test Dynamic NMSE (dB)'].tolist()
    psnr = results_df['Test Total PSNR (dB)'].tolist()
    display_names = [MODEL_NAMES.get(cat, cat) for cat in categories]
    
    # 按总NMSE排序
    sorted_indices = np.argsort(total_nmse)
    names_sorted = [display_names[i] for i in sorted_indices]
    total_sorted = [total_nmse[i] for i in sorted_indices]
    static_sorted = [static_nmse[i] for i in sorted_indices]
    dynamic_sorted = [dynamic_nmse[i] for i in sorted_indices]
    psnr_sorted = [psnr[i] for i in sorted_indices]
    
    # 简化名称用于显示
    short_names = []
    for n in names_sorted:
        if 'Baseline' in n:
            short_names.append('Baseline')
        elif 'Full' in n:
            short_names.append('DSS-Net')
        elif 'Temporal' in n:
            short_names.append('w/o Temp.')
        elif 'Smooth' in n:
            short_names.append('w/o Smooth')
        elif 'Physics' in n:
            short_names.append('w/o Physics')
        elif 'Separation' in n:
            short_names.append('w/o Sep.')
        elif 'Attention' in n:
            short_names.append('w/o Attn.')
        else:
            short_names.append(n[:10])
    
    # 创建子图布局 - 增加间距
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
    
    # ===== (a) 总NMSE对比 =====
    ax1 = fig.add_subplot(gs[0, 0])
    n_bars = len(total_sorted)
    y_pos = np.arange(n_bars)
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, n_bars))
    
    bars = ax1.barh(y_pos, total_sorted, height=0.6, color=colors, 
                    edgecolor='black', linewidth=2.0)
    bars[0].set_edgecolor('#1565C0')
    bars[0].set_linewidth(3.5)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names_sorted, fontsize=11, fontweight='bold')
    ax1.set_xlabel('NMSE (dB)', fontsize=13, fontweight='bold')
    ax1.set_title('(a) Total NMSE Comparison', fontsize=14, fontweight='bold', pad=10)
    ax1.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax1.set_axisbelow(True)
    
    # ===== (b) 静态/动态分量对比 =====
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(short_names))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, static_sorted, width, label='Static', 
                    color='#1565C0', edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, dynamic_sorted, width, label='Dynamic', 
                    color='#C62828', edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Configuration', fontsize=13, fontweight='bold')
    ax2.set_ylabel('NMSE (dB)', fontsize=13, fontweight='bold')
    ax2.set_title('(b) Static vs Dynamic NMSE', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax2.set_axisbelow(True)
    
    # ===== (c) 热力图 =====
    ax3 = fig.add_subplot(gs[1, 0])
    
    heatmap_data = np.array([
        total_sorted,
        static_sorted,
        dynamic_sorted,
    ])
    
    im = ax3.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
    
    ax3.set_xticks(range(len(short_names)))
    ax3.set_xticklabels(short_names, rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['Total', 'Static', 'Dynamic'], fontsize=12, fontweight='bold')
    ax3.set_title('(c) Performance Heatmap (NMSE dB)', fontsize=14, fontweight='bold', pad=10)
    
    # 添加数值标注
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            text = ax3.text(j, i, f'{heatmap_data[i, j]:.1f}', ha='center', va='center', 
                           color='white' if heatmap_data[i, j] < -22 else 'black', 
                           fontsize=10, fontweight='bold')
    
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="4%", pad=0.15)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('NMSE (dB)', fontsize=11, fontweight='bold')
    
    # ===== (d) PSNR对比 =====
    ax4 = fig.add_subplot(gs[1, 1])
    colors_psnr = plt.cm.YlGn(np.linspace(0.3, 0.9, len(psnr_sorted)))
    
    bars = ax4.barh(y_pos, psnr_sorted, height=0.6, color=colors_psnr, 
                    edgecolor='black', linewidth=2.0)
    bars[-1].set_edgecolor('#2E7D32')
    bars[-1].set_linewidth(3.5)
    
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(names_sorted, fontsize=11, fontweight='bold')
    ax4.set_xlabel('PSNR (dB)', fontsize=13, fontweight='bold')
    ax4.set_title('(d) PSNR Comparison', fontsize=14, fontweight='bold', pad=10)
    ax4.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax4.set_axisbelow(True)
    
    # 使用subplots_adjust进一步调整
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    
    # 保存
    fig.savefig(os.path.join(output_dir, 'fig_comprehensive_ablation.pdf'), format='pdf', bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig_comprehensive_ablation.eps'), format='eps', bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig_comprehensive_ablation.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✓ Saved: fig_comprehensive_ablation.[pdf/eps/png]")


def main():
    # 路径设置
    base_dir = '/LSEM/user/chenyinda/code/signal_dy_static/1104/results_20251104_092511'
    eval_summary_path = os.path.join(base_dir, 'paper_eval', 'test_evaluation_summary.csv')
    output_dir = '/LSEM/user/chenyinda/code/signal_dy_static/xiaoyu_paper/figs'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    if not os.path.exists(eval_summary_path):
        print(f"Error: {eval_summary_path} not found")
        return
    
    results_df = pd.read_csv(eval_summary_path)
    print(f"Loaded {len(results_df)} experiment results")
    print(results_df[['Category', 'Test Total NMSE (dB)', 'Test Static NMSE (dB)', 'Test Dynamic NMSE (dB)']])
    
    print("\n" + "="*60)
    print("Generating IEEE-quality paper figures...")
    print("="*60 + "\n")
    
    # 生成所有图表
    create_ablation_nmse_figure(results_df, output_dir)
    create_component_comparison_figure(results_df, output_dir)
    create_improvement_waterfall_figure(results_df, output_dir)
    create_gamma_distribution_figure(output_dir)
    create_comprehensive_ablation_figure(results_df, output_dir)
    
    print("\n" + "="*60)
    print(f"All figures saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

