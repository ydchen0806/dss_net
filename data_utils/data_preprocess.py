"""
数据预处理工具
用于批量处理.mat文件、数据清洗、归一化等
"""

import os
import sys
import argparse
import glob
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from typing import Dict, List, Optional
import h5py


def load_mat_file(filepath: str) -> Dict[str, np.ndarray]:
    """
    加载.mat文件
    
    Args:
        filepath: .mat文件路径
    Returns:
        data: 数据字典
    """
    try:
        # 尝试使用scipy加载
        data = sio.loadmat(filepath)
        return data
    except:
        # 如果是v7.3格式，使用h5py
        try:
            with h5py.File(filepath, 'r') as f:
                data = {}
                for key in f.keys():
                    if not key.startswith('__'):
                        data[key] = np.array(f[key])
                return data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None


def verify_data_integrity(data: Dict[str, np.ndarray]) -> bool:
    """
    验证数据完整性
    
    Args:
        data: 数据字典
    Returns:
        is_valid: 是否有效
    """
    required_keys = [
        'save_channel_matrix_noise',
        'save_channel_matrix',
        'save_static_component',
        'save_dynamic_component',
        'save_static_paths',
        'save_dynamic_paths'
    ]
    
    # 检查所有必需的键是否存在
    for key in required_keys:
        if key not in data:
            print(f"Missing key: {key}")
            return False
    
    # 检查数据形状是否一致
    H_noise = data['save_channel_matrix_noise']
    H = data['save_channel_matrix']
    H_static = data['save_static_component']
    H_dynamic = data['save_dynamic_component']
    
    if not (H_noise.shape == H.shape == H_static.shape == H_dynamic.shape):
        print("Shape mismatch between channel matrices")
        return False
    
    # 验证分量合成
    residual = H - (H_static + H_dynamic)
    error = np.mean(np.abs(residual))
    
    if error > 1e-10:
        print(f"Component composition error: {error:.2e}")
        return False
    
    return True


def compute_statistics(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    计算数据统计信息
    
    Args:
        data: 数据字典
    Returns:
        stats: 统计信息字典
    """
    H_noise = data['save_channel_matrix_noise']
    H = data['save_channel_matrix']
    H_static = data['save_static_component']
    H_dynamic = data['save_dynamic_component']
    
    noise = H_noise - H
    
    stats = {
        # 功率
        'total_power': float(np.mean(np.abs(H)**2)),
        'static_power': float(np.mean(np.abs(H_static)**2)),
        'dynamic_power': float(np.mean(np.abs(H_dynamic)**2)),
        'noise_power': float(np.mean(np.abs(noise)**2)),
        
        # 信噪比
        'snr_db': float(10 * np.log10(np.mean(np.abs(H)**2) / (np.mean(np.abs(noise)**2) + 1e-10))),
        
        # 幅度范围
        'max_magnitude': float(np.max(np.abs(H_noise))),
        'min_magnitude': float(np.min(np.abs(H_noise))),
        
        # 稀疏性
        'static_sparsity': float(np.sum(np.abs(H_static) < 1e-6) / H_static.size),
        'dynamic_sparsity': float(np.sum(np.abs(H_dynamic) < 1e-6) / H_dynamic.size),
        
        # 路径统计
        'avg_static_paths': float(np.mean(np.sum(data['save_static_paths'] > 0, axis=-1))),
        'avg_dynamic_paths': float(np.mean(np.sum(data['save_dynamic_paths'] > 0, axis=-1))),
    }
    
    return stats


def normalize_data(
    data: np.ndarray,
    method: str = 'power',
    axis: Optional[tuple] = None
) -> np.ndarray:
    """
    归一化数据
    
    Args:
        data: 输入数据
        method: 归一化方法 ('power', 'standard', 'minmax')
        axis: 归一化的轴
    Returns:
        normalized: 归一化后的数据
    """
    if method == 'power':
        power = np.sqrt(np.mean(np.abs(data)**2, axis=axis, keepdims=True))
        normalized = data / (power + 1e-8)
    
    elif method == 'standard':
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        normalized = (data - mean) / (std + 1e-8)
    
    elif method == 'minmax':
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
    
    else:
        normalized = data
    
    return normalized


def remove_outliers(data: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """
    移除异常值（基于z-score）
    
    Args:
        data: 输入数据
        threshold: z-score阈值
    Returns:
        cleaned: 清洗后的数据
    """
    magnitude = np.abs(data)
    mean = np.mean(magnitude)
    std = np.std(magnitude)
    
    z_scores = np.abs((magnitude - mean) / (std + 1e-8))
    mask = z_scores < threshold
    
    cleaned = data.copy()
    cleaned[~mask] = 0
    
    num_outliers = np.sum(~mask)
    print(f"  Removed {num_outliers} outliers ({num_outliers/data.size*100:.2f}%)")
    
    return cleaned


def preprocess_single_file(
    filepath: str,
    output_dir: Optional[str] = None,
    normalize_method: str = 'power',
    remove_outliers_flag: bool = False,
    verify: bool = True
) -> bool:
    """
    预处理单个文件
    
    Args:
        filepath: 输入文件路径
        output_dir: 输出目录
        normalize_method: 归一化方法
        remove_outliers_flag: 是否移除异常值
        verify: 是否验证数据完整性
    Returns:
        success: 是否成功
    """
    print(f"\nProcessing: {os.path.basename(filepath)}")
    
    # 加载数据
    data = load_mat_file(filepath)
    if data is None:
        return False
    
    # 验证数据
    if verify and not verify_data_integrity(data):
        print("  ✗ Data integrity check failed")
        return False
    print("  ✓ Data integrity verified")
    
    # 计算统计信息
    stats = compute_statistics(data)
    print(f"  SNR: {stats['snr_db']:.2f} dB")
    print(f"  Static/Dynamic power ratio: {stats['static_power']/stats['dynamic_power']:.4f}")
    
    # 预处理
    processed_data = {}
    
    for key in ['save_channel_matrix_noise', 'save_channel_matrix',
                'save_static_component', 'save_dynamic_component']:
        
        array = data[key]
        
        # 移除异常值
        if remove_outliers_flag:
            array = remove_outliers(array)
        
        # 归一化
        if normalize_method != 'none':
            array = normalize_data(array, method=normalize_method)
        
        processed_data[key] = array
    
    # 保留路径信息
    processed_data['save_static_paths'] = data['save_static_paths']
    processed_data['save_dynamic_paths'] = data['save_dynamic_paths']
    
    # 保存
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(filepath))
        sio.savemat(output_path, processed_data)
        print(f"  ✓ Saved to: {output_path}")
    
    return True


def batch_preprocess(
    data_dir: str,
    pattern: str = "channel_generation*/Tx_all_2D_*.mat",
    output_dir: str = "./data/processed",
    normalize_method: str = 'power',
    remove_outliers_flag: bool = False,
    max_files: Optional[int] = None
):
    """
    批量预处理文件
    
    Args:
        data_dir: 数据目录
        pattern: 文件匹配模式
        output_dir: 输出目录
        normalize_method: 归一化方法
        remove_outliers_flag: 是否移除异常值
        max_files: 最大文件数
    """
    # 查找文件
    search_pattern = os.path.join(data_dir, pattern)
    files = sorted(glob.glob(search_pattern))
    
    if max_files:
        files = files[:max_files]
    
    print(f"\n{'='*60}")
    print(f"Batch Preprocessing")
    print(f"{'='*60}")
    print(f"Found {len(files)} files")
    print(f"Output directory: {output_dir}")
    print(f"Normalize method: {normalize_method}")
    print(f"Remove outliers: {remove_outliers_flag}")
    print(f"{'='*60}\n")
    
    # 处理每个文件
    success_count = 0
    for filepath in tqdm(files, desc="Processing files"):
        if preprocess_single_file(
            filepath,
            output_dir,
            normalize_method,
            remove_outliers_flag
        ):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Preprocessing completed!")
    print(f"Successful: {success_count}/{len(files)}")
    print(f"{'='*60}\n")


def analyze_dataset(
    data_dir: str,
    pattern: str = "channel_generation*/Tx_all_2D_*.mat",
    max_files: Optional[int] = None
):
    """
    分析数据集统计信息
    
    Args:
        data_dir: 数据目录
        pattern: 文件匹配模式
        max_files: 最大文件数
    """
    search_pattern = os.path.join(data_dir, pattern)
    files = sorted(glob.glob(search_pattern))
    
    if max_files:
        files = files[:max_files]
    
    print(f"\n{'='*60}")
    print(f"Dataset Analysis")
    print(f"{'='*60}")
    print(f"Files: {len(files)}\n")
    
    all_stats = []
    
    for filepath in tqdm(files, desc="Analyzing"):
        data = load_mat_file(filepath)
        if data and verify_data_integrity(data):
            stats = compute_statistics(data)
            all_stats.append(stats)
    
    if len(all_stats) == 0:
        print("No valid files found!")
        return
    
    # 计算总体统计
    print(f"\n{'='*60}")
    print(f"Overall Statistics")
    print(f"{'='*60}\n")
    
    for key in all_stats[0].keys():
        values = [s[key] for s in all_stats]
        print(f"{key:25s}: mean={np.mean(values):.6e}, std={np.std(values):.6e}")
    
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Preprocess channel data')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                       help='Output directory')
    parser.add_argument('--pattern', type=str,
                       default='channel_generation*/Tx_all_2D_*.mat',
                       help='File pattern')
    parser.add_argument('--normalize', type=str, default='power',
                       choices=['none', 'power', 'standard', 'minmax'],
                       help='Normalization method')
    parser.add_argument('--remove_outliers', action='store_true',
                       help='Remove outliers')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process')
    parser.add_argument('--analyze', action='store_true',
                       help='Only analyze dataset statistics')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_dataset(args.data_dir, args.pattern, args.max_files)
    else:
        batch_preprocess(
            args.data_dir,
            args.pattern,
            args.output_dir,
            args.normalize,
            args.remove_outliers,
            args.max_files
        )


if __name__ == '__main__':
    main()
