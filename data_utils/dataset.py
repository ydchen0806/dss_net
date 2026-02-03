"""
数据集类 - 加载和处理.mat文件
优化版：支持多进程并行加载、数据共享、DDP
"""

import os
import glob
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def _load_single_mat_file(file_path: str) -> Optional[Dict[str, np.ndarray]]:
    """
    加载单个.mat文件（用于多进程）
    
    Args:
        file_path: 文件路径
    Returns:
        data: 包含所有变量的字典，如果加载失败返回None
    """
    try:
        data = sio.loadmat(file_path)
        
        result = {
            'input': data['save_channel_matrix_noise'],
            'target': data['save_channel_matrix'],
            'static': data['save_static_component'],
            'dynamic': data['save_dynamic_component'],
            'static_paths': data['save_static_paths'],
            'dynamic_paths': data['save_dynamic_paths']
        }
        
        return result
        
    except Exception as e:
        print(f"  Error loading {os.path.basename(file_path)}: {e}")
        return None


def load_all_data_parallel(
    file_paths: List[str],
    max_workers: int = 8,
    rank: int = 0
) -> Dict[str, np.ndarray]:
    """
    并行加载所有.mat文件
    
    Args:
        file_paths: 文件路径列表
        max_workers: 最大进程数
        rank: DDP rank（只在rank 0打印）
    Returns:
        dataset: 合并后的数据集
    """
    if rank == 0:
        print(f"\n🚀 Loading {len(file_paths)} files in parallel ({max_workers} workers)...")
    
    all_data = {
        'input': [],
        'target': [],
        'static': [],
        'dynamic': [],
        'static_paths': [],
        'dynamic_paths': []
    }
    
    # 使用进程池并行加载
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_path = {
            executor.submit(_load_single_mat_file, fp): fp 
            for fp in file_paths
        }
        
        # 收集结果（带进度显示，只在rank 0显示）
        if rank == 0:
            pbar = tqdm(
                as_completed(future_to_path), 
                total=len(file_paths),
                desc="Loading files"
            )
        else:
            pbar = as_completed(future_to_path)
        
        for future in pbar:
            file_path = future_to_path[future]
            
            try:
                data = future.result()
                
                if data is not None:
                    for key in all_data:
                        all_data[key].append(data[key])
                        
            except Exception as e:
                if rank == 0:
                    print(f"\n  ❌ Error processing {os.path.basename(file_path)}: {e}")
    
    # 合并所有数据
    if rank == 0:
        print("\n📦 Concatenating data...")
    
    dataset = {k: np.concatenate(v, axis=0) for k, v in all_data.items()}
    
    if rank == 0:
        print(f"✅ Loaded successfully!")
        print(f"   Total samples: {dataset['input'].shape[0]}")
        print(f"   Shape: {dataset['input'].shape}")
        print(f"   Dtype: {dataset['input'].dtype}")
        memory_gb = dataset['input'].nbytes / 1024**3
        print(f"   Memory: {memory_gb:.2f} GB\n")
    
    return dataset


class ChannelDataset(Dataset):
    """无线信道数据集"""
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        pattern: str = "channel_generation*/Tx_all_2D_*.mat",
        mode: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        normalize_method: str = "power",
        complex_representation: str = "real_imag",
        augmentation: Optional[Dict] = None,
        max_files: Optional[int] = None,
        seed: int = 42,
        preloaded_data: Optional[Dict[str, np.ndarray]] = None,
        rank: int = 0
    ):
        """
        Args:
            data_dir: 数据根目录（如果preloaded_data为None则需要）
            pattern: 文件匹配模式
            mode: 'train', 'val', or 'test'
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            normalize_method: 归一化方法 ('standard', 'power', 'minmax')
            complex_representation: 复数表示 ('real_imag', 'mag_phase')
            augmentation: 数据增强配置字典
            max_files: 最大文件数（用于调试）
            seed: 随机种子
            preloaded_data: 预加载的数据字典（优先使用）
            rank: DDP rank（只在rank 0打印）
        """
        super().__init__()
        
        self.mode = mode
        self.normalize_method = normalize_method
        self.complex_representation = complex_representation
        self.augmentation = augmentation or {}
        self.seed = seed
        self.rank = rank
        
        # 优先使用预加载的数据
        if preloaded_data is not None:
            if rank == 0:
                print(f"📥 Using preloaded data for {mode}")
            self.data = preloaded_data
        else:
            # 如果没有预加载数据，按原方式加载
            if data_dir is None:
                raise ValueError("data_dir must be provided if preloaded_data is None")
            
            self.file_paths = self._get_file_paths(data_dir, pattern, max_files)
            if rank == 0:
                print(f"Loading {mode} data...")
            self.data = self._load_all_data_sequential()
        
        # 划分数据集
        self._split_dataset(train_ratio, val_ratio, test_ratio)
        
        if rank == 0:
            print(f"{mode.upper()} dataset: {len(self)} samples")
    
    def _get_file_paths(
        self, 
        data_dir: str, 
        pattern: str, 
        max_files: Optional[int]
    ) -> List[str]:
        """获取所有.mat文件路径"""
        search_pattern = os.path.join(data_dir, pattern)
        file_paths = sorted(glob.glob(search_pattern))
        
        if len(file_paths) == 0:
            raise FileNotFoundError(f"No files found matching: {search_pattern}")
        
        if max_files is not None:
            file_paths = file_paths[:max_files]
        
        if self.rank == 0:
            print(f"Found {len(file_paths)} .mat files")
        
        return file_paths
    
    def _load_all_data_sequential(self) -> Dict[str, np.ndarray]:
        """加载所有数据（单进程版本，用于向后兼容）"""
        all_data = {
            'input': [],
            'target': [],
            'static': [],
            'dynamic': [],
            'static_paths': [],
            'dynamic_paths': []
        }
        
        for i, file_path in enumerate(self.file_paths):
            if self.rank == 0:
                print(f"  [{i+1}/{len(self.file_paths)}] Loading {os.path.basename(file_path)}...")
            
            try:
                data = sio.loadmat(file_path)
                
                all_data['input'].append(data['save_channel_matrix_noise'])
                all_data['target'].append(data['save_channel_matrix'])
                all_data['static'].append(data['save_static_component'])
                all_data['dynamic'].append(data['save_dynamic_component'])
                all_data['static_paths'].append(data['save_static_paths'])
                all_data['dynamic_paths'].append(data['save_dynamic_paths'])
                
            except Exception as e:
                if self.rank == 0:
                    print(f"  Error loading {file_path}: {e}")
                continue
        
        dataset = {k: np.concatenate(v, axis=0) for k, v in all_data.items()}
        
        if self.rank == 0:
            print(f"  Total samples: {dataset['input'].shape[0]}")
            print(f"  Shape: {dataset['input'].shape}")
            print(f"  Dtype: {dataset['input'].dtype}")
        
        return dataset
    
    def _split_dataset(
        self, 
        train_ratio: float, 
        val_ratio: float, 
        test_ratio: float
    ):
        """划分数据集"""
        total_samples = len(self.data['input'])
        
        # 设置随机种子
        np.random.seed(self.seed)
        indices = np.arange(total_samples)
        
        # 按时间顺序划分（避免数据泄露）
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        if self.mode == 'train':
            self.indices = indices[:train_size]
        elif self.mode == 'val':
            self.indices = indices[train_size:train_size+val_size]
        elif self.mode == 'test':
            self.indices = indices[train_size+val_size:]
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        actual_idx = self.indices[idx]
        
        # 提取数据
        input_data = self.data['input'][actual_idx]      # (100, 150) complex
        target_data = self.data['target'][actual_idx]    # (100, 150) complex
        static_data = self.data['static'][actual_idx]    # (100, 150) complex
        dynamic_data = self.data['dynamic'][actual_idx]  # (100, 150) complex
        
        # 转换为实数表示
        input_tensor = self._complex_to_tensor(input_data)      # (2, 100, 150)
        target_tensor = self._complex_to_tensor(target_data)
        static_tensor = self._complex_to_tensor(static_data)
        dynamic_tensor = self._complex_to_tensor(dynamic_data)
        
        # 归一化
        if self.normalize_method != "none":
            input_tensor, scale = self._normalize(input_tensor)
            target_tensor = target_tensor / (scale + 1e-8)
            static_tensor = static_tensor / (scale + 1e-8)
            dynamic_tensor = dynamic_tensor / (scale + 1e-8)
        
        # 数据增强
        if self.mode == 'train' and self.augmentation.get('enabled', False):
            input_tensor, static_tensor, dynamic_tensor = self._augment(
                input_tensor, static_tensor, dynamic_tensor
            )
        
        sample = {
            'input': input_tensor.float(),
            'target': target_tensor.float(),
            'static': static_tensor.float(),
            'dynamic': dynamic_tensor.float(),
            'index': actual_idx
        }
        
        return sample
    
    def _complex_to_tensor(self, complex_data: np.ndarray) -> torch.Tensor:
        """将复数数组转换为张量"""
        if self.complex_representation == "real_imag":
            real = np.real(complex_data)
            imag = np.imag(complex_data)
            tensor = np.stack([real, imag], axis=0)  # (2, H, W)
        elif self.complex_representation == "mag_phase":
            mag = np.abs(complex_data)
            phase = np.angle(complex_data)
            tensor = np.stack([mag, phase], axis=0)
        else:
            raise ValueError(f"Unknown representation: {self.complex_representation}")
        
        return torch.from_numpy(tensor).float()
    
    def _normalize(
        self, 
        tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """归一化张量"""
        if self.normalize_method == "standard":
            mean = tensor.mean()
            std = tensor.std() + 1e-8
            normalized = (tensor - mean) / std
            scale = std
        
        elif self.normalize_method == "power":
            power = torch.sqrt((tensor ** 2).mean())
            normalized = tensor / (power + 1e-8)
            scale = power
        
        elif self.normalize_method == "minmax":
            min_val = tensor.min()
            max_val = tensor.max()
            normalized = (tensor - min_val) / (max_val - min_val + 1e-8)
            scale = max_val - min_val
        
        else:
            normalized = tensor
            scale = torch.tensor(1.0)
        
        return normalized, scale
    
    def _augment(
        self, 
        input_t: torch.Tensor, 
        static_t: torch.Tensor, 
        dynamic_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """数据增强"""
        # 1. 空间Masking
        if random.random() < self.augmentation.get('spatial_masking', {}).get('prob', 0.3):
            mask_ratio = self.augmentation['spatial_masking'].get('mask_ratio', 0.1)
            input_t = self._spatial_mask(input_t, mask_ratio)
        
        # 2. 噪声注入
        if random.random() < self.augmentation.get('noise_injection', {}).get('prob', 0.3):
            snr_range = self.augmentation['noise_injection'].get('snr_range', [20, 35])
            input_t = self._add_noise(input_t, snr_range)
        
        # 3. 相位旋转
        if random.random() < 0.2:
            angle = random.uniform(-np.pi/4, np.pi/4)
            input_t = self._phase_rotate(input_t, angle)
            static_t = self._phase_rotate(static_t, angle)
            dynamic_t = self._phase_rotate(dynamic_t, angle)
        
        return input_t, static_t, dynamic_t
    
    def _spatial_mask(self, tensor: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """空间Masking"""
        mask = torch.rand_like(tensor[0:1]) > mask_ratio
        return tensor * mask
    
    def _add_noise(self, tensor: torch.Tensor, snr_range: List[float]) -> torch.Tensor:
        """添加噪声"""
        snr_db = random.uniform(snr_range[0], snr_range[1])
        snr_linear = 10 ** (snr_db / 10)
        
        signal_power = (tensor ** 2).mean()
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(tensor) * torch.sqrt(noise_power)
        
        return tensor + noise
    
    def _phase_rotate(self, tensor: torch.Tensor, angle: float) -> torch.Tensor:
        """相位旋转（仅适用于real_imag表示）"""
        if self.complex_representation != "real_imag":
            return tensor
        
        real = tensor[0]
        imag = tensor[1]
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        new_real = real * cos_a - imag * sin_a
        new_imag = real * sin_a + imag * cos_a
        
        return torch.stack([new_real, new_imag], dim=0)


def create_dataloaders(
    config: Dict, 
    rank: int = 0  # ✅ 接收rank参数
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器（优化版 - 只加载一次数据 + 支持DDP）
    
    Args:
        config: 配置字典
        rank: DDP rank（默认0）
    Returns:
        train_loader, val_loader, test_loader
    """
    data_config = config['data']
    
    # ✅ 检查是否使用DDP
    use_ddp = config['hardware'].get('use_ddp', False)
    
    # 获取world_size
    if use_ddp:
        import torch.distributed as dist
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
    else:
        world_size = 1
    
    # ✅ 只在rank 0打印信息
    if rank == 0:
        print("\n" + "="*60)
        print("🔄 Loading data (shared across train/val/test)")
        if use_ddp:
            print(f"   DDP Mode: {world_size} processes")
        print("="*60)
    
    # 获取文件路径
    search_pattern = os.path.join(data_config['data_dir'], data_config['pattern'])
    file_paths = sorted(glob.glob(search_pattern))
    
    if len(file_paths) == 0:
        raise FileNotFoundError(f"No files found matching: {search_pattern}")
    
    max_files = data_config.get('max_files', None)
    if max_files is not None:
        file_paths = file_paths[:max_files]
    
    if rank == 0:
        print(f"Found {len(file_paths)} .mat files")
    
    # ✅ 使用多进程并行加载（只加载一次！）
    max_workers = min(8, len(file_paths), os.cpu_count() or 1)
    shared_data = load_all_data_parallel(file_paths, max_workers=max_workers, rank=rank)
    
    if rank == 0:
        print("="*60)
        print("📊 Creating datasets from shared data...")
        print("="*60 + "\n")
    
    # 三个数据集共享同一份数据
    train_dataset = ChannelDataset(
        mode='train',
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        test_ratio=data_config['test_ratio'],
        normalize_method=data_config['normalize_method'],
        complex_representation=data_config['complex_representation'],
        augmentation=data_config.get('augmentation'),
        seed=config['experiment']['seed'],
        preloaded_data=shared_data,
        rank=rank
    )
    
    val_dataset = ChannelDataset(
        mode='val',
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        test_ratio=data_config['test_ratio'],
        normalize_method=data_config['normalize_method'],
        complex_representation=data_config['complex_representation'],
        seed=config['experiment']['seed'],
        preloaded_data=shared_data,
        rank=rank
    )
    
    test_dataset = ChannelDataset(
        mode='test',
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        test_ratio=data_config['test_ratio'],
        normalize_method=data_config['normalize_method'],
        complex_representation=data_config['complex_representation'],
        seed=config['experiment']['seed'],
        preloaded_data=shared_data,
        rank=rank
    )
    
    # DDP模式下使用DistributedSampler
    if use_ddp:
        from torch.utils.data.distributed import DistributedSampler
        
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config['experiment']['seed']
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        train_shuffle = False
        
        if rank == 0:
            print(f"✅ Using DistributedSampler (world_size={world_size})")
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        train_shuffle = True
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        drop_last=True,
        persistent_workers=True if data_config['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        persistent_workers=True if data_config['num_workers'] > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        sampler=test_sampler,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        persistent_workers=True if data_config['num_workers'] > 0 else False
    )
    
    if rank == 0:
        print("\n✅ All dataloaders created successfully!")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}\n")
    
    return train_loader, val_loader, test_loader