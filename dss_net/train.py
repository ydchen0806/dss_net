"""
训练脚本 - 改进版本
- 修复单GPU模式下的多进程问题
- 优化内存使用
- 抑制不必要的警告
- ✨ 在checkpoint中保存验证指标
"""

import os
import sys

# 🔧 在所有import前设置环境变量和警告抑制
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['UCX_LOG_LEVEL'] = 'error'
os.environ['NCCL_DEBUG'] = 'WARN'

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime

# DDP相关
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 导入自定义模块
from dataset import create_dataloaders
from model import UNetDecomposer, UNetBaseline, count_parameters
from loss import ChannelDecompositionLoss
from visualization import (
    create_comparison_grid,
    create_error_histogram,
    create_temporal_variation_plot
)

class Trainer:
    """训练器 - 支持DDP + 消融实验 + 验证指标记录"""
    
    def __init__(self, config: dict, rank: int = 0, world_size: int = 1):
        """
        Args:
            config: 配置字典（直接传递，不通过pickle）
            rank: 进程rank
            world_size: 总进程数
        """
        self.rank = rank
        self.world_size = world_size
        self.config = config
        
        # 🆕 只有world_size > 1时才使用DDP
        self.use_ddp = world_size > 1 and self.config['hardware'].get('use_ddp', False)
        
        # 设置随机种子
        self._set_seed(self.config['experiment']['seed'] + rank)
        
        # 创建输出目录（只在rank 0）
        self._setup_output_dirs()
        
        # 等待rank 0创建完目录
        if self.use_ddp:
            dist.barrier()
        
        # 设置设备
        if self.use_ddp:
            self.device = torch.device(f'cuda:{rank}')
            torch.cuda.set_device(self.device)
        else:
            # 🆕 单GPU模式：使用CUDA_VISIBLE_DEVICES指定的设备
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if rank == 0:
            print(f"\n🚀 Training Setup:")
            if self.use_ddp:
                print(f"   Mode: DDP with {world_size} GPUs")
            else:
                print(f"   Mode: Single GPU")
            print(f"   Device: {self.device}")
            if torch.cuda.is_available():
                print(f"   GPU: {torch.cuda.get_device_name(self.device)}")
        
        # 🔧 数据加载器创建
        if rank == 0:
            print("\n" + "="*80)
            print("📦 Creating Dataloaders...")
            print("="*80)
        
        # 🆕 传递use_ddp参数，只在真正使用DDP时才使用DistributedSampler
        self.train_loader, self.val_loader, self.test_loader = \
            create_dataloaders(self.config, rank=rank, world_size=world_size, use_ddp=self.use_ddp)
        
        # 同步所有进程
        if self.use_ddp:
            dist.barrier()
        
        # 创建模型
        if rank == 0:
            print("\n" + "="*80)
            print("🏗️  Building Model...")
            print("="*80)
        
        self.model = self._build_model()
        
        # 创建损失函数
        self.criterion = ChannelDecompositionLoss(self.config)
        
        # 创建优化器
        self.optimizer = self._build_optimizer()
        
        # 创建学习率调度器
        self.scheduler = self._build_scheduler()
        
        # 混合精度训练
        self.scaler = torch.amp.GradScaler('cuda') \
            if self.config['hardware']['use_amp'] and torch.cuda.is_available() else None
        
        # TensorBoard
        if rank == 0 and self.config['logging']['tensorboard']:
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # ✨ 新增：记录最佳验证指标
        self.best_val_metrics = {}
        
        # ✨ 新增：记录训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        if rank == 0:
            print("\n✅ Trainer initialized successfully!\n")
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def _setup_output_dirs(self):
        """创建输出目录"""
        exp_name = self.config['experiment']['name']
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 构建实验标识
        model_name = self.config['model']['name']
        is_ablation = self.config['model'].get('ablation', {}).get('enabled', False)
        temporal_enabled = self.config['loss'].get('temporal_correlation', {}).get('enabled', True)
        
        if is_ablation or model_name == 'UNetBaseline':
            exp_suffix = f"{model_name}_ablation"
        else:
            exp_suffix = model_name
            if not temporal_enabled:
                exp_suffix += "_no_temporal"
        
        self.exp_dir = Path(self.config['experiment']['output_dir']) / f"{exp_name}_{exp_suffix}_{timestamp}"
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.log_dir = self.exp_dir / "logs"
        
        if self.rank == 0:
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(exist_ok=True)
            self.log_dir.mkdir(exist_ok=True)
            
            # 保存配置
            with open(self.exp_dir / "config.yaml", 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            experiment_info = {
                'experiment_name': exp_name,
                'model_name': model_name,
                'is_ablation': is_ablation,
                'temporal_enabled': temporal_enabled,
                'timestamp': timestamp,
                'output_dir': str(self.exp_dir)
            }
            
            with open(self.exp_dir / "experiment_info.yaml", 'w') as f:
                yaml.dump(experiment_info, f, default_flow_style=False)
    
    def _build_model(self) -> nn.Module:
        """构建模型"""
        model_name = self.config['model']['name']
        is_ablation = self.config['model'].get('ablation', {}).get('enabled', False)
        
        if is_ablation or model_name == 'UNetBaseline':
            if self.rank == 0:
                print("📊 Building Baseline Model (No Decomposition - Ablation Study)")
            
            model = UNetBaseline(
                in_channels=self.config['model']['in_channels'],
                out_channels=self.config['model']['in_channels'],
                base_channels=self.config['model']['base_channels'],
                depth=self.config['model']['depth'],
                norm_type=self.config['model']['norm_type'],
                dropout=self.config['model']['dropout']
            )
            self.is_baseline = True
        
        elif model_name == 'UNetDecomposer':
            if self.rank == 0:
                print("🔬 Building Decomposition Model (Static + Dynamic)")
            
            model = UNetDecomposer(
                in_channels=self.config['model']['in_channels'],
                base_channels=self.config['model']['base_channels'],
                depth=self.config['model']['depth'],
                norm_type=self.config['model']['norm_type'],
                dropout=self.config['model']['dropout'],
                use_attention=self.config['model'].get('use_attention', False)  # 🆕 支持attention
            )
            self.is_baseline = False
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = model.to(self.device)
        
        if self.rank == 0:
            count_parameters(model)
            
            temporal_enabled = self.config['loss'].get('temporal_correlation', {}).get('enabled', True)
            if not self.is_baseline:
                if temporal_enabled:
                    print("✅ Temporal correlation constraints: ENABLED")
                else:
                    print("⚠️  Temporal correlation constraints: DISABLED (ablation)")
        
        # 🆕 只在真正使用DDP时才包装
        if self.use_ddp:
            model = DDP(
                model, 
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=False
            )
        
        return model
    
    def _build_optimizer(self) -> optim.Optimizer:
        """构建优化器"""
        if self.config['training']['optimizer'].lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif self.config['training']['optimizer'].lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['training']['optimizer']}")
        
        return optimizer
    
    def _build_scheduler(self):
        """构建学习率调度器"""
        scheduler_config = self.config['training']['scheduler']
        
        if scheduler_config['type'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=scheduler_config['min_lr']
            )
        elif scheduler_config['type'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_config['type'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=10,
                factor=0.5
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        
        # DDP: 设置epoch用于sampler的shuffle
        if self.use_ddp and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
        
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
        
        if self.rank == 0:
            print(f"\n🔄 Starting epoch {epoch}, total batches: {len(self.train_loader)}")
        
        pbar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch}/{self.config['training']['epochs']}",
            disable=(self.rank != 0)
        )
        
        for batch_idx, batch in enumerate(pbar):
            # 🔧 添加调试信息（仅第一个batch）
            if batch_idx == 0 and self.rank == 0:
                print(f"\n✅ 成功加载第一个batch")
            
            # 数据转移到设备
            inputs = batch['input'].to(self.device, non_blocking=True)
            target_static = batch['static'].to(self.device, non_blocking=True)
            target_dynamic = batch['dynamic'].to(self.device, non_blocking=True)
            target_total = batch['target'].to(self.device, non_blocking=True)
            
            # 🔧 第一个batch的调试信息
            if batch_idx == 0 and self.rank == 0:
                print(f"✅ 数据已传输到GPU")
                print(f"   Input shape: {inputs.shape}")
            
            # 前向传播
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    pred = self.model(inputs)
                    
                    losses = self.criterion(
                        pred,
                        {
                            'static': target_static,
                            'dynamic': target_dynamic,
                            'target': target_total
                        },
                        is_baseline=self.is_baseline
                    )
                
                # 反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(losses['total_loss']).backward()
                
                # 梯度裁剪
                if self.config['training'].get('gradient_clip'):
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            else:
                pred = self.model(inputs)
                
                losses = self.criterion(
                    pred,
                    {
                        'static': target_static,
                        'dynamic': target_dynamic,
                        'target': target_total
                    },
                    is_baseline=self.is_baseline
                )
                
                # 反向传播
                self.optimizer.zero_grad()
                losses['total_loss'].backward()
                
                # 梯度裁剪
                if self.config['training'].get('gradient_clip'):
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer.step()
            
            # 累积指标
            total_loss += losses['total_loss'].item()
            for key in metrics:
                if key in losses:
                    metrics[key] += losses[key].item()
            
            # 更新进度条
            if self.rank == 0:
                postfix_dict = {
                    'loss': losses['total_loss'].item(),
                    'total_db': losses['total_nmse_db'].item()
                }
                
                if not self.is_baseline:
                    postfix_dict.update({
                        'static_db': losses['static_nmse_db'].item(),
                        'dynamic_db': losses['dynamic_nmse_db'].item(),
                    })
                    
                    if self.config['loss'].get('temporal_correlation', {}).get('enabled', True):
                        postfix_dict.update({
                            's_temp': losses['static_temporal'].item(),
                            'd_temp': losses['dynamic_temporal'].item()
                        })
                
                pbar.set_postfix(postfix_dict)
            
            # TensorBoard日志
            if self.rank == 0 and self.writer is not None:
                global_step = epoch * len(self.train_loader) + batch_idx
                
                if batch_idx % self.config['logging']['log_interval'] == 0:
                    self.writer.add_scalar('Train/Loss', losses['total_loss'].item(), global_step)
                    self.writer.add_scalar('Train/Total_NMSE_dB', losses['total_nmse_db'].item(), global_step)
                    
                    if not self.is_baseline:
                        self.writer.add_scalar('Train/Static_NMSE_dB', losses['static_nmse_db'].item(), global_step)
                        self.writer.add_scalar('Train/Dynamic_NMSE_dB', losses['dynamic_nmse_db'].item(), global_step)
                        
                        if self.config['loss'].get('temporal_correlation', {}).get('enabled', True):
                            self.writer.add_scalar('Train/Static_Temporal', losses['static_temporal'].item(), global_step)
                            self.writer.add_scalar('Train/Dynamic_Temporal', losses['dynamic_temporal'].item(), global_step)
            
            # 🆕 清理缓存
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 平均指标
        avg_loss = total_loss / len(self.train_loader)
        avg_metrics = {k: v / len(self.train_loader) for k, v in metrics.items()}
        
        # DDP: 同步指标
        if self.use_ddp:
            avg_loss = self._reduce_value(avg_loss)
            avg_metrics = {k: self._reduce_value(v) for k, v in avg_metrics.items()}
        
        return avg_loss, avg_metrics
    
    @torch.no_grad()
    def validate(self, epoch: int):
        """验证"""
        self.model.eval()
        
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
        
        pbar = tqdm(
            self.val_loader,
            desc="Validating",
            disable=(self.rank != 0)
        )
        
        for batch in pbar:
            inputs = batch['input'].to(self.device, non_blocking=True)
            target_static = batch['static'].to(self.device, non_blocking=True)
            target_dynamic = batch['dynamic'].to(self.device, non_blocking=True)
            target_total = batch['target'].to(self.device, non_blocking=True)
            
            pred = self.model(inputs)
            
            losses = self.criterion(
                pred,
                {
                    'static': target_static,
                    'dynamic': target_dynamic,
                    'target': target_total
                },
                is_baseline=self.is_baseline
            )
            
            total_loss += losses['total_loss'].item()
            for key in metrics:
                if key in losses:
                    metrics[key] += losses[key].item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = {k: v / len(self.val_loader) for k, v in metrics.items()}
        
        # DDP: 同步指标
        if self.use_ddp:
            avg_loss = self._reduce_value(avg_loss)
            avg_metrics = {k: self._reduce_value(v) for k, v in avg_metrics.items()}
        
        # TensorBoard
        if self.rank == 0 and self.writer is not None:
            self.writer.add_scalar('Val/Loss', avg_loss, epoch)
            self.writer.add_scalar('Val/Total_NMSE_dB', avg_metrics['total_nmse_db'], epoch)
            
            if not self.is_baseline:
                self.writer.add_scalar('Val/Static_NMSE_dB', avg_metrics['static_nmse_db'], epoch)
                self.writer.add_scalar('Val/Dynamic_NMSE_dB', avg_metrics['dynamic_nmse_db'], epoch)
                
                if self.config['loss'].get('temporal_correlation', {}).get('enabled', True):
                    self.writer.add_scalar('Val/Static_Temporal', avg_metrics['static_temporal'], epoch)
                    self.writer.add_scalar('Val/Dynamic_Temporal', avg_metrics['dynamic_temporal'], epoch)
            
            # 🆕 可视化：根据配置决定是否记录图像
            vis_config = self.config['logging'].get('visualization', {})
            if vis_config.get('enabled', True):
                vis_interval = vis_config.get('interval', 5)
                if epoch % vis_interval == 0:
                    self._log_visualizations(epoch)
        
        return avg_loss, avg_metrics
    
    def _reduce_value(self, value: float) -> float:
        """DDP: 跨进程平均"""
        if not self.use_ddp:
            return value
        
        tensor = torch.tensor(value, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item() / self.world_size
    
    @torch.no_grad()
    def _log_visualizations(self, epoch: int):
        """
        记录可视化图像到TensorBoard
        
        Args:
            epoch: 当前epoch
        """
        if self.rank != 0 or self.writer is None:
            return
        
        print(f"\n📊 Generating visualizations for epoch {epoch}...")
        
        self.model.eval()
        
        # 获取可视化配置
        vis_config = self.config['logging'].get('visualization', {})
        num_samples = vis_config.get('num_samples', 4)
        modes = vis_config.get('modes', ['magnitude', 'phase'])
        
        # 从验证集获取一个batch
        try:
            # 获取验证集的第一个batch
            val_iter = iter(self.val_loader)
            batch = next(val_iter)
            
            inputs = batch['input'].to(self.device)
            target_static = batch['static'].to(self.device)
            target_dynamic = batch['dynamic'].to(self.device)
            target_total = batch['target'].to(self.device)
            
            # 前向传播
            pred = self.model(inputs)
            
            target = {
                'static': target_static,
                'dynamic': target_dynamic,
                'target': target_total
            }
            
            # 根据配置创建对比图
            for mode in modes:
                print(f"   - Creating {mode} comparison grid...")
                try:
                    grid_img = create_comparison_grid(
                        inputs, pred, target,
                        is_baseline=self.is_baseline,
                        num_samples=min(num_samples, inputs.size(0)),
                        mode=mode
                    )
                    # 转换为CHW格式 (TensorBoard需要)
                    grid_img = torch.from_numpy(grid_img).permute(2, 0, 1)
                    self.writer.add_image(f'Visualization/{mode.capitalize()}_Comparison', 
                                         grid_img, epoch, dataformats='CHW')
                except Exception as e:
                    print(f"   ⚠️  Failed to create {mode} grid: {e}")
            
            # 误差分布直方图
            print("   - Creating error histogram...")
            try:
                hist_img = create_error_histogram(pred, target, is_baseline=self.is_baseline)
                hist_img = torch.from_numpy(hist_img).permute(2, 0, 1)
                self.writer.add_image('Visualization/Error_Histogram', 
                                     hist_img, epoch, dataformats='CHW')
            except Exception as e:
                print(f"   ⚠️  Failed to create histogram: {e}")
            
            # 时间变化图（仅分解模型）
            if not self.is_baseline:
                print("   - Creating temporal variation plot...")
                try:
                    temporal_dim = self.config['loss'].get('temporal_correlation', {}).get('dim', -1)
                    temporal_img = create_temporal_variation_plot(
                        pred, target, 
                        is_baseline=False, 
                        dim=temporal_dim
                    )
                    if temporal_img is not None:
                        temporal_img = torch.from_numpy(temporal_img).permute(2, 0, 1)
                        self.writer.add_image('Visualization/Temporal_Variation', 
                                             temporal_img, epoch, dataformats='CHW')
                except Exception as e:
                    print(f"   ⚠️  Failed to create temporal plot: {e}")
            
            print(f"✅ Visualizations saved to TensorBoard!")
        
        except Exception as e:
            print(f"❌ Failed to generate visualizations: {e}")
            import traceback
            traceback.print_exc()
    
    def save_checkpoint(self, epoch: int, val_loss: float, val_metrics: dict, is_best: bool = False):
        """
        保存检查点 - ✨ 增加验证指标保存
        
        Args:
            epoch: 当前epoch
            val_loss: 验证损失
            val_metrics: 验证指标字典
            is_best: 是否为最佳模型
        """
        if self.rank != 0:
            return
        
        # DDP: 保存模型时去掉module.前缀
        model_state = self.model.module.state_dict() if self.use_ddp else self.model.state_dict()
        
        # ✨ 构建完整的checkpoint
        checkpoint = {
            # 模型相关
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            
            # ✨ 验证指标
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'best_val_loss': self.best_val_loss,
            'best_val_metrics': self.best_val_metrics,
            
            # ✨ 训练历史
            'train_history': self.train_history,
            
            # 配置信息
            'config': self.config,
            'is_baseline': self.is_baseline,
            
            # ✨ 额外信息
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_epochs': self.config['training']['epochs'],
        }
        
        # 保存最新
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        print(f"💾 Saved latest checkpoint to {latest_path}")
        
        # 保存最佳
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"🏆 Saved best checkpoint to {best_path}")
            print(f"   Val Loss: {val_loss:.6f}")
            print(f"   Val Total NMSE: {val_metrics['total_nmse_db']:.2f} dB")
            if not self.is_baseline:
                print(f"   Val Static NMSE: {val_metrics['static_nmse_db']:.2f} dB")
                print(f"   Val Dynamic NMSE: {val_metrics['dynamic_nmse_db']:.2f} dB")
        
        # 定期保存
        if epoch % self.config['logging']['save_checkpoint_interval'] == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{epoch:03d}.pth'
            torch.save(checkpoint, epoch_path)
            print(f"📌 Saved epoch checkpoint to {epoch_path}")
    
    def train(self):
        """主训练循环"""
        if self.rank == 0:
            print("\n" + "="*80)
            print("🚀 Starting Training...")
            print("="*80 + "\n")
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            self.current_epoch = epoch
            
            # 训练
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # ✨ 记录训练历史
            if self.rank == 0:
                self.train_history['train_loss'].append(train_loss)
                self.train_history['train_metrics'].append(train_metrics.copy())
            
            # 验证
            if epoch % self.config['validation']['interval'] == 0:
                val_loss, val_metrics = self.validate(epoch)
                
                # ✨ 记录验证历史
                if self.rank == 0:
                    self.train_history['val_loss'].append(val_loss)
                    self.train_history['val_metrics'].append(val_metrics.copy())
                
                # 打印结果
                if self.rank == 0:
                    print(f"\n{'='*80}")
                    print(f"Epoch {epoch}/{self.config['training']['epochs']}")
                    print(f"{'='*80}")
                    print(f"Train Loss: {train_loss:.6f}")
                    
                    if not self.is_baseline:
                        print(f"  Static NMSE:  {train_metrics['static_nmse_db']:.2f} dB")
                        print(f"  Dynamic NMSE: {train_metrics['dynamic_nmse_db']:.2f} dB")
                        
                        if self.config['loss'].get('temporal_correlation', {}).get('enabled', True):
                            print(f"  Static Temp:  {train_metrics['static_temporal']:.6f}")
                            print(f"  Dynamic Temp: {train_metrics['dynamic_temporal']:.6f}")
                    
                    print(f"  Total NMSE:   {train_metrics['total_nmse_db']:.2f} dB")
                    print(f"\nVal Loss: {val_loss:.6f}")
                    
                    if not self.is_baseline:
                        print(f"  Static NMSE:  {val_metrics['static_nmse_db']:.2f} dB")
                        print(f"  Dynamic NMSE: {val_metrics['dynamic_nmse_db']:.2f} dB")
                        
                        if self.config['loss'].get('temporal_correlation', {}).get('enabled', True):
                            print(f"  Static Temp:  {val_metrics['static_temporal']:.6f}")
                            print(f"  Dynamic Temp: {val_metrics['dynamic_temporal']:.6f}")
                    
                    print(f"  Total NMSE:   {val_metrics['total_nmse_db']:.2f} dB")
                    print(f"{'='*80}\n")
                
                # ✨ 保存最佳模型并更新最佳指标
                is_best = val_loss < self.best_val_loss
                if is_best:
                    if self.rank == 0:
                        print(f"🎉 New best model at epoch {epoch}!")
                        print(f"   Previous best loss: {self.best_val_loss:.6f}")
                        print(f"   New best loss: {val_loss:.6f}")
                        print(f"   Improvement: {self.best_val_loss - val_loss:.6f}")
                    
                    self.best_val_loss = val_loss
                    self.best_val_metrics = val_metrics.copy()  # ✨ 保存最佳指标
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.rank == 0:
                        print(f"⏳ No improvement for {self.patience_counter} validation(s)")
                        print(f"   Best loss so far: {self.best_val_loss:.6f}")
                
                # ✨ 保存checkpoint（传入验证指标）
                self.save_checkpoint(epoch, val_loss, val_metrics, is_best)
                
                # 早停
                if self.patience_counter >= self.config['training']['early_stopping']['patience']:
                    if self.rank == 0:
                        print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
                        print(f"   Best validation loss: {self.best_val_loss:.6f}")
                        print(f"   Best metrics:")
                        for key, value in self.best_val_metrics.items():
                            print(f"      {key}: {value:.6f}")
                    break
            
            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                
                # 记录学习率
                if self.rank == 0 and self.writer is not None:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar('Train/Learning_Rate', current_lr, epoch)
        
        # ✨ 训练结束后保存最终统计信息
        if self.rank == 0:
            self._save_training_summary()
            
            print("\n✅ Training completed!")
            print(f"📁 Results saved to: {self.exp_dir}")
            print(f"\n🏆 Best Results:")
            print(f"   Epoch: {self.train_history['val_loss'].index(self.best_val_loss) + 1}")
            print(f"   Val Loss: {self.best_val_loss:.6f}")
            print(f"   Metrics:")
            for key, value in self.best_val_metrics.items():
                print(f"      {key}: {value:.6f}")
            print()
    
    def _save_training_summary(self):
        """✨ 保存训练摘要"""
        if self.rank != 0:
            return
        
        summary = {
            'experiment_name': self.config['experiment']['name'],
            'model_name': self.config['model']['name'],
            'is_baseline': self.is_baseline,
            'total_epochs': self.current_epoch,
            'best_epoch': self.train_history['val_loss'].index(self.best_val_loss) + 1 if self.train_history['val_loss'] else -1,
            'best_val_loss': self.best_val_loss,
            'best_val_metrics': self.best_val_metrics,
            'final_train_loss': self.train_history['train_loss'][-1] if self.train_history['train_loss'] else None,
            'final_val_loss': self.train_history['val_loss'][-1] if self.train_history['val_loss'] else None,
            'training_completed': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        summary_path = self.exp_dir / 'training_summary.yaml'
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        print(f"📊 Training summary saved to {summary_path}")

def setup_ddp(rank: int, world_size: int):
    """初始化DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def cleanup_ddp():
    """清理DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()

def main_worker(rank: int, world_size: int, config: dict):
    """
    单个进程的工作函数
    
    Args:
        rank: 进程rank
        world_size: 总进程数
        config: 配置字典
    """
    # 🆕 只在world_size > 1时初始化DDP
    if world_size > 1:
        setup_ddp(rank, world_size)
    
    try:
        trainer = Trainer(config, rank=rank, world_size=world_size)
        trainer.train()
    
    except Exception as e:
        print(f"❌ Rank {rank} failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理
        if world_size > 1:
            cleanup_ddp()

def main():
    import argparse
    import torch.multiprocessing as mp
    
    parser = argparse.ArgumentParser(description='Train Channel Decomposition Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--gpus', type=int, default=None,
                        help='Number of GPUs to use (overrides config)')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 🆕 确定GPU数量：优先使用命令行参数
    if args.gpus is not None:
        world_size = args.gpus
    else:
        # 如果没有指定--gpus，检查CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible:
            # CUDA_VISIBLE_DEVICES被设置，计算可见GPU数量
            world_size = len([x for x in cuda_visible.split(',') if x.strip()])
        else:
            # 使用所有可用GPU
            world_size = torch.cuda.device_count() if config['hardware'].get('use_ddp', False) else 1
    
    # 🆕 如果只有1个GPU，强制单GPU模式
    if world_size == 1:
        config['hardware']['use_ddp'] = False
    
    print(f"\n{'='*80}")
    print(f"🚀 Starting Training")
    print(f"{'='*80}")
    print(f"   Experiment: {config['experiment']['name']}")
    print(f"   Model: {config['model']['name']}")
    
    # 打印实验信息
    is_ablation = config['model'].get('ablation', {}).get('enabled', False)
    temporal_enabled = config['loss'].get('temporal_correlation', {}).get('enabled', True)
    
    if is_ablation or config['model']['name'] == 'UNetBaseline':
        print(f"   Type: 🔬 ABLATION STUDY - Baseline (No Decomposition)")
    else:
        print(f"   Type: 🏆 Full Model (Static + Dynamic Decomposition)")
        if temporal_enabled:
            print(f"   Temporal Constraints: ✅ ENABLED")
        else:
            print(f"   Temporal Constraints: ⚠️  DISABLED (ablation)")
    
    print(f"   GPUs: {world_size}")
    print(f"   Config: {args.config}")
    print(f"{'='*80}\n")
    
    # 🆕 只有world_size > 1时才使用spawn
    if world_size > 1:
        mp.spawn(
            main_worker,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    else:
        # 🆕 单GPU直接运行，不使用spawn
        print("💡 Running in single GPU mode (no multiprocessing)\n")
        main_worker(0, 1, config)

if __name__ == '__main__':
    # 🆕 设置multiprocessing启动方法
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 已经设置过了
    
    main()