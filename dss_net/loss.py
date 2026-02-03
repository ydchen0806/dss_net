"""
损失函数定义
修复：混合精度下的SVD计算
新增：时间相关性约束（物理机理约束）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class ChannelDecompositionLoss(nn.Module):
    """
    改进的信道分解损失函数
    关键改进：
    1. 更平衡的重建权重（动态分量权重加大）
    2. 更弱的正则化约束（避免过度约束）
    3. 添加分离质量度量（确保静态和动态真的不同）
    4. 自适应权重调整（训练过程中动态调整）
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.weights = config['loss']['weights']
        # 🔧 减弱正则化强度
        self.sparsity_lambda = config['loss'].get('sparsity_lambda', 0.0001)  # 从0.001降到0.0001
        self.nuclear_lambda = config['loss'].get('nuclear_lambda', 0.0001)    # 从0.001降到0.0001
        
        # 🆕 时间相关性配置
        self.temporal_config = config['loss'].get('temporal_correlation', {})
        self.temporal_enabled = self.temporal_config.get('enabled', True)
        self.static_smooth = self.temporal_config.get('static_smooth', True)
        self.dynamic_varying = self.temporal_config.get('dynamic_varying', True)
        self.temporal_dim = self.temporal_config.get('dim', -1)
        
        # 🆕 分离质量配置
        self.separation_weight = config['loss'].get('separation_weight', 0.1)  # 鼓励静态和动态不同
    
    def forward(
        self, 
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        is_baseline: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: {
                'static': (B, 2, H, W) or None,
                'dynamic': (B, 2, H, W) or None,
                'total': (B, 2, H, W)
            }
            target: {
                'static': (B, 2, H, W),
                'dynamic': (B, 2, H, W),
                'target': (B, 2, H, W)
            }
            is_baseline: 是否为基线模型（不分离）
        
        Returns:
            losses: dict
        """
        # ============================================
        # 基线模型：只计算总重建损失
        # ============================================
        if is_baseline or pred['static'] is None:
            total_mse = F.mse_loss(pred['total'], target['target'])
            total_nmse_db = self._compute_nmse_db(pred['total'], target['target'])
            
            return {
                'total_loss': total_mse,
                'total_mse': total_mse,
                'total_nmse_db': total_nmse_db,
                # 占位符（用于日志兼容）
                'static_mse': torch.tensor(0.0, device=total_mse.device),
                'dynamic_mse': torch.tensor(0.0, device=total_mse.device),
                'static_l1': torch.tensor(0.0, device=total_mse.device),
                'dynamic_nuclear': torch.tensor(0.0, device=total_mse.device),
                'static_temporal': torch.tensor(0.0, device=total_mse.device),
                'dynamic_temporal': torch.tensor(0.0, device=total_mse.device),
                'static_nmse_db': torch.tensor(0.0, device=total_mse.device),
                'dynamic_nmse_db': torch.tensor(0.0, device=total_mse.device)
            }
        
        # ============================================
        # 分解模型：完整损失
        # ============================================
        # 1. MSE损失（最重要）
        static_mse = F.mse_loss(pred['static'], target['static'])
        dynamic_mse = F.mse_loss(pred['dynamic'], target['dynamic'])
        total_mse = F.mse_loss(pred['total'], target['target'])
        
        # 🔧 2. 减弱的L1稀疏性约束（静态分量）
        static_l1 = torch.mean(torch.abs(pred['static']))
        
        # 🔧 3. 减弱的核范数约束（动态分量低秩）
        dynamic_nuclear = self._compute_nuclear_norm(pred['dynamic'])
        
        # 🆕 4. 分离质量度量（确保静态和动态真的不同）
        separation_loss = self._compute_separation_quality(pred['static'], pred['dynamic'])
        
        # 5. 时间相关性约束（可选）
        if self.temporal_enabled:
            static_temporal = self._compute_temporal_variation(
                pred['static'], 
                should_be_smooth=self.static_smooth
            )
            dynamic_temporal = self._compute_temporal_variation(
                pred['dynamic'], 
                should_be_smooth=not self.dynamic_varying
            )
        else:
            static_temporal = torch.tensor(0.0, device=pred['static'].device)
            dynamic_temporal = torch.tensor(0.0, device=pred['dynamic'].device)
        
        # 🔧 6. 改进的总损失（更注重重建质量）
        # 核心思想：重建损失 >> 正则化约束
        reconstruction_loss = (
            self.weights.get('static_mse', 1.0) * static_mse +
            self.weights.get('dynamic_mse', 2.0) * dynamic_mse +  # 🔧 动态分量权重加大
            self.weights.get('total_mse', 3.0) * total_mse         # 🔧 总重建最重要
        )
        
        regularization_loss = (
            self.weights.get('static_l1', 0.01) * self.sparsity_lambda * static_l1 +  # 🔧 降低权重
            self.weights.get('dynamic_nuclear', 0.01) * self.nuclear_lambda * dynamic_nuclear  # 🔧 降低权重
        )
        
        temporal_loss = 0.0
        if self.temporal_enabled:
            temporal_loss = (
                self.weights.get('static_temporal', 0.01) * static_temporal +  # 🔧 降低权重
                self.weights.get('dynamic_temporal', 0.01) * dynamic_temporal   # 🔧 降低权重
            )
        
        separation_term = self.separation_weight * separation_loss
        
        total_loss = reconstruction_loss + regularization_loss + temporal_loss + separation_term
        
        # 6. NMSE (dB) - 用于评估
        static_nmse_db = self._compute_nmse_db(pred['static'], target['static'])
        dynamic_nmse_db = self._compute_nmse_db(pred['dynamic'], target['dynamic'])
        total_nmse_db = self._compute_nmse_db(pred['total'], target['target'])
        
        return {
            'total_loss': total_loss,
            'static_mse': static_mse,
            'dynamic_mse': dynamic_mse,
            'total_mse': total_mse,
            'static_l1': static_l1,
            'dynamic_nuclear': dynamic_nuclear,
            'static_temporal': static_temporal,      # 🆕
            'dynamic_temporal': dynamic_temporal,    # 🆕
            'static_nmse_db': static_nmse_db,
            'dynamic_nmse_db': dynamic_nmse_db,
            'total_nmse_db': total_nmse_db,
            'separation_loss': separation_loss  # 🆕 添加分离质量
        }
    
    def _compute_separation_quality(
        self,
        static: torch.Tensor,
        dynamic: torch.Tensor
    ) -> torch.Tensor:
        """
        计算静态和动态分量的分离质量
        目标：确保两个分量真的不同
        
        使用负相关性：如果静态和动态高度相关，说明分离不好
        
        Args:
            static: (B, 2, H, W)
            dynamic: (B, 2, H, W)
        
        Returns:
            separation_loss: 相关性的负值（越小越好，表示相关性低）
        """
        # 将张量展平
        static_flat = static.view(static.size(0), -1)  # (B, 2*H*W)
        dynamic_flat = dynamic.view(dynamic.size(0), -1)
        
        # 计算每个样本的相关系数
        correlations = []
        for i in range(static_flat.size(0)):
            s = static_flat[i]
            d = dynamic_flat[i]
            
            # 中心化
            s_mean = s.mean()
            d_mean = d.mean()
            s_centered = s - s_mean
            d_centered = d - d_mean
            
            # 计算相关系数
            numerator = torch.sum(s_centered * d_centered)
            denominator = torch.sqrt(torch.sum(s_centered**2) * torch.sum(d_centered**2))
            
            correlation = numerator / (denominator + 1e-8)
            correlations.append(torch.abs(correlation))  # 取绝对值
        
        # 平均相关性（我们希望这个值尽可能小）
        avg_correlation = torch.mean(torch.stack(correlations))
        
        return avg_correlation
    
    def _compute_temporal_variation(
        self, 
        tensor: torch.Tensor, 
        should_be_smooth: bool = True
    ) -> torch.Tensor:
        """
        改进的时间变化性计算
        
        Args:
            tensor: (B, 2, H, W)
            should_be_smooth: True表示惩罚大变化，False表示鼓励变化
        
        Returns:
            temporal_loss: scalar
        """
        # 沿指定维度计算相邻差异
        if self.temporal_dim == -1:  # W维度
            diff = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
        elif self.temporal_dim == -2:  # H维度
            diff = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
        else:
            raise ValueError(f"Unsupported temporal_dim: {self.temporal_dim}")
        
        # 计算L2范数的平方
        variation = torch.mean(diff ** 2)
        
        if should_be_smooth:
            # 静态分量：惩罚大的变化
            return variation
        else:
            # 🔧 动态分量：改用更温和的鼓励方式
            # 不再使用倒数（过于激进），而是用负数（温和鼓励）
            # 如果variation小，损失为正；如果variation大，损失为负（减小总损失）
            target_variation = 0.01  # 期望的变化量
            return F.relu(target_variation - variation)  # 只在变化太小时惩罚
    
    def _compute_nuclear_norm(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        计算核范数（奇异值之和）
        🔧 修复：强制使用float32避免half精度下的SVD错误
        
        Args:
            tensor: (B, 2, H, W)
        Returns:
            nuclear_norm: scalar
        """
        # 保存原始dtype
        original_dtype = tensor.dtype
        
        # 转为float32（SVD不支持half）
        tensor = tensor.float()
        
        # 转为复数
        dynamic_complex = self._to_complex(tensor)  # (B, H, W)
        
        # 计算每个样本的核范数
        nuclear_norms = []
        for b in range(dynamic_complex.shape[0]):
            matrix = dynamic_complex[b]  # (H, W)
            
            # SVD计算（float32）
            s = torch.linalg.svdvals(matrix)  # 奇异值
            nuclear_norms.append(torch.sum(s))
        
        # 平均并转回原始dtype
        nuclear_norm = torch.mean(torch.stack(nuclear_norms))
        
        return nuclear_norm.to(original_dtype)
    
    def _to_complex(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        将 (B, 2, H, W) 转为复数 (B, H, W)
        """
        real = tensor[:, 0, :, :]
        imag = tensor[:, 1, :, :]
        return torch.complex(real, imag)
    
    def _compute_nmse_db(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算NMSE (dB)
        NMSE_dB = 10 * log10(||pred - target||^2 / ||target||^2)
        """
        mse = torch.mean((pred - target) ** 2)
        target_power = torch.mean(target ** 2)
        
        nmse = mse / (target_power + 1e-10)
        nmse_db = 10 * torch.log10(nmse + 1e-10)
        
        return nmse_db


if __name__ == '__main__':
    # 测试损失函数
    config = {
        'loss': {
            'weights': {
                'static_mse': 1.0,
                'dynamic_mse': 1.0,
                'total_mse': 2.0,
                'static_l1': 0.5,
                'dynamic_nuclear': 0.3,
                'static_temporal': 0.1,
                'dynamic_temporal': 0.1
            },
            'sparsity_lambda': 0.01,
            'nuclear_lambda': 0.01,
            'temporal_correlation': {
                'enabled': True,
                'static_smooth': True,
                'dynamic_varying': True,
                'dim': -1
            }
        }
    }
    
    criterion = ChannelDecompositionLoss(config)
    
    # 创建假数据
    B, H, W = 4, 100, 150
    
    # 模拟静态分量（应该平滑）
    static = torch.randn(B, 2, H, 1).repeat(1, 1, 1, W)  # 复制，变化小
    static += torch.randn(B, 2, H, W) * 0.01  # 添加小噪声
    
    # 模拟动态分量（应该变化大）
    dynamic = torch.randn(B, 2, H, W)  # 完全随机，变化大
    
    pred = {
        'static': static,
        'dynamic': dynamic,
        'total': static + dynamic
    }
    
    target = {
        'static': static + torch.randn_like(static) * 0.1,
        'dynamic': dynamic + torch.randn_like(dynamic) * 0.1,
        'target': static + dynamic + torch.randn_like(static) * 0.1
    }
    
    # 测试混合精度
    with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
        losses = criterion(pred, target, is_baseline=False)
    
    print("="*60)
    print("📊 Loss Components (With Temporal Correlation)")
    print("="*60)
    for key, val in losses.items():
        if 'db' in key:
            print(f"{key:20s}: {val.item():8.2f} dB")
        else:
            print(f"{key:20s}: {val.item():.6f}")
    
    print("\n" + "="*60)
    print("🔬 Testing Ablation (Baseline Model)")
    print("="*60)
    
    pred_baseline = {
        'static': None,
        'dynamic': None,
        'total': static + dynamic
    }
    
    losses_baseline = criterion(pred_baseline, target, is_baseline=True)
    
    for key, val in losses_baseline.items():
        if 'db' in key:
            print(f"{key:20s}: {val.item():8.2f} dB")
        else:
            print(f"{key:20s}: {val.item():.6f}")