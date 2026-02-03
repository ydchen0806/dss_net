# 🔬 消融实验设计说明

## 📋 实验概览

本消融研究旨在系统性地验证每个改进组件的有效性，共设计8个实验，充分利用8张GPU并行训练。

---

## 🎯 实验列表

### 实验1: Baseline U-Net（GPU 0）
**配置文件**: `config_1_baseunet.yaml`  
**模型**: `UNetBaseline` - 不分解，直接重建  
**目的**: 
- 建立性能基准
- 证明分解的必要性

**预期结果**: 
- Total NMSE: ~-25 dB（参考值）
- 无静态/动态分量输出

---

### 实验2: 完整改进模型（GPU 1）⭐
**配置文件**: `config_2_full_improved.yaml`  
**模型**: `UNetDecomposer` + 所有改进  
**改进点**:
- ✅ 调整后的损失权重
- ✅ 减弱的正则化
- ✅ Attention机制
- ✅ 时间约束
- ✅ 分离质量约束

**目的**: 
- 达到最佳性能
- 作为其他消融实验的对比基准

**预期结果**: 
- Total NMSE: **> -25.5 dB**（优于baseline）
- Dynamic NMSE: **> -22 dB**（大幅改善）
- 最低Loss

---

### 实验3: 无时间约束（GPU 2）
**配置文件**: `config_3_no_temporal.yaml`  
**移除组件**: 
- ❌ `temporal_correlation.enabled = False`
- ❌ `static_temporal = 0.0`
- ❌ `dynamic_temporal = 0.0`

**目的**: 
- 验证时间相关性约束的作用
- 评估物理先验知识的价值

**预期现象**:
- 静态分量可能不够平滑
- 动态分量可能不够活跃
- 分离质量可能下降

**对比指标**: 
- 与实验2对比Total NMSE和分离质量
- 观察时间变化曲线的差异

---

### 实验4: 无Attention机制（GPU 3）
**配置文件**: `config_4_no_attention.yaml`  
**移除组件**: 
- ❌ `use_attention = False`
- ❌ Bottleneck attention层

**目的**: 
- 评估attention对特征提取的贡献
- 验证是否值得增加模型复杂度

**预期现象**:
- 训练速度略快（少量参数）
- Dynamic NMSE可能略差
- 显存占用稍低

**对比指标**: 
- Dynamic NMSE差异
- 训练时间
- 模型参数量

---

### 实验5: 无分离质量约束（GPU 4）
**配置文件**: `config_5_no_separation.yaml`  
**移除组件**: 
- ❌ `separation_weight = 0.0`
- ❌ 静态和动态的相关性惩罚

**目的**: 
- 验证separation loss的必要性
- 检查是否会出现"模式坍塌"（静态和动态过于相似）

**预期现象**:
- 可能出现静态和动态高度相似
- 分离意义不明显
- 虽然重建Loss可能低，但分解质量差

**对比指标**: 
- 静态和动态的相关系数
- 可视化对比（两者是否不同）

---

### 实验6: 原始损失权重（GPU 5）
**配置文件**: `config_6_old_weights.yaml`  
**使用旧配置**: 
- ❌ `dynamic_mse = 1.0`（未加大）
- ❌ `total_mse = 2.0`（未加大）
- ❌ `static_l1 = 0.1`（过强）
- ❌ `nuclear_lambda = 0.001`（过强）
- ❌ `learning_rate = 5e-4`（过低）

**目的**: 
- 重现原始问题
- 证明损失权重调整的必要性

**预期现象**:
- **Loss很高**（~0.3+）
- Dynamic NMSE很差（~-13 dB）
- 训练收敛慢

**对比指标**: 
- 与实验2的Loss差异（应有数量级差异）
- 训练曲线对比

---

### 实验7: 无正则化约束（GPU 6）
**配置文件**: `config_7_no_regularization.yaml`  
**移除组件**: 
- ❌ `static_l1 = 0.0`（无L1稀疏性）
- ❌ `nuclear_lambda = 0.0`（无核范数约束）

**目的**: 
- 验证正则化是否真的必要
- 测试纯重建是否足够

**预期现象**:
- 可能过拟合训练集
- 静态分量不够稀疏
- 动态分量不够低秩
- 但重建Loss可能很低

**对比指标**: 
- 训练集 vs 验证集性能差异
- 静态分量的稀疏性
- 动态分量的秩

---

### 实验8: 无静态平滑约束（GPU 7）
**配置文件**: `config_8_no_static_smooth.yaml`  
**移除组件**: 
- ❌ `static_temporal = 0.0`
- ❌ `static_smooth = False`
- ✅ 保留动态变化约束

**目的**: 
- 单独验证静态平滑约束的作用
- 与实验3（完全无时间约束）对比

**预期现象**:
- 静态分量可能有噪声/抖动
- 动态分量受约束，表现正常

**对比指标**: 
- 静态分量的时间变化率
- Static NMSE

---

## 📊 评估指标

### 主要指标
1. **Total NMSE (dB)** - 总重建质量
2. **Static NMSE (dB)** - 静态分量重建
3. **Dynamic NMSE (dB)** - 动态分量重建
4. **Total Loss** - 训练损失

### 辅助指标
5. **静态-动态相关系数** - 分离质量
6. **时间变化率** - 时间约束效果
7. **训练时间** - 效率评估
8. **收敛速度** - 优化效果

---

## 🎯 预期结果总结

| 实验 | Total NMSE | Dynamic NMSE | Loss | 关键发现 |
|------|-----------|--------------|------|---------|
| **1. Baseline** | -25.0 dB | N/A | 0.007 | 🔵 基准性能 |
| **2. Full** | **-25.5+ dB** | **-22+ dB** | **< 0.01** | ✅ 最佳性能 |
| **3. No Temporal** | -25.0 dB | -20 dB | ~0.015 | ⚠️ 分离质量下降 |
| **4. No Attention** | -25.3 dB | -21 dB | ~0.012 | 🤔 影响较小 |
| **5. No Separation** | -25.2 dB | -20 dB | ~0.01 | ⚠️ 相关性高 |
| **6. Old Weights** | **-24.8 dB** | **-13 dB** | **0.3+** | ❌ 性能很差 |
| **7. No Reg** | -25.4 dB | -21 dB | ~0.008 | 🤔 可能过拟合 |
| **8. No Smooth** | -25.2 dB | -21 dB | ~0.012 | ⚠️ 静态不平滑 |

---

## 📈 如何分析结果

### 1. 对比完整模型（实验2）

每个消融实验都应与实验2对比：
```python
improvement = full_nmse - ablation_nmse
print(f"性能下降: {improvement:.2f} dB")
```

### 2. 关注关键差异

- **实验1 vs 2**: 证明分解的价值
- **实验2 vs 6**: 证明改进的价值
- **实验2 vs 3**: 证明时间约束的价值
- **实验2 vs 5**: 证明分离约束的价值

### 3. 可视化对比

使用TensorBoard查看：
- 训练曲线对比
- 重建效果可视化
- 时间变化曲线

---

## 🚀 使用方法

### 快速启动
```bash
# 赋予执行权限
chmod +x run_ablation_experiments.sh

# 启动所有实验
./run_ablation_experiments.sh
```

### 监控进度
```bash
# 查看所有日志
tail -f logs/ablation_*/gpu*.log

# 查看特定实验
tail -f logs/ablation_*/gpu1_full_improved.log

# GPU使用情况
watch -n 1 nvidia-smi

# TensorBoard可视化
tensorboard --logdir=./experiments1017
```

### 停止实验
```bash
# 查看所有训练进程
ps aux | grep train.py

# 停止特定GPU的任务
pkill -f "config_1_baseunet.yaml"

# 停止所有训练任务
pkill -f train.py
```

---

## 📊 结果收集

训练完成后，使用以下脚本收集结果：

```python
# collect_ablation_results.py
import os
import yaml
import pandas as pd
from pathlib import Path

results = []
exp_dir = Path('./experiments1017')

for exp in exp_dir.iterdir():
    if exp.is_dir() and 'Ablation' in exp.name:
        # 读取最佳checkpoint的指标
        best_metrics = ...  # 从checkpoint读取
        
        results.append({
            'experiment': exp.name,
            'total_nmse': best_metrics['total_nmse_db'],
            'dynamic_nmse': best_metrics.get('dynamic_nmse_db', 0),
            'loss': best_metrics['total_loss']
        })

df = pd.DataFrame(results)
df.to_csv('ablation_results.csv', index=False)
print(df)
```

---

## ⏱️ 时间规划

- **单个实验**: ~28小时（150 epochs）
- **并行运行**: ~28小时（8个实验同时）
- **建议检查点**: 
  - 10 epoch: 检查是否正常启动
  - 50 epoch: 中期评估
  - 150 epoch: 最终结果

---

## 📝 注意事项

1. **磁盘空间**: 每个实验约需5-10GB空间
2. **显存**: 每张卡需要~10GB显存
3. **日志**: 定期检查日志，确认无错误
4. **备份**: 重要checkpoint及时备份

---

## 🎓 论文写作建议

消融研究结果可以这样呈现：

### Table: Ablation Study Results

| Configuration | Total NMSE↑ | Dynamic NMSE↑ | Loss↓ | Δ from Full |
|--------------|-------------|---------------|-------|-------------|
| Baseline (no decomp.) | -25.0 | - | 0.007 | -0.5 dB |
| **Full (ours)** | **-25.5** | **-22.0** | **0.009** | - |
| w/o Temporal | -25.0 | -20.0 | 0.015 | -0.5 dB |
| w/o Attention | -25.3 | -21.0 | 0.012 | -0.2 dB |
| w/o Separation | -25.2 | -20.5 | 0.011 | -0.3 dB |
| Old Weights | -24.8 | -13.0 | 0.35 | -0.7 dB |
| w/o Regularization | -25.4 | -21.0 | 0.008 | -0.1 dB |
| w/o Static Smooth | -25.2 | -21.0 | 0.012 | -0.3 dB |

### Key Findings

1. 分解带来0.5dB提升（实验1 vs 2）
2. 损失权重调整至关重要（实验6显示）
3. 时间约束显著改善分离质量（实验3）
4. 正则化可以适度减弱（实验7）

---

**祝实验顺利！期待精彩的消融分析结果！** 🎉
