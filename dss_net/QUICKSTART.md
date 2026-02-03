# 🚀 快速开始 - 改进代码使用指南

## ⚡ 30秒快速部署

```bash
# 1. 备份原文件
cd /your/project/directory
mkdir backup
cp loss.py model.py config.yaml train.py backup/

# 2. 使用改进版本（包含可视化）
cp /path/to/improved/loss.py ./
cp /path/to/improved/model.py ./
cp /path/to/improved/config.yaml ./
cp /path/to/improved/train.py ./
cp /path/to/improved/visualization.py ./  # 🆕 可视化模块

# 3. 开始训练（可视化自动启用）
python train.py --config config.yaml

# 4. 🆕 启动TensorBoard查看效果
tensorboard --logdir=./experiments1017
# 浏览器打开 http://localhost:6006
```

---

## 📊 三种使用策略

### 策略1️⃣: 完整改进（推荐）

**适合**: 有充足时间和计算资源

```bash
# 直接使用所有改进
cp improved/*.py ./
python train.py --config config.yaml --gpus 1
```

**预期时间**: ~28小时（150 epochs）  
**预期效果**: Total NMSE < -25 dB，优于baseline

---

### 策略2️⃣: 渐进验证（稳妥）

**适合**: 想确认每步改进效果

#### 步骤1: 只改损失（最关键）
```bash
cp improved/loss.py ./
# 手动修改config.yaml中的损失权重
python train.py --config config.yaml
```

**验证点**: 10个epoch后
- ✅ Loss应该 < 0.1（原来0.3+）
- ✅ Total NMSE应该改善

#### 步骤2: 增加模型容量
```bash
cp improved/model.py ./
cp improved/train.py ./
python train.py --config config.yaml
```

**验证点**: 20个epoch后
- ✅ Dynamic NMSE应该 > -15 dB（原来-12.81 dB）

---

### 策略3️⃣: 最小改动（快速测试）

**适合**: 资源有限，快速验证

只修改config.yaml中的关键参数：

```yaml
loss:
  weights:
    dynamic_mse: 2.0         # 从1.0改为2.0
    total_mse: 3.0           # 从2.0改为3.0
    static_l1: 0.01          # 从0.1改为0.01
    dynamic_nuclear: 0.01    # 从0.1改为0.01
  
  sparsity_lambda: 0.0001    # 从0.001改为0.0001
  nuclear_lambda: 0.0001     # 从0.001改为0.0001

training:
  learning_rate: 1.0e-3      # 从5e-4改为1e-3
  epochs: 100                # 保持不变（快速测试）
```

**预期时间**: ~17小时（100 epochs）  
**预期效果**: 已经能看到明显改善

---

## 🎯 监控训练进度

### 关键指标

训练时关注这些指标：

```python
✅ 好的迹象:
- Train Loss 快速下降: 0.3 → 0.1 → 0.01
- Total NMSE: -24.8 → -25.0 → -25.5 dB
- Dynamic NMSE: -12.8 → -18.0 → -22.0 dB

⚠️ 需要调整:
- Loss卡在0.1以上不动 → 学习率太低
- Loss震荡很大 → 学习率太高或batch size太小
- Dynamic NMSE没改善 → 检查模型是否正确加载
```

### TensorBoard监控

```bash
tensorboard --logdir=./experiments1017

# 在浏览器查看 http://localhost:6006

# 🆕 可视化监控：
# 1. SCALARS标签 - 查看损失曲线和NMSE指标
# 2. IMAGES标签 - 查看重建效果可视化
#    - Magnitude_Comparison: 幅度对比
#    - Phase_Comparison: 相位对比
#    - Error_Histogram: 误差分布
#    - Temporal_Variation: 时间变化（验证约束效果）
```

### 🆕 可视化解读

**好的可视化特征**：
- 误差图（红色）整体较暗
- 预测与真值高度相似
- 动态分量逐渐清晰（不再像噪声）
- 时间变化图符合预期（静态平滑，动态变化）

详见 **VISUALIZATION.md** 获取完整解读指南

---

## 🔧 故障排查

### 问题1: ImportError相关错误

```bash
# 检查导入
python -c "from loss import ChannelDecompositionLoss; print('OK')"
python -c "from model import UNetDecomposer; print('OK')"
```

**解决**: 确保文件在正确位置

---

### 问题2: CUDA Out of Memory

**症状**: 显存不足

**解决方案A**: 减小batch size
```yaml
training:
  batch_size: 8    # 从16改为8
```

**解决方案B**: 禁用attention
```yaml
model:
  use_attention: false
```

---

### 问题3: Loss不下降

**可能原因**:
1. 学习率太低
2. 数据加载有问题
3. 模型初始化问题

**检查步骤**:
```bash
# 1. 验证数据
python -c "
from dataset import create_dataloaders
import yaml
config = yaml.safe_load(open('config.yaml'))
train_loader, _, _ = create_dataloaders(config)
batch = next(iter(train_loader))
print('Data shape:', batch['input'].shape)
"

# 2. 测试前向传播
python -c "
import torch
from model import UNetDecomposer
model = UNetDecomposer(use_attention=True)
x = torch.randn(2, 2, 100, 150)
out = model(x)
print('Static:', out['static'].shape)
print('Dynamic:', out['dynamic'].shape)
"
```

---

### 问题4: Dynamic NMSE仍然很差

**尝试**:
1. 增加dynamic_mse权重到3.0或更高
```yaml
loss:
  weights:
    dynamic_mse: 3.0  # 进一步加大
```

2. 增加模型容量
```yaml
model:
  base_channels: 128  # 从64增加到128
```

3. 降低regularization
```yaml
loss:
  weights:
    dynamic_nuclear: 0.001  # 进一步降低
```

---

## 📈 预期训练曲线

### 理想情况

```
Epoch | Train Loss | Val Loss | Total NMSE | Dynamic NMSE
------|-----------|----------|------------|-------------
1     | 0.35      | 0.38     | -23.5 dB   | -10.2 dB
10    | 0.08      | 0.10     | -24.2 dB   | -15.8 dB
20    | 0.03      | 0.04     | -24.8 dB   | -18.5 dB
50    | 0.008     | 0.012    | -25.3 dB   | -21.2 dB
100   | 0.005     | 0.008    | -25.8 dB   | -23.5 dB
150   | 0.004     | 0.007    | -26.1 dB   | -24.8 dB
```

### 判断标准

**20 epoch时的检查点**:
- ✅ Total Loss < 0.05 → 训练正常
- ⚠️ Total Loss > 0.1 → 可能需要调整
- ❌ Total Loss > 0.2 → 有问题，检查配置

**50 epoch时的检查点**:
- ✅ Total NMSE < -25 dB → 已超越baseline
- ✅ Dynamic NMSE < -20 dB → 分解有效
- ⚠️ Dynamic NMSE < -15 dB → 还需改善

---

## 💾 检查点管理

### 保存最佳模型

```bash
# 训练会自动保存到:
experiments1017/ChannelDecomposition_UNetDecomposer_*/checkpoints/

# 文件:
best.pth       # 验证集上最好的模型
latest.pth     # 最新的checkpoint
epoch_*.pth    # 每N个epoch的快照
```

### 恢复训练

```bash
# 如果训练中断，可以从checkpoint恢复
python train.py --config config.yaml --resume experiments1017/.../checkpoints/latest.pth
```

---

## 📊 评估改进效果

### 训练完成后

```bash
# 评估最佳模型
python eval.py --checkpoint experiments1017/.../checkpoints/best.pth

# 对比baseline
python eval.py --checkpoint baseline_model/best.pth
```

### 生成对比报告

```python
# compare_results.py
import pandas as pd

# 读取结果
baseline = pd.read_csv('baseline_results.csv')
improved = pd.read_csv('improved_results.csv')

# 对比
print("Total NMSE:")
print(f"  Baseline: {baseline['total_nmse_db'].values[0]:.2f} dB")
print(f"  Improved: {improved['total_nmse_db'].values[0]:.2f} dB")
print(f"  Gain: {improved['total_nmse_db'].values[0] - baseline['total_nmse_db'].values[0]:.2f} dB")

print("\nDynamic NMSE:")
print(f"  Baseline: N/A")
print(f"  Original: -12.81 dB")
print(f"  Improved: {improved['dynamic_nmse_db'].values[0]:.2f} dB")
print(f"  Gain: {improved['dynamic_nmse_db'].values[0] - (-12.81):.2f} dB")
```

---

## 🎉 成功标准

训练成功的标志：

### 必须达到（核心）
- ✅ Total NMSE ≥ -25 dB（至少持平baseline）
- ✅ Total Loss < 0.01
- ✅ Dynamic NMSE > -20 dB（从-12.81大幅提升）

### 期望达到（理想）
- 🎯 Total NMSE > -25.5 dB（超越baseline）
- 🎯 Dynamic NMSE > -22 dB
- 🎯 Static NMSE > -24 dB

### 完美达到（最佳）
- 🌟 Total NMSE > -26 dB
- 🌟 Dynamic NMSE > -24 dB
- 🌟 分离质量好（静态和动态确实不同）

---

## 🔄 迭代优化

如果效果还不理想，按此顺序调整：

1. **损失权重**: dynamic_mse继续加大（2.0 → 3.0 → 5.0）
2. **学习率**: 如果收敛慢，提高到2e-3
3. **模型容量**: base_channels增加到128
4. **训练时间**: epochs延长到200
5. **正则化**: 完全禁用（lambda设为0）

---

## 📞 需要帮助?

### 日志分析

如果遇到问题，提供以下信息：
1. 训练日志的最后50行
2. 第10、20、50 epoch的指标
3. TensorBoard截图（Loss曲线）
4. GPU内存使用情况

### 配置检查

```bash
# 验证配置文件
python -c "
import yaml
config = yaml.safe_load(open('config.yaml'))
print('Dynamic MSE weight:', config['loss']['weights']['dynamic_mse'])
print('Learning rate:', config['training']['learning_rate'])
print('Use attention:', config['model'].get('use_attention', False))
"
```

---

## ✨ 预期时间线

```
Day 1:
- 部署改进代码: 15分钟
- 启动训练: 5分钟
- 等待初步结果: 2-3小时

Day 2-3:
- 持续训练: 24-48小时
- 定期检查进度: 每8小时
- 中期评估: 在50 epoch时

Day 4:
- 训练完成
- 评估结果
- 对比分析

Total: ~3-4天（大部分时间是自动训练）
```

---

**祝训练顺利！期待看到改进后的优异结果！** 🚀
