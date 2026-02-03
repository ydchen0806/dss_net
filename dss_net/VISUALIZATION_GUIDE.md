# 🎨 TensorBoard 可视化使用指南

## 📊 概述

改进代码已完全集成 TensorBoard 可视化功能，训练时自动生成精美的对比图和分析图表。

---

## ✨ 可视化内容

### 1️⃣ 幅度对比图 (Magnitude_Comparison)

10列网格展示：

```
输入 | 预测静态 | 真值静态 | 静态误差 | 预测动态 | 真值动态 | 动态误差 | 预测总量 | 真值总量 | 总误差
```

**用途**：
- 直观对比模型预测和真值
- 快速定位重建问题区域
- 验证静态/动态分离效果

**颜色说明**：
- 🟦 信道幅度：viridis colormap（蓝→绿→黄）
- 🔥 误差热图：hot colormap（黑→红→白）

---

### 2️⃣ 相位对比图 (Phase_Comparison)

布局同幅度图，展示相位信息（-π 到 π）

**用途**：
- 检查相位重建质量
- 发现幅度图看不到的问题

**颜色说明**：
- 🌈 相位：twilight colormap（循环色）

---

### 3️⃣ 误差分布直方图 (Error_Histogram)

**分解模型**（3个子图）：
- 静态分量误差分布
- 动态分量误差分布
- 总重建误差分布

**用途**：
- 分析误差统计特性
- 判断模型是否过拟合
- 对比不同分量的重建质量

---

### 4️⃣ 时间变化曲线 (Temporal_Variation)

2×2网格展示：

```
┌─────────────────┬─────────────────┐
│ 静态时间曲线     │ 静态变化率       │
├─────────────────┼─────────────────┤
│ 动态时间曲线     │ 动态变化率       │
└─────────────────┴─────────────────┘
```

**用途**：
- 验证物理约束有效性
- 静态分量应该平滑（变化率低）
- 动态分量应该变化（变化率高）

---

## 🚀 使用方法

### 方法1：训练时自动生成

```bash
# 1. 确保配置中启用了可视化
# config.yaml:
# logging:
#   visualization:
#     enabled: true

# 2. 启动训练
python train.py --config config.yaml

# 3. 在另一个终端启动TensorBoard
tensorboard --logdir=./experiments1017

# 4. 浏览器访问
# http://localhost:6006
```

### 方法2：查看已有实验

```bash
# 查看特定实验
tensorboard --logdir=./experiments1017/ChannelDecomposition_UNetDecomposer_20241103_*/logs

# 对比多个实验
tensorboard --logdir_spec=\
  "Improved:./experiments1017/improved_*/logs,\
   Baseline:./experiments1017/baseline_*/logs"
```

---

## ⚙️ 配置选项

### 基本配置

```yaml
logging:
  visualization:
    enabled: true              # 是否启用
    interval: 5                # 每N个epoch记录一次
    num_samples: 4             # 展示样本数
    modes: ['magnitude', 'phase']  # 可视化模式
```

### 参数说明

| 参数 | 默认值 | 说明 | 建议 |
|------|--------|------|------|
| `enabled` | true | 主开关 | 训练时开启 |
| `interval` | 5 | 记录频率 | 5-10（太频繁影响速度） |
| `num_samples` | 4 | 样本数量 | 2-4（太多图片很大） |
| `modes` | ['magnitude', 'phase'] | 可视化类型 | 可只保留magnitude |

### 性能优化配置

#### 快速训练（减少开销）
```yaml
visualization:
  enabled: true
  interval: 10        # 降低频率
  num_samples: 2      # 减少样本
  modes: ['magnitude']  # 只保留幅度
```

#### 详细分析（完整可视化）
```yaml
visualization:
  enabled: true
  interval: 2         # 频繁记录
  num_samples: 8      # 更多样本
  modes: ['magnitude', 'phase']
```

#### 禁用可视化（纯训练）
```yaml
visualization:
  enabled: false      # 关闭
```

---

## 📊 TensorBoard 界面导航

### SCALARS 标签页

查看训练曲线：
```
Train/
  ├─ Loss
  ├─ Total_NMSE_dB
  ├─ Static_NMSE_dB
  └─ Dynamic_NMSE_dB

Val/
  ├─ Loss
  ├─ Total_NMSE_dB
  ├─ Static_NMSE_dB
  └─ Dynamic_NMSE_dB
```

### IMAGES 标签页 ⭐

查看可视化图像：
```
Visualization/
  ├─ Magnitude_Comparison  # 幅度对比
  ├─ Phase_Comparison      # 相位对比
  ├─ Error_Histogram       # 误差分布
  └─ Temporal_Variation    # 时间变化
```

**操作技巧**：
- 点击图片查看大图
- 使用时间滑块切换不同epoch
- 右键保存图片
- 使用对比功能查看多个实验

---

## 🔍 如何解读可视化

### 好的训练结果

#### 幅度对比图
✅ 预测图和真值图高度相似  
✅ 误差图整体较暗（误差小）  
✅ 静态和动态分量明显不同

#### 误差直方图
✅ 误差集中在0附近  
✅ 分布较窄（标准差小）  
✅ 没有明显的异常峰

#### 时间变化图
✅ 静态分量变化率低且平稳  
✅ 动态分量变化率高且波动  
✅ 预测曲线与真值曲线接近

---

### 需要改进的信号

#### 幅度对比图
⚠️ 误差图出现亮斑（局部误差大）  
⚠️ 预测与真值差异明显  
⚠️ 静态和动态分量过于相似

#### 误差直方图
⚠️ 误差分布宽且分散  
⚠️ 出现多个峰（不稳定）  
⚠️ 动态分量误差远大于静态

#### 时间变化图
⚠️ 静态分量变化率过高  
⚠️ 动态分量变化率过低  
⚠️ 预测与真值曲线偏离

---

## 💡 实用技巧

### 技巧1：对比不同epoch

在TensorBoard中：
1. 选择 IMAGES 标签
2. 使用底部的时间滑块
3. 观察从epoch 5 → 50 → 100的改善

**关注点**：
- 误差图是否变暗（误差减小）
- 预测是否更接近真值
- 分离质量是否提升

---

### 技巧2：对比不同实验

```bash
tensorboard --logdir_spec=\
  "Improved:./experiments/improved/logs,\
   Original:./experiments/original/logs,\
   Baseline:./experiments/baseline/logs"
```

在TensorBoard中：
- 切换不同实验查看对比
- 使用不同颜色区分

---

### 技巧3：导出高质量图片

用于论文/报告：
1. 在TensorBoard点击图片放大
2. 右键"另存为"或使用下载按钮
3. 保存为PNG格式
4. 分辨率足够高，可直接用于论文

---

### 技巧4：监控关键epoch

建议重点查看：
- **Epoch 5**: 初期效果
- **Epoch 20**: 是否在正确方向
- **Epoch 50**: 中期质量
- **Epoch 100+**: 最终效果

---

## 🔧 故障排查

### 问题1: TensorBoard不显示图像

**症状**：IMAGES标签为空

**排查**：
1. 确认 `visualization.enabled: true`
2. 检查是否已到达记录的epoch（如第5个）
3. 刷新TensorBoard页面
4. 查看训练日志是否有"Generating visualizations"

**解决**：
```bash
# 检查配置
python -c "
import yaml
config = yaml.safe_load(open('config.yaml'))
print(config['logging']['visualization'])
"
```

---

### 问题2: 可视化导致训练变慢

**症状**：每个epoch时间增加很多

**解决方案A**：降低频率
```yaml
visualization:
  interval: 10  # 从5改为10
```

**解决方案B**：减少样本
```yaml
visualization:
  num_samples: 2  # 从4改为2
```

**解决方案C**：只保留关键模式
```yaml
visualization:
  modes: ['magnitude']  # 只保留幅度
```

---

### 问题3: 图像显示异常

**症状**：图像全黑/全白/颜色异常

**原因**：数据归一化问题或有NaN值

**解决**：
```python
# 检查数据
import torch
from dataset import create_dataloaders
import yaml

config = yaml.safe_load(open('config.yaml'))
train_loader, _, _ = create_dataloaders(config)
batch = next(iter(train_loader))

print("Input range:", batch['input'].min(), batch['input'].max())
print("Has NaN:", torch.isnan(batch['input']).any())
```

---

### 问题4: 显存不足

**症状**：添加可视化后Out of Memory

**解决**：
```yaml
visualization:
  num_samples: 1  # 最小值
  
training:
  batch_size: 8   # 减小batch size
```

---

## 📈 性能影响

### 时间开销

| 配置 | 单次可视化时间 | 对总训练时间影响 |
|------|---------------|----------------|
| interval=5, samples=4 | 8-15秒 | ~3-5% |
| interval=10, samples=4 | 8-15秒 | ~1-2% |
| interval=5, samples=2 | 5-8秒 | ~2-3% |

### 磁盘占用

| 可视化类型 | 单次大小 | 150 epochs总计 |
|-----------|---------|---------------|
| Magnitude | ~3-5 MB | ~90-150 MB |
| Phase | ~3-5 MB | ~90-150 MB |
| Histogram | ~0.5 MB | ~15 MB |
| Temporal | ~1 MB | ~30 MB |
| **总计** | ~8-12 MB | **~250-350 MB** |

---

## ✨ 最佳实践

### 训练阶段策略

#### 🔄 探索阶段（前20 epochs）
```yaml
visualization:
  enabled: true
  interval: 2      # 频繁查看
  num_samples: 2
```
**目的**：快速发现问题

#### 🚀 稳定训练（20-80 epochs）
```yaml
visualization:
  enabled: true
  interval: 10     # 降低频率
  num_samples: 4
```
**目的**：减少开销，定期监控

#### 📊 最终阶段（80+ epochs）
```yaml
visualization:
  enabled: true
  interval: 5      # 增加频率
  num_samples: 8   # 更多样本
```
**目的**：详细分析最终效果

---

## 🎯 总结

### 核心价值

✅ **实时监控** - 不用等训练完成就能看效果  
✅ **直观对比** - 图像比数字更容易理解  
✅ **问题定位** - 快速发现哪里出错  
✅ **论文素材** - 高质量图片可直接用

### 使用建议

1. 训练时始终开启可视化
2. 根据资源调整interval和num_samples
3. 定期查看TensorBoard，不要只盯着loss
4. 对比不同实验的可视化结果
5. 导出关键图片用于分析和报告

---

**祝可视化帮助您更好地理解和改进模型！** 🎨✨
