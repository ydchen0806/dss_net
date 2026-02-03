# ⚡ 消融实验快速启动指南

## 🎯 5分钟快速部署

### 步骤1: 准备配置文件（1分钟）

将8个配置文件放到项目根目录：

```bash
cd /LSEM/user/chenyinda/code/signal_dy_static

# 确认文件都在
ls -1 config_*.yaml
# 应该看到:
# config_1_baseunet.yaml
# config_2_full_improved.yaml
# config_3_no_temporal.yaml
# config_4_no_attention.yaml
# config_5_no_separation.yaml
# config_6_old_weights.yaml
# config_7_no_regularization.yaml
# config_8_no_static_smooth.yaml
```

### 步骤2: 启动训练（1分钟）

```bash
# 赋予执行权限
chmod +x run_ablation_experiments.sh

# 启动所有实验
./run_ablation_experiments.sh
```

按提示输入 `y` 确认启动。

### 步骤3: 验证启动（1分钟）

```bash
# 检查所有训练进程
ps aux | grep train.py | grep -v grep
# 应该看到8个进程

# 检查GPU使用情况
nvidia-smi
# 应该看到8张卡都在使用
```

### 步骤4: 启动监控（2分钟）

```bash
# 终端1: 实时查看一个实验的日志
tail -f logs/ablation_*/gpu1_full_improved.log

# 终端2: 启动TensorBoard
tensorboard --logdir=./experiments1017 --port=6006
# 浏览器访问: http://localhost:6006
```

---

## 📊 监控指南

### 实时监控命令

```bash
# 1. 查看所有进程状态
watch -n 5 'ps aux | grep train.py | grep -v grep | wc -l'

# 2. GPU实时监控
watch -n 1 nvidia-smi

# 3. 查看特定实验日志
tail -f logs/ablation_*/gpu0_baseunet.log     # Baseline
tail -f logs/ablation_*/gpu1_full_improved.log # Full
tail -f logs/ablation_*/gpu6_old_weights.log   # Old weights

# 4. 查看所有实验的最新Loss
grep "Train Loss:" logs/ablation_*/gpu*.log | tail -n 8
```

### 关键检查点

#### 第1小时（Epoch 1-5）
- ✅ 所有进程正常运行
- ✅ Loss开始下降
- ✅ GPU利用率>80%

**检查命令:**
```bash
# 查看前5个epoch的Loss
grep "Epoch [1-5]" logs/ablation_*/gpu*.log | grep "Train Loss"
```

#### 第12小时（Epoch 50）
- ✅ Full模型Loss < 0.02
- ✅ Baseline Loss < 0.01
- ⚠️ Old Weights Loss可能仍>0.1

**检查命令:**
```bash
# 查看50 epoch的结果
grep "Epoch 50" logs/ablation_*/gpu*.log | grep "Val Loss"
```

#### 第28小时（完成）
- ✅ 所有实验收敛
- ✅ 生成checkpoint

---

## 🚨 故障排查

### 问题1: 进程启动失败

**症状:** `ps aux | grep train.py` 看不到8个进程

**解决:**
```bash
# 查看错误日志
cat logs/ablation_*/gpu*.log | grep -i error

# 常见问题:
# 1. 配置文件路径错误
ls -l config_*.yaml

# 2. Python环境问题
which python
python --version

# 3. CUDA问题
nvidia-smi
```

### 问题2: GPU显存不足

**症状:** 日志显示 "CUDA out of memory"

**解决:**
```bash
# 方法1: 减小batch size (在配置文件中)
# training.batch_size: 192 -> 8

# 方法2: 禁用某些实验的attention
# model.use_attention: true -> false

# 方法3: 关闭部分实验
pkill -f "config_4_no_attention.yaml"
```

### 问题3: 训练速度慢

**症状:** 每个epoch超过15分钟

**检查:**
```bash
# 1. GPU利用率
nvidia-smi

# 2. 数据加载
# 检查日志中的数据加载时间

# 3. 网络IO
iostat -x 1
```

**解决:**
```bash
# 增加data workers (在配置文件中)
# data.num_workers: 4 -> 8
```

---

## 🎯 预期时间线

| 时间 | 里程碑 | 检查内容 |
|------|--------|---------|
| **0h** | 启动 | 8个进程运行 |
| **1h** | 初步验证 | Loss开始下降 |
| **3h** | 稳定训练 | 无错误，GPU稳定 |
| **12h** | 中期检查 | Epoch 50，初步对比 |
| **28h** | 完成 | 生成所有checkpoints |
| **29h** | 结果分析 | 运行分析脚本 |

---

## 📈 中期评估（Epoch 50）

在训练到50 epoch时，可以先评估一次：

```bash
# 收集中期结果
python collect_ablation_results.py

# 查看中期对比
cat ablation_results.csv

# TensorBoard对比曲线
# 浏览器查看各实验的Loss曲线
```

**预期中期结果:**
- Full模型应该明显优于其他
- Old Weights仍然表现很差
- Baseline已经收敛

---

## 🛑 如何停止实验

### 停止单个实验

```bash
# 查找进程PID
ps aux | grep config_1_baseunet.yaml

# 停止该进程
kill <PID>
```

### 停止所有实验

```bash
# 方法1: 优雅停止
pkill -SIGTERM -f train.py

# 方法2: 强制停止
pkill -9 -f train.py

# 验证
ps aux | grep train.py | grep -v grep
# 应该没有输出
```

---

## 📊 结果收集（训练完成后）

### 自动收集

```bash
# 运行分析脚本
python collect_ablation_results.py

# 生成的文件:
# - ablation_results.csv (数据表)
# - ablation_table.tex (LaTeX表格)
# - ablation_analysis/loss_comparison.png (对比图)
# - ablation_analysis/config_heatmap.png (配置热力图)
```

### 手动查看

```bash
# 查看所有best checkpoints
find experiments1017 -name "best.pth" -ls

# 对比文件大小
du -sh experiments1017/Ablation*

# 查看TensorBoard
tensorboard --logdir=./experiments1017
```

---

## 📋 检查清单

训练前：
- [ ] 8个配置文件已就位
- [ ] 数据路径正确
- [ ] 有足够磁盘空间（~80GB）
- [ ] 8张GPU都可用

训练中：
- [ ] 每小时检查一次进程状态
- [ ] 监控GPU温度（<85°C）
- [ ] 查看Loss是否正常下降
- [ ] 定期查看TensorBoard

训练后：
- [ ] 运行结果收集脚本
- [ ] 备份重要checkpoints
- [ ] 生成对比报告
- [ ] 写论文/报告

---

## 💡 高级技巧

### 1. 优先级调整

如果想让某个实验优先级更高：

```bash
# 降低其他实验的优先级
renice +10 $(pgrep -f "config_3_no_temporal.yaml")

# 提高Full模型的优先级  
renice -5 $(pgrep -f "config_2_full_improved.yaml")
```

### 2. 远程监控

```bash
# SSH端口转发（本地电脑运行）
ssh -L 6006:localhost:6006 user@server

# 然后本地浏览器访问
# http://localhost:6006
```

### 3. 邮件通知

在脚本中添加：

```bash
# 训练完成后发邮件
echo "Experiments finished!" | mail -s "Training Complete" your@email.com
```

---

## 🎓 论文使用建议

### 实验部分

```markdown
### Ablation Study

We conduct comprehensive ablation studies to validate 
the effectiveness of each component. Table X shows the 
results of 8 experiments on 8 GPUs:

1. **Baseline**: Direct reconstruction without decomposition
2. **Full Model**: Our complete approach (best performance)
3-8. **Component Ablations**: Removing one component each
```

### 结果表格

使用生成的 `ablation_table.tex` 直接插入论文。

---

**准备好了吗？开始吧！** 🚀

```bash
./run_ablation_experiments.sh
```
