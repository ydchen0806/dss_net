#!/bin/bash
# ====================================================================
# 消融实验启动脚本 - 前台后台并行版本（不使用nohup）
# 进程在后台运行，但attached到当前终端
# 日志输出到文件，位置与nohup版本一致
# ====================================================================

set -e

echo "=========================================="
echo "🚀 启动8个消融实验（前台后台并行）"
echo "=========================================="

# 切换到项目目录
cd /LSEM/user/chenyinda/code/signal_dy_static/1104 || exit

# 升级tensorboard
echo "📦 升级TensorBoard..."
pip install --upgrade tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple

# 创建时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 创建日志和结果目录
LOG_DIR="./logs_${TIMESTAMP}"
RESULTS_DIR="./results_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

echo "📝 日志目录: $LOG_DIR"
echo "📁 结果目录: $RESULTS_DIR"
echo ""

# 显示实验列表
echo "=========================================="
echo "📋 消融实验列表"
echo "=========================================="
echo "GPU 0: Baseline U-Net        -> ${RESULTS_DIR}/baseunet"
echo "GPU 1: 完整改进模型          -> ${RESULTS_DIR}/full"
echo "GPU 2: 无时间约束            -> ${RESULTS_DIR}/no_temporal"
echo "GPU 3: 无Attention           -> ${RESULTS_DIR}/no_attention"
echo "GPU 4: 无分离质量约束        -> ${RESULTS_DIR}/no_separation"
echo "GPU 5: 原始损失权重          -> ${RESULTS_DIR}/old_weights"
echo "GPU 6: 无正则化              -> ${RESULTS_DIR}/no_reg"
echo "GPU 7: 无静态平滑约束        -> ${RESULTS_DIR}/no_smooth"
echo "=========================================="
echo ""

# 为每个实验创建独立的结果目录
mkdir -p "${RESULTS_DIR}/baseunet"
mkdir -p "${RESULTS_DIR}/full"
mkdir -p "${RESULTS_DIR}/no_temporal"
mkdir -p "${RESULTS_DIR}/no_attention"
mkdir -p "${RESULTS_DIR}/no_separation"
mkdir -p "${RESULTS_DIR}/old_weights"
mkdir -p "${RESULTS_DIR}/no_reg"
mkdir -p "${RESULTS_DIR}/no_smooth"

# 启动训练函数
launch_experiment() {
    local gpu=$1
    local config=$2
    local exp_name=$3
    local output_dir="${RESULTS_DIR}/${exp_name}"
    local log_file="${LOG_DIR}/gpu${gpu}_${exp_name}.log"
    
    # 创建临时配置文件
    local temp_config="temp_${exp_name}_${TIMESTAMP}.yaml"
    sed "s|output_dir:.*|output_dir: \"${output_dir}\"|g" "$config" > "$temp_config"
    
    # 启动训练（使用 & 放入后台，日志重定向到文件）
    CUDA_VISIBLE_DEVICES=$gpu python train.py --config "$temp_config" --gpus 1 \
        > "$log_file" 2>&1 &
    
    local pid=$!
    echo "✅ GPU $gpu: $exp_name (PID: $pid)"
    
    # 将PID添加到数组
    PIDS+=($pid)
    EXP_NAMES+=($exp_name)
}

# 初始化PID数组
PIDS=()
EXP_NAMES=()

# 启动所有实验
echo "🚀 启动训练任务..."
echo ""

launch_experiment 0 "config_baseunet.yaml" "baseunet"
launch_experiment 1 "config_full.yaml" "full"
launch_experiment 2 "config_no_temporal.yaml" "no_temporal"
launch_experiment 3 "config_no_attention.yaml" "no_attention"
launch_experiment 4 "config_no_separation.yaml" "no_separation"
launch_experiment 5 "config_old_weights.yaml" "old_weights"
launch_experiment 6 "config_no_reg.yaml" "no_reg"
launch_experiment 7 "config_no_smooth.yaml" "no_smooth"

echo ""
echo "=========================================="
echo "✅ 所有训练任务已启动！"
echo "=========================================="
echo ""
echo "📝 进程ID列表:"
for i in "${!PIDS[@]}"; do
    printf "  GPU %d (%-15s): PID %d\n" "$i" "${EXP_NAMES[$i]}" "${PIDS[$i]}"
done
echo ""
echo "📂 目录结构:"
echo "  日志: $LOG_DIR/"
echo "    ├── gpu0_baseunet.log"
echo "    ├── gpu1_full.log"
echo "    ├── gpu2_no_temporal.log"
echo "    ├── gpu3_no_attention.log"
echo "    ├── gpu4_no_separation.log"
echo "    ├── gpu5_old_weights.log"
echo "    ├── gpu6_no_reg.log"
echo "    └── gpu7_no_smooth.log"
echo ""
echo "  结果: $RESULTS_DIR/"
echo "    ├── baseunet/"
echo "    ├── full/"
echo "    ├── no_temporal/"
echo "    ├── no_attention/"
echo "    ├── no_separation/"
echo "    ├── old_weights/"
echo "    ├── no_reg/"
echo "    └── no_smooth/"
echo ""
echo "📊 监控方式："
echo "  1. 查看单个日志: tail -f ${LOG_DIR}/gpu0_baseunet.log"
echo "  2. 查看所有日志: tail -f ${LOG_DIR}/*.log"
echo "  3. TensorBoard: tensorboard --logdir=${RESULTS_DIR}"
echo "  4. 进程状态: jobs -l"
echo "  5. 进程监控: ps aux | grep train.py"
echo "  6. GPU监控: watch -n 1 nvidia-smi"
echo ""
echo "💡 管理命令："
echo "  - 查看后台任务: jobs -l"
echo "  - 将任务调到前台: fg %N  (N是任务编号)"
echo "  - 停止单个训练: kill ${PIDS[0]}  (使用对应的PID)"
echo "  - 停止所有训练: kill ${PIDS[@]}"
echo "  - 或者: pkill -f train.py"
echo ""
echo "⚠️  注意："
echo "  - 这些进程attached到当前终端"
echo "  - 关闭终端会停止训练"
echo "  - 如需断开终端继续运行，请使用 nohup 版本"
echo ""
echo "⏰ 按 Ctrl+C 退出此脚本（不会停止训练）"
echo "⏰ 等待所有训练完成..."
echo ""

# 保存PID到文件，方便后续管理
PID_FILE="${LOG_DIR}/pids.txt"
echo "# Training PIDs - $(date)" > "$PID_FILE"
for i in "${!PIDS[@]}"; do
    echo "${PIDS[$i]} GPU$i ${EXP_NAMES[$i]}" >> "$PID_FILE"
done
echo "📝 PID已保存到: $PID_FILE"
echo ""

# 等待所有后台进程完成
wait

echo ""
echo "=========================================="
echo "🎉 所有训练任务已完成！"
echo "=========================================="
echo ""
echo "📊 结果收集："
echo "  python collect_ablation_results.py --base_dir ${RESULTS_DIR}"
echo ""
echo "🧹 清理临时配置文件："
echo "  rm -f temp_*_${TIMESTAMP}.yaml"
echo ""