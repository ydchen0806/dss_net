#!/bin/bash
# ====================================================================
# 消融实验启动脚本 - 后台运行版本（使用nohup）
# ====================================================================

set -e

echo "=========================================="
echo "🚀 启动8个消融实验（后台运行）"
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

# 启动训练（后台运行）
echo "🚀 启动训练任务..."

# 临时修改config文件的output_dir并启动训练
launch_experiment() {
    local gpu=$1
    local config=$2
    local exp_name=$3
    local output_dir="${RESULTS_DIR}/${exp_name}"
    local log_file="${LOG_DIR}/gpu${gpu}_${exp_name}.log"
    
    # 创建临时配置文件
    local temp_config="temp_${exp_name}_${TIMESTAMP}.yaml"
    sed "s|output_dir:.*|output_dir: \"${output_dir}\"|g" "$config" > "$temp_config"
    
    # 启动训练
    CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --config "$temp_config" --gpus 1 \
        > "$log_file" 2>&1 &
    
    echo "✅ GPU $gpu: $exp_name (PID: $!, Config: $temp_config)"
}

# 启动所有实验
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
echo "📂 目录结构:"
echo "  日志: $LOG_DIR/"
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
echo "  4. 进程监控: ps aux | grep train.py"
echo "  5. GPU监控: watch -n 1 nvidia-smi"
echo ""
echo "⏳ 预计完成时间: $(date -d '+28 hours' '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "💡 停止所有训练: pkill -f train.py"
echo "💡 清理临时配置: rm -f temp_*_${TIMESTAMP}.yaml"
echo ""