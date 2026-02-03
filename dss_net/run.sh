#!/bin/bash
cd /LSEM/user/chenyinda/code/signal_dy_static/1104 || exit
pip install --upgrade tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
CUDA_VISIBLE_DEVICES=1 python train.py --config config_baseloss_wosmooth.yaml --gpus 1 &
CUDA_VISIBLE_DEVICES=2 python train.py --config config_baseloss_wotime.yaml --gpus 1 &
CUDA_VISIBLE_DEVICES=3 python train.py --config config_baseunet.yaml --gpus 1 &
CUDA_VISIBLE_DEVICES=4 python train.py --config config.yaml --gpus 1 &

wait  # 等待所有后台进程完成
echo "所有训练任务已结束 ✅"