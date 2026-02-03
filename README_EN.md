<p align="center">
  <a href="README.md">🇨🇳 中文</a> | <a href="README_EN.md">🇬🇧 English</a>
</p>

# DSS-Net: Dynamic-Static Separation Networks for UWA Channel Denoising

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.10+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <a href="https://huggingface.co/cyd0806/dss_net_checkpoints"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Checkpoints-yellow.svg" alt="HuggingFace"></a>
</p>

## 📌 Introduction

This project implements **DSS-Net (Dynamic-Static Separation Networks)**, a physics-inspired deep learning framework for underwater acoustic (UWA) channel denoising. The method decomposes the channel into static and dynamic components, combined with physics-constrained loss function design, significantly improving channel estimation accuracy.

## 🏗️ Method Framework

![DSS-Net Architecture](dss_net_architecture.png)

DSS-Net employs a **dual-decoder U-Net architecture**, with the core idea of decomposing noisy channels into:
- **Static Component**: From stable propagation paths (direct path, seabed reflection), characterized by sparsity and temporal stability
- **Dynamic Component**: From time-varying sea surface reflections, characterized by low-rank properties and rapid temporal variation

### 💡 Key Innovations

1. **Dynamic-Static Decomposition Architecture**: Shared encoder + dual symmetric decoders for explicit separation of two components
2. **Physics-Informed Loss Function**:
   - L1 sparsity constraint (static component)
   - Nuclear norm low-rank constraint (dynamic component)
   - Temporal correlation prior
   - Separation quality metric
3. **SE Attention Mechanism**: Squeeze-and-Excitation module for enhanced feature selection

---

## 📊 Performance

### Simulation Data (Ray-Tracing)

| Method | NMSE (dB) | Improvement |
|--------|-----------|-------------|
| No Processing | -20.41 | - |
| U-Net Baseline | -23.49 | +3.08 |
| **DSS-Net (Ours)** | **-25.27** | **+4.86** |

### Sea Trial Data (Fuxian Lake)

| Depth | Input Power | Output Power | Power Reduction | Static Ratio | Dynamic Ratio |
|-------|-------------|--------------|-----------------|--------------|---------------|
| 5m | 2.35 dB | -0.03 dB | 2.38 dB | 69.3% | 23.8% |
| 7m | 3.60 dB | 2.22 dB | 1.38 dB | 65.3% | 24.6% |
| 9m | 2.72 dB | 1.42 dB | 1.30 dB | 52.8% | 29.0% |

> **⚠️ Important Note**: Sea trial data lacks Ground Truth, so **true NMSE or SNR improvement cannot be computed**. The table above only reports objective power changes; static/dynamic ratios reflect the learned channel decomposition characteristics.

**🔬 Physical Law Verification**: Increased depth → higher dynamic component (sea surface reflection) ratio, consistent with acoustic propagation principles.

---

## 📁 Project Structure

```
signal_dy_static/
├── dss_net/                       # Core code directory
│   ├── model.py                   # DSS-Net model definition
│   ├── loss.py                    # Physics-informed loss functions
│   ├── dataset.py                 # Data loader
│   ├── train.py                   # Training script
│   ├── eval.py                    # Evaluation script
│   ├── process_sea_trial.py       # Sea trial data processing
│   ├── config.yaml                # Main configuration file
│   └── results_20251104_092511/   # Experiment results
│
├── paper/                         # IEEE paper files
│   ├── bare_jrnl_new_sample4.tex  # Paper LaTeX source
│   └── figs/                      # Paper figures
│
├── sea_trial_data/                # Fuxian Lake sea trial data
│   ├── 484_5m_01_LS.mat           # 5m depth raw data
│   ├── 484_7m_01_LS.mat           # 7m depth raw data
│   ├── 484_9m_01_LS.mat           # 9m depth raw data
│   └── compare/                   # Processed comparison results
│
├── data_utils/                    # Data preprocessing utilities
├── dss_net_architecture.png       # Method framework diagram
└── README.md
```

---

## 🚀 Quick Start

### Requirements

```bash
pip install torch numpy scipy matplotlib pyyaml tqdm pandas
```

### 📦 Pretrained Model Download

Model files are hosted on Hugging Face: 🤗 **[cyd0806/dss_net_checkpoints](https://huggingface.co/cyd0806/dss_net_checkpoints)**

| Model | File | Size | NMSE |
|-------|------|------|------|
| **DSS-Net (Full)** | `dss_net_full_best.pth` | 499MB | -25.27 dB |
| Baseline U-Net | `baseline_unet_best.pth` | 355MB | -20.41 dB |

**Download Methods:**

```bash
# Using huggingface-cli
pip install huggingface_hub
huggingface-cli download cyd0806/dss_net_checkpoints dss_net_full_best.pth --local-dir ./checkpoints

# Or direct download
wget https://huggingface.co/cyd0806/dss_net_checkpoints/resolve/main/dss_net_full_best.pth
```

### 🏋️ Training

```bash
cd dss_net

# Single GPU training
python train.py --config config.yaml

# Multi-GPU training (DDP)
torchrun --nproc_per_node=4 train.py --config config.yaml
```

### 🔍 Processing Sea Trial Data

```bash
cd dss_net
python process_sea_trial.py
```

Output files are saved in `sea_trial_data/compare/`:
- `*_processed.mat`: Contains `est_h_original` (before denoising) and `est_h_denoised` (after denoising)
- `compare_*.png`: Visualization comparisons

---

## 📐 Model Details

### Input/Output

- **Input**: Noisy channel `H_noise` ∈ ℂ^(M×N) → Real representation [real, imag] ∈ ℝ^(2×M×N)
- **Output**:
  - `H_static`: Static component
  - `H_dynamic`: Dynamic component  
  - `H_total = H_static + H_dynamic`: Denoised channel

### Key Configuration

```yaml
model:
  name: "UNetDecomposer"
  base_channels: 64
  depth: 4
  use_attention: true

loss:
  weights:
    static_mse: 1.0
    dynamic_mse: 2.0      # Dynamic component is harder to estimate
    total_mse: 3.0        # Overall reconstruction is most important
  sparsity_lambda: 0.0001
  nuclear_lambda: 0.0001
  separation_weight: 0.05
```

---

## 📖 Citation

```bibtex
@article{yang2025dssnet,
  title={DSS-Net: Dynamic--Static Separation Networks for Physics-Inspired UWA Channel Denoising},
  author={Yang, Xiaoyu and Chen, Yinda and Tong, Feng and Zhou, Yuehai},
  journal={IEEE Transactions on Wireless Communications},
  year={2025}
}
```

---

## 📧 Contact

- **Xiaoyu Yang**: xiaoyuyang@stu.xmu.edu.cn (Channel modeling, sea trial validation)
- **Yinda Chen**: yindachen@mail.ustc.edu.cn (Algorithm design, code implementation)

---

## 📄 License

MIT License

