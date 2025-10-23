# Baseline Methods for CLIN-FuseDiff++ Evaluation

This directory contains baseline fusion methods for comparison with CLIN-FuseDiff++ in the CVPR 2026 paper.

---

## 📂 Directory Structure

```
baselines/
├── README.md                      # This file
├── simple_methods/                # Simple fusion baselines
│   ├── average.py                 # Simple average fusion
│   ├── weighted_average.py        # Weighted average fusion
│   └── laplacian_pyramid.py       # Multi-scale pyramid fusion
├── diffusion_methods/             # Diffusion-based baselines
│   ├── ttfusion/                  # TTTFusion (arXiv 2025)
│   ├── diff_if/                   # Diff-IF (Information Fusion 2024)
│   ├── fusion_diff/               # FusionDiff (CVIU 2024)
│   ├── tfs_diff/                  # TFS-Diff (MICCAI 2024)
│   ├── ddpm_emf/                  # DDPM-EMF (JOSA A 2025)
│   └── dm_fnet/                   # DM-FNet (arXiv 2025)
├── neural_methods/                # Non-diffusion neural baselines
│   ├── swinfusion/                # SwinFusion (2022)
│   └── u2fusion/                  # U2Fusion (TPAMI 2020)
├── evaluate_baseline.py           # Unified evaluation script
└── compare_all.py                 # Generate comparison tables/plots
```

---

## 🎯 Baseline Methods

### A. Simple Methods (No Training)

| Method | Type | Description |
|--------|------|-------------|
| **Simple Average** | Pixel-wise | `F = (MRI + CT) / 2` |
| **Weighted Average** | ROI-adaptive | `F = α·MRI + β·CT` (optimized per ROI) |
| **Laplacian Pyramid** | Multi-scale | Frequency-domain fusion |

### B. Diffusion-Based Methods

| Method | Year | Venue | Code | Focus |
|--------|------|-------|------|-------|
| **TTTFusion** | 2025 | arXiv | [Link](https://arxiv.org/abs/2504.20362) | Test-time training, medical, speed |
| **Diff-IF** | 2024 | Info Fusion | [GitHub](https://github.com/XunpengYi/Diff-IF) | Knowledge prior, no GT needed |
| **FusionDiff** | 2024 | CVIU | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1077314224000924) | End-to-end probabilistic |
| **TFS-Diff** | 2024 | MICCAI | [Paper](https://papers.miccai.org/miccai-2024/703-Paper3901.html) | Tri-modal + super-resolution |
| **DDPM-EMF** | 2025 | JOSA A | [PubMed](https://pubmed.ncbi.nlm.nih.gov/40793553/) | CT-MRI specific |
| **DM-FNet** | 2025 | arXiv | [GitHub](https://arxiv.org/abs/2506.15218) | Two-stage diffusion |

### C. Non-Diffusion Neural Methods

| Method | Year | Venue | Code | Architecture |
|--------|------|-------|------|--------------|
| **SwinFusion** | 2022 | IEEE/CAA JAS | [GitHub](https://github.com/Linfeng-Tang/SwinFusion) | Transformer (Swin) |
| **U2Fusion** | 2020 | TPAMI | [Paper](https://dl.acm.org/doi/abs/10.1109/TPAMI.2020.3012548) | Unsupervised CNN |

---

## 🚀 Quick Start

### 1. Run Simple Baselines

```bash
# Simple average
python baselines/simple_methods/average.py \
    --data-dir data/apis/preproc \
    --split test \
    --output work/results/baselines/average

# Weighted average (optimized)
python baselines/simple_methods/weighted_average.py \
    --data-dir data/apis/preproc \
    --split test \
    --optimize-weights \
    --output work/results/baselines/weighted
```

### 2. Run Pretrained Diffusion Baselines

```bash
# Diff-IF (with pretrained model)
python baselines/evaluate_baseline.py \
    --method diff_if \
    --checkpoint baselines/diffusion_methods/diff_if/pretrained.pth \
    --data-dir data/apis/preproc \
    --split test \
    --output work/results/baselines/diff_if

# TTTFusion
python baselines/evaluate_baseline.py \
    --method ttfusion \
    --checkpoint baselines/diffusion_methods/ttfusion/pretrained.pth \
    --data-dir data/apis/preproc \
    --split test \
    --output work/results/baselines/ttfusion
```

### 3. Run Neural Baselines

```bash
# SwinFusion
python baselines/evaluate_baseline.py \
    --method swinfusion \
    --checkpoint baselines/neural_methods/swinfusion/pretrained.pth \
    --data-dir data/apis/preproc \
    --split test \
    --output work/results/baselines/swinfusion
```

### 4. Compare All Methods

```bash
# Generate comparison tables and plots
python baselines/compare_all.py \
    --results-dir work/results \
    --output work/results/comparison_cvpr2026.pdf
```

---

## 📊 Evaluation Metrics (Unified)

All baselines are evaluated with **the same metrics** for fair comparison:

### Primary ROI Metrics
- **Lesion**: Dice, NSD@2mm, HD95
- **Brain**: SSIM, FSIM
- **Bone**: PSNR, SSIM

### Global Metrics
- PSNR, SSIM, FSIM, FMI, EN (Entropy)

### Uncertainty Metrics (CLIN-FuseDiff++ only)
- ECE, Brier Score

---

## 📝 Citation

When using these baselines, please cite the original papers:

```bibtex
@article{ttfusion2025,
  title={TTTFusion: A Test-Time Training-Based Strategy for Multimodal Medical Image Fusion},
  journal={arXiv preprint arXiv:2504.20362},
  year={2025}
}

@article{diff_if2024,
  title={Diff-IF: Multi-modality image fusion via diffusion model with fusion knowledge prior},
  journal={Information Fusion},
  year={2024}
}

@article{swinfusion2022,
  title={SwinFusion: Cross-domain long-range learning for general image fusion via swin transformer},
  journal={IEEE/CAA Journal of Automatica Sinica},
  year={2022}
}

@article{u2fusion2020,
  title={U2Fusion: A unified unsupervised image fusion network},
  journal={IEEE TPAMI},
  year={2020}
}
```

---

## 🔧 Implementation Status

| Method | Status | Priority | Notes |
|--------|--------|----------|-------|
| Simple Average | ✅ Ready | HIGH | Trivial baseline |
| Weighted Average | ✅ Ready | HIGH | Optimized per-ROI weights |
| Laplacian Pyramid | 🟡 Planned | MEDIUM | Classic multi-scale |
| TTTFusion | 🔴 TODO | HIGH | Medical-specific, test-time |
| Diff-IF | 🔴 TODO | HIGH | Official code available |
| FusionDiff | 🔴 TODO | MEDIUM | Diffusion family |
| SwinFusion | 🔴 TODO | HIGH | Transformer baseline |
| U2Fusion | 🔴 TODO | MEDIUM | Unsupervised baseline |
| TFS-Diff | 🟡 Planned | LOW | Tri-modal (overkill for CT-MRI) |
| DDPM-EMF | 🟡 Planned | LOW | Similar to Diff-IF |
| DM-FNet | 🟡 Planned | LOW | Two-stage (complex) |

**Priority for CVPR 2026:**
1. **Must-have**: Simple Average, Weighted Average, TTTFusion, Diff-IF, SwinFusion
2. **Nice-to-have**: FusionDiff, U2Fusion
3. **Optional**: TFS-Diff, DDPM-EMF, DM-FNet, Laplacian Pyramid

---

## 📧 Contact

For questions about baseline implementations, see main repository README.