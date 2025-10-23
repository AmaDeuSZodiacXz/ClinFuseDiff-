# Diffusion-Based Baseline Methods

Integration wrappers for pretrained diffusion fusion models.

---

## 📥 Downloading Pretrained Weights

```bash
# Run the download script
bash baselines/download_pretrained.sh
```

This will download:
- **Diff-IF**: From [GitHub](https://github.com/XunpengYi/Diff-IF)
- **TTTFusion**: From author's shared link (TBD)
- **SwinFusion**: From [GitHub](https://github.com/Linfeng-Tang/SwinFusion)

---

## 🔧 Directory Structure

```
diffusion_methods/
├── diff_if/
│   ├── wrapper.py          # Inference wrapper
│   ├── pretrained/         # Downloaded weights
│   │   └── model.pth
│   └── README.md
├── ttfusion/
│   ├── wrapper.py
│   ├── pretrained/
│   └── README.md
├── fusion_diff/
│   └── wrapper.py
├── tfs_diff/
│   └── wrapper.py
├── ddpm_emf/
│   └── wrapper.py
└── dm_fnet/
    └── wrapper.py
```

---

## 🚀 Usage

### Diff-IF

```bash
python baselines/evaluate_baseline.py \
    --method diff_if \
    --checkpoint baselines/diffusion_methods/diff_if/pretrained/model.pth \
    --data-dir data/apis/preproc \
    --split test \
    --output work/results/baselines/diff_if
```

### TTTFusion

```bash
python baselines/evaluate_baseline.py \
    --method ttfusion \
    --checkpoint baselines/diffusion_methods/ttfusion/pretrained/model.pth \
    --data-dir data/apis/preproc \
    --split test \
    --output work/results/baselines/ttfusion
```

---

## 📝 Implementation Status

| Method | Wrapper | Pretrained | Priority |
|--------|---------|------------|----------|
| Diff-IF | ✅ | 🔴 TODO | HIGH |
| TTTFusion | ✅ | 🔴 TODO | HIGH |
| FusionDiff | 🟡 Planned | 🔴 TODO | MEDIUM |
| TFS-Diff | 🟡 Planned | 🔴 TODO | LOW |
| DDPM-EMF | 🟡 Planned | 🔴 TODO | LOW |
| DM-FNet | 🟡 Planned | 🔴 TODO | LOW |

---

## 🔗 Original Repositories

- **Diff-IF**: https://github.com/XunpengYi/Diff-IF
- **TTTFusion**: https://arxiv.org/abs/2504.20362 (code TBD)
- **FusionDiff**: (code not publicly available)
- **TFS-Diff**: https://papers.miccai.org/miccai-2024/703-Paper3901.html
- **DDPM-EMF**: (code not available)
- **DM-FNet**: https://arxiv.org/abs/2506.15218