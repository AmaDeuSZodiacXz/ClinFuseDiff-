# CLAUDE.md

This brief keeps coding agents aligned with the current ClinFuseDiff roadmap.

---

## Project Snapshot (2025-10-21)
- **Scope**: Diffusion-based multimodal **image fusion** (MRI + CT → fused volume) with ROI-aware guidance, registration robustness, and calibrated uncertainty, targeting CVPR 2026 (primary) with ICLR 2026 as fallback.
- **Status**: ~60 % implementation complete. Core architecture, ROI guidance, lesion head, training/evaluation scripts, and documentation are all in place (`PROGRESS_UPDATE.md:3-58`). Remaining work centers on data acquisition, registration-aware augmentation, uncertainty calibration polish, and experiments.
- **Orientation**: Start with `👉_START_HERE_👈.txt`, `START_HERE.md`, `QUICKSTART_CVPR2026.md`, and `agent.md` for the operative plan, dataset priorities, and open questions.

---

## Architecture Overview

### Conditioning & Denoiser (`src/models`)
- `unet3d.py`: 3D UNet denoiser with timestep embeddings, cross-attention blocks, and lightweight conditioning encoders for MRI/CT.
- `roi_guided_diffusion.py`: Implements Algorithm 1 (CVPR proposal) — ROI-aware reverse diffusion with gradient guidance terms for brain, bone, and lesion regions, plus hooks for uncertainty-modulated step sizes.
- `lesion_head.py`: Pluggable lesion segmentation head (simple UNet + pretrained weights support) used inline at sampling step 7 (`Ŝ ← LesionHead(F)`).

### Training & Losses (`src/training`)
- `fusion_trainer.py`: Full training loop for image-to-image diffusion with mixed precision, gradient accumulation, schedulers, checkpointing, and ROI metric tracking.
- `roi_losses.py`: Equation 2 composite ROI loss  
  `L_roi = α·(1 - SSIM(F, MRI|M_brain)) + β·(1 - SSIM(F, CT|M_bone)) + γ·(λ₁ Dice + λ₂ NSD_τ + λ₃ HD95)`  
  ready for disease-specific presets and used both during training and guided sampling.

### Evaluation & Metrics
- `evaluate.py`: Batch inference with ensemble sampling for uncertainty, ROI/global metrics, CSV/JSON exports, and optional fused-image/uncertainty dumps.
- `src/utils/roi_metrics.py`: Implements ROI-first scorecard (Dice/NSD/HD95, SSIM/FSIM, PSNR) plus calibration metrics (ECE, Brier).

### Config & Entrypoints
- `configs/cvpr2026/train_roi.yaml`: Canonical training configuration (sampling steps, ROI weights, presets, augmentation flags).
- `train.py`: CLI for training with preset overrides, resume, and WandB integration.
- `evaluate.py`: See above. All commands expect preprocessed data and ROI masks as described in the workflow docs.

---

## Data & Preprocessing Pipeline
- **Primary dataset**: APIS (paired NCCT + MRI/ADC with expert lesion masks). Manual download after challenge registration (see `docs/DATASET_SETUP.md` and `QUICKSTART_CVPR2026.md`).
- **Registration robustness**: SynthRAD2023 (180 multi-center brain CT-MRI pairs) for registration tolerance stress testing (see `docs/SYNTHRAD_DATASET.md`).
- **Scripts** (`scripts/`):
  - `register_ants.sh`: Wraps ANTs SyN registration (CT→MRI).
  - `make_masks_totalseg.py`: Runs TotalSegmentator to produce brain/bone ROI masks.
  - `make_splits.py`: Deterministic train/val/test splits.
  - `setup_env.sh`, `download_*.sh`: Environment/dataset automation.
- **Dataset loader**: `src/data/fusion_dataset.py` expects preprocessed volumes in a consistent directory layout (`data/<dataset>/preproc/<case_id>/…`) with accompanying ROI masks.

---

## Current Gaps & Priorities
1. **Environment/data bring-up**: Run `bash workflow/01_setup_environment.sh`, download APIS, and preprocess initial cases (registration + masks) to unblock training.
2. **Registration-aware robustness**: Implement jitter/warp augmentation and tolerance-aware evaluation harness per Proposal §3.3.
3. **Uncertainty calibration**: Validate ensemble-based uncertainty, finalize ECE/Brier computation, and add clinician-facing visualization overlays (Proposal §3.4).
4. **Efficiency**: Integrate DDIM/few-step sampling and score distillation (Proposal §3.5) once baseline fusion is stable.
5. **Experiments**: Establish baselines (e.g., deterministic fusion, TTTFusion), ablation matrix, and documentation for CVPR results.

Consult `agent.md` for detailed open questions and LeFusion reference insights.

---

## Development Guidelines
- Follow the ROI-centric workflow: data preprocessing → ROI mask generation → training → evaluation. Log every transform (registration matrices, resampling details) under `work/`.
- Use the provided configs rather than hardcoding hyperparameters. Introduce new settings through YAML + CLI flags when needed.
- Maintain ROI metric focus; any new modules should expose brain/bone/lesion behavior explicitly.
- Reuse MONAI utilities for boundary metrics (NSD, HD95) to stay consistent with clinical tolerance definitions.
- Before large changes, sync with the latest status docs (`PROGRESS_UPDATE.md`, `CURRENT_STATUS.md`) to avoid reviving deprecated TTT feature-level code.

---

### Quick Start for New Agents
1. Read `👉_START_HERE_👈.txt` and `START_HERE.md` to confirm environment expectations (Miniconda/PyTorch, TotalSegmentator, ANTs).
2. Review `QUICKSTART_CVPR2026.md` for the end-to-end workflow.
3. Check `PROGRESS_UPDATE.md` for milestones and `agent.md` for proposal-aligned context.
4. Coordinate with the active work log before modifying configs or scripts; document notable decisions in `PROGRESS_UPDATE.md` or a new progress note.

Good luck, and keep the ROI guidance front and centre. 🚀
