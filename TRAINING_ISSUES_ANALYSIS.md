# Training Issues Analysis - Epoch 0

**Date:** 2025-10-23
**Status:** Early training (Epoch 0/200)
**GPU:** A100 80GB (Google Colab Enterprise)

---

## 🔴 Critical Issues Found

### 1. **Normalization Mismatch (HIGHEST PRIORITY)**

**Problem:**
```
MRI range: [-1.88 to 4.65]  ← Z-score normalized (good)
CT range:  [0.01 to 0.76]   ← Min-max [0,1] (too compressed!)
Fused:     [-3.00 to 5.00]  ← Hard clipped
```

**Why this is bad:**
- MRI has ~6.5 dynamic range
- CT has only ~0.75 dynamic range (compressed 8-10×!)
- Model learns to ignore CT because MRI dominates the signal
- Bone PSNR = -4.5 dB → CT information is being lost

**Expected behavior:**
Both MRI and CT should be in similar ranges (e.g., both [-3, 3] or both [0, 1])

---

### 2. **Aggressive Output Clamping Killing Gradients**

**Problem:**
```python
# In roi_guided_diffusion.py line 370
x_t = torch.clamp(x_t, -3.0, 5.0)
```

**Why this is bad:**
- Fused output hits clamp boundaries: `[-3.000, 5.000]` (exactly at limits!)
- Gradients = 0 at clamp boundaries → no learning
- This is why SSIM = 0.0000 (no structure learned)

**Evidence:**
```
Fused shape: torch.Size([1, 1, 256, 256, 26]), range: [-3.000, 5.000]
                                                           ↑           ↑
                                                     hitting clamp!
```

---

### 3. **SSIM ≈ 0 (No Structural Learning)**

**Current metrics:**
```
brain_ssim: 0.0001  ← Essentially zero
bone_ssim:  0.0000  ← Completely zero
global_ssim: 0.0000 ← No structure at all
```

**Why:**
- SSIM requires structural correlation
- Random noise (Epoch 0) has no structure
- **Normal for Epoch 0**, but should improve by Epoch 10-20
- BUT: Normalization + clamping issues will prevent improvement

---

### 4. **Negative PSNR (Signal Weaker Than Noise)**

**Current:**
```
bone_psnr: -4.55 dB  ← Negative! MSE > signal²
```

**What this means:**
```
PSNR = 10 * log10(MAX² / MSE)

If PSNR < 0:
  → MAX² < MSE
  → Prediction error is larger than signal range
  → Model is worse than random guessing!
```

**Root cause:** CT normalization [0, 0.76] makes MAX too small

---

### 5. **Lesion Metrics Are Fake**

**Current:**
```
lesion_dice: 1.0000   ← Perfect!
lesion_nsd:  0.7778   ← Good
lesion_hd95: 22.22    ← Reasonable
```

**But these are fake because:**
```python
# In fusion_trainer.py line 401
lesion_pred=lesion_mask,  # Using GT as prediction!
```

This gives us upper bounds but doesn't reflect actual model performance.

---

## ✅ Solutions (Priority Order)

### 🔧 Fix #1: Normalize CT to Same Scale as MRI (CRITICAL)

**Current normalization:**
```python
# Data loader (somewhere)
mri = (mri - mean) / std  # Z-score → [-2, 5]
ct = (ct - min) / (max - min)  # Min-max → [0, 1]
```

**Fix:** Normalize CT with Z-score too
```python
# For CT
ct_mean = ct.mean()
ct_std = ct.std()
ct = (ct - ct_mean) / ct_std  # Now similar range to MRI
```

**Expected result:**
```
MRI: [-2, 5]  (6-7 dynamic range)
CT:  [-2, 3]  (4-5 dynamic range)
     ↑ Similar scales!
```

---

### 🔧 Fix #2: Remove or Soften Output Clamping

**Option A: Remove clamp entirely (recommended)**
```python
# In src/models/roi_guided_diffusion.py:370
# REMOVE THIS LINE:
# x_t = torch.clamp(x_t, -3.0, 5.0)

# No clamping during training!
```

**Option B: Use soft clamp (tanh)**
```python
# Soft clamp that preserves gradients
x_t = 4.0 * torch.tanh(x_t / 4.0)  # Smooth limit to [-4, 4]
```

**Why this works:**
- Gradients flow even at boundaries
- Model learns natural output range
- No artificial gradient killing

---

### 🔧 Fix #3: Add Intensity Clipping at Data Loading

**Problem from EDA:**
```
MRI max: 4547 (extreme outlier!)
CT min:  -1183 (artifact)
```

**Fix in data loader:**
```python
# Before normalization
mri = np.clip(mri, np.percentile(mri, 1), np.percentile(mri, 99))
ct = np.clip(ct, -1000, 1000)  # HU window

# Then normalize
mri = (mri - mri.mean()) / mri.std()
ct = (ct - ct.mean()) / ct.std()
```

---

### 🔧 Fix #4: Adjust Loss Weights

**Current stroke preset:**
```yaml
α (brain): 1.0
β (bone):  0.8
γ (lesion): 2.5
```

**Problem:** With CT compressed to [0, 0.76], bone loss is too weak

**Recommended adjustment:**
```yaml
α (brain): 1.0
β (bone):  1.2  ← Increase from 0.8
γ (lesion): 3.0  ← Increase from 2.5
```

**Rationale:**
- Bone needs more weight to compensate for CT compression
- Lesion needs more weight (61% are small <5mL)

---

### 🔧 Fix #5: Monitor Loss Components Separately

**Add to training log:**
```python
print(f"  Loss breakdown:")
print(f"    Brain SSIM loss: {brain_loss:.4f}")
print(f"    Bone SSIM loss:  {bone_loss:.4f}")
print(f"    Lesion loss:     {lesion_loss:.4f}")
print(f"    Diffusion loss:  {diff_loss:.4f}")
```

**Why:** Track which ROI is learning (or not learning)

---

## 📊 Expected Behavior After Fixes

### Epoch 0 (Current, with fixes):
```
brain_ssim: 0.00-0.05    (random baseline)
bone_psnr:  0-5 dB       (positive!)
bone_ssim:  0.00-0.03
lesion_dice: N/A         (no real prediction yet)
global_ssim: 0.00-0.02
```

### Epoch 10 (After some learning):
```
brain_ssim: 0.10-0.30    (structure emerging)
bone_psnr:  8-12 dB      (reasonable)
bone_ssim:  0.05-0.15
global_ssim: 0.05-0.15
```

### Epoch 50 (Convergence):
```
brain_ssim: 0.70-0.85    (good)
bone_psnr:  15-20 dB     (good)
bone_ssim:  0.50-0.70
lesion_dice: 0.60-0.75   (when real head added)
global_ssim: 0.60-0.75
```

---

## 🚨 Red Flags to Watch

**Training is broken if:**
1. ❌ SSIM stays at 0.0000 after 20 epochs
2. ❌ Bone PSNR stays negative after 10 epochs
3. ❌ Training loss doesn't decrease after 5 epochs
4. ❌ Fused output range stays clamped at [-3.000, 5.000]

**Training is working if:**
1. ✅ SSIM increases slowly (0.00 → 0.10 by epoch 20)
2. ✅ Bone PSNR becomes positive by epoch 10
3. ✅ Training loss decreases steadily
4. ✅ Fused output range shrinks naturally (e.g., [-2.5, 4.2])

---

## 🔨 Implementation Priority

**Do these in order:**

1. **Fix CT normalization** (1 hour)
   - File: `src/data/fusion_dataset.py`
   - Change min-max to Z-score
   - Test: Print CT range, should be ~[-2, 3]

2. **Remove output clamping** (5 minutes)
   - File: `src/models/roi_guided_diffusion.py:370`
   - Comment out or remove clamp line
   - Test: Fused range should vary naturally

3. **Add input clipping** (30 minutes)
   - File: `src/data/fusion_dataset.py`
   - Clip before normalization
   - Test: No extreme outliers in loaded data

4. **Adjust loss weights** (5 minutes)
   - File: `configs/cvpr2026/train_roi.yaml`
   - Update β=1.2, γ=3.0
   - Test: Monitor bone loss increases

5. **Restart training** (overnight)
   - Run for 50-100 epochs
   - Check metrics at epoch 10, 25, 50
   - Expect SSIM > 0.1 by epoch 20

---

## 📝 Quick Diagnosis Commands

**Check normalization in training:**
```python
# Add to trainer validation loop
print(f"MRI stats: mean={mri.mean():.3f}, std={mri.std():.3f}")
print(f"CT stats:  mean={ct.mean():.3f}, std={ct.std():.3f}")
```

**Expected after fix:**
```
MRI stats: mean=0.000, std=1.000  ← Z-score
CT stats:  mean=0.000, std=1.000  ← Z-score (not 0.3!)
```

**Check for clamp boundary hitting:**
```python
# After diffusion sampling
n_min_clamp = (fused == -3.0).sum()
n_max_clamp = (fused == 5.0).sum()
print(f"Clamped voxels: {n_min_clamp} min, {n_max_clamp} max")
```

**Expected after fix:**
```
Clamped voxels: 0 min, 0 max  ← No clamping!
```

---

## 🎯 Success Criteria

**Training is on track when:**

```
Epoch 20 metrics:
  brain_ssim > 0.15
  bone_psnr > 5 dB
  bone_ssim > 0.05
  loss decreasing steadily

Epoch 50 metrics:
  brain_ssim > 0.50
  bone_psnr > 12 dB
  bone_ssim > 0.30
  lesion_dice > 0.40 (with real head)
```

**If not meeting these, re-check:**
1. CT normalization (should be Z-score, not [0,1])
2. Output clamping (should be removed or very loose)
3. Loss weights (β should be ≥1.0 for bone)
4. Learning rate (should start at 1e-4, warmup to 5e-5)

---

## 📚 References

- EDA Results: `eda_results/INSIGHTS.md` (Section 9: Expected metrics)
- Proposal Algorithm 1: ROI-guided diffusion with gradient guidance
- Config: `configs/cvpr2026/train_roi.yaml`
- Model: `src/models/roi_guided_diffusion.py:370` (clamping line)
- Data: `src/data/fusion_dataset.py` (normalization)

---

**Next steps:** Fix normalization → Remove clamp → Restart training → Monitor for 20 epochs