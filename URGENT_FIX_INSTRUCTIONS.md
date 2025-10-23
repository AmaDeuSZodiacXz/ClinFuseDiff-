# 🚨 URGENT FIX - Training Stagnant at Epoch 5

**Status:** Training is NOT learning (metrics unchanged from Epoch 0→5)
**Root Cause:** CT normalization mismatch
**Time to fix:** 5 minutes
**Impact:** CRITICAL - training cannot proceed without this

---

## 🔴 Confirmed Issues

### Issue #1: CT Normalization (CRITICAL)

**Location:** [`src/data/fusion_dataset.py:134-140`](src/data/fusion_dataset.py#L134-L140)

**Current code (WRONG):**
```python
def _normalize_ct(self, ct_volume: np.ndarray) -> np.ndarray:
    """Normalize CT intensity (Hounsfield Units)"""
    ct_volume = np.clip(ct_volume, -1000, 3000)
    ct_volume = (ct_volume + 1000) / 4000.0  # ← PROBLEM: [0, 1] range
    return ct_volume.astype(np.float32)
```

**Result:**
```
MRI: [-1.88, 4.65]  (6.5 dynamic range via Z-score)
CT:  [0.01, 0.76]   (0.75 dynamic range - compressed 8×!)
```

**Why this breaks training:**
- CT signal is 8-10× weaker than MRI signal
- Model ignores CT → Bone PSNR = -4.5 dB (negative!)
- No bone information learned

---

### Issue #2: Output Clamping

**Location:** [`src/models/roi_guided_diffusion.py:370`](src/models/roi_guided_diffusion.py#L370)

**Current code:**
```python
x_t = torch.clamp(x_t, -3.0, 5.0)
```

**Result:**
```
Fused output: [-3.000, 5.000]  ← Hitting clamp boundaries exactly!
```

**Why this breaks training:**
- Gradients = 0 at clamp boundaries
- SSIM = 0.0000 (no structure learning)
- Model cannot adjust output range

---

## ✅ FIXES (Apply in order)

### Fix #1: Change CT Normalization to Z-score (MUST DO)

**Edit:** [`src/data/fusion_dataset.py:134-140`](src/data/fusion_dataset.py#L134-L140)

**Replace with:**
```python
def _normalize_ct(self, ct_volume: np.ndarray) -> np.ndarray:
    """Normalize CT intensity (Hounsfield Units) - Z-score like MRI"""
    # Clip outliers
    ct_volume = np.clip(ct_volume, -1000, 1000)

    # Z-score normalization (same as MRI)
    mask = ct_volume > -900  # Exclude air/background
    if mask.sum() > 0:
        mean = ct_volume[mask].mean()
        std = ct_volume[mask].std()
        if std > 0:
            ct_volume = (ct_volume - mean) / std
        else:
            ct_volume = ct_volume - mean

    return ct_volume.astype(np.float32)
```

**Expected result after fix:**
```
MRI: [-2, 5]   (Z-score)
CT:  [-2, 3]   (Z-score, similar scale!)
```

---

### Fix #2: Remove Output Clamping

**Edit:** [`src/models/roi_guided_diffusion.py:370`](src/models/roi_guided_diffusion.py#L370)

**Comment out the line:**
```python
# REMOVED: Killing gradients
# x_t = torch.clamp(x_t, -3.0, 5.0)
```

**Or use soft clamp (preserves gradients):**
```python
# Soft clamp with tanh (gradients preserved)
x_t = 4.0 * torch.tanh(x_t / 4.0)
```

**Expected result after fix:**
```
Fused output: [-2.5, 4.2]  ← Natural range, not clamped!
```

---

### Fix #3: Adjust Loss Weights (Optional but recommended)

**Edit:** [`configs/cvpr2026/train_roi.yaml`](configs/cvpr2026/train_roi.yaml)

**Change:**
```yaml
roi_guidance:
  stroke:
    alpha: 1.0   # Brain (keep)
    beta: 1.2    # Bone (increase from 0.8)
    gamma: 3.0   # Lesion (increase from 2.5)
```

**Rationale:**
- β=1.2: More weight on bone (CT information)
- γ=3.0: More weight on lesions (61% are small <5mL)

---

## 🚀 How to Apply Fixes

### Step 1: Apply CT Normalization Fix

```bash
# Edit the file
code src/data/fusion_dataset.py

# Or use sed (Linux/Mac):
# (backup first!)
cp src/data/fusion_dataset.py src/data/fusion_dataset.py.backup
```

Then manually replace the `_normalize_ct` function (lines 134-140) with the new version above.

### Step 2: Remove Output Clamping

```bash
# Edit the file
code src/models/roi_guided_diffusion.py

# Go to line 370 and comment out:
# x_t = torch.clamp(x_t, -3.0, 5.0)
```

### Step 3: Update Config (Optional)

```bash
code configs/cvpr2026/train_roi.yaml

# Update beta and gamma values
```

### Step 4: Restart Training

```bash
# Stop current training (Ctrl+C)

# Delete old checkpoints to start fresh
rm -rf experiments/*/checkpoints/*

# Restart training
python train.py --config configs/cvpr2026/train_roi.yaml --preset stroke --no-wandb
```

---

## 📊 Expected Results After Fixes

### Immediately (Epoch 0 with fixes):
```
CT range:       [-2, 3]       ← Similar to MRI now!
MRI range:      [-2, 5]
Fused range:    [-2.5, 4.5]   ← Not clamped!
bone_psnr:      0-5 dB        ← POSITIVE now!
```

### After 10 Epochs:
```
brain_ssim:  0.15-0.30    (was 0.0002)
bone_psnr:   8-12 dB      (was -4.5)
bone_ssim:   0.05-0.15    (was 0.0000)
loss:        1.5-2.0       (was 2.7)
```

### After 50 Epochs (convergence):
```
brain_ssim:  0.70-0.85
bone_psnr:   15-20 dB
bone_ssim:   0.50-0.70
lesion_dice: 0.60-0.75 (when real head added)
```

---

## 🚨 Red Flags - Stop if you see these

**After 10 epochs with fixes, if:**
- ❌ bone_psnr still negative
- ❌ brain_ssim still < 0.05
- ❌ Fused range still [-3.000, 5.000] (clamped)
- ❌ CT range still [0.0, 0.8] (not Z-score)

**→ Something went wrong. Check:**
1. Did you save the file after editing?
2. Did you restart training (not resume)?
3. Did Python reload the module?

---

## ✅ Success Indicators

**You'll know it's working when (Epoch 0-1):**
```
Debug output shows:
  CT range: [-2.xxx, 3.xxx]      ✅ Z-score working!
  Fused range: NOT [-3.000, 5.000]  ✅ Clamp removed!
```

**Epoch 10:**
```
  bone_psnr > 5 dB               ✅ CT learning!
  brain_ssim > 0.10              ✅ Structure learning!
  Train loss < 2.0               ✅ Convergence!
```

---

## 📞 If You Need Help

**Check these files:**
1. [TRAINING_ISSUES_ANALYSIS.md](TRAINING_ISSUES_ANALYSIS.md) - Full problem analysis
2. [eda_results/INSIGHTS.md](eda_results/INSIGHTS.md) - Expected metrics at each epoch

**Debug commands:**
```python
# In Python, load a sample and check normalization:
from src.data.fusion_dataset import ImageFusionDataset
dataset = ImageFusionDataset("data/apis/preproc", normalize=True)
sample = dataset[0]
print(f"MRI: [{sample['mri'].min():.3f}, {sample['mri'].max():.3f}]")
print(f"CT:  [{sample['ct'].min():.3f}, {sample['ct'].max():.3f}]")

# Should show:
# MRI: [-2.xxx, 5.xxx]
# CT:  [-2.xxx, 3.xxx]  ← Similar scales!
```

---

**TIME ESTIMATE:**
- Fix CT norm: 2 minutes
- Remove clamp: 1 minute
- Update config: 1 minute
- Restart training: 1 minute
- **Total: 5 minutes**

**PRIORITY: URGENT - Do this NOW before training more epochs!**