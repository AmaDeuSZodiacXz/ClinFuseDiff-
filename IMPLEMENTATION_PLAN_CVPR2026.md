# CLIN-FuseDiff++: Implementation Plan for CVPR 2026

## Project Pivot: From Feature Fusion to Image Fusion

### **Critical Change**
- **OLD (ICLR)**: Multimodal feature fusion for disease prediction (MRI features + CT features → prediction)
- **NEW (CVPR)**: Multimodal medical **image fusion** (MRI image + CT image → Fused image)

### **Key Differences**

| Aspect | ICLR Version (Current) | CVPR Version (Proposal) |
|--------|------------------------|-------------------------|
| **Task** | Disease prediction | Image fusion for visualization |
| **Input** | Image features (2048-d) | Full 3D images (H×W×D) |
| **Output** | Prediction (class/survival) | Fused image (H×W×D) |
| **Goal** | Missing modality robustness | ROI-specific fidelity + uncertainty |
| **Evaluation** | AUC, Accuracy, F1 | SSIM, FSIM, Dice, NSD, HD95 |
| **Conference** | ICLR (ML) | CVPR (Computer Vision) |

## Core Components to Implement

### 1. ROI-Aware Guided Diffusion (Algorithm 1)

**File**: `src/models/roi_guided_diffusion.py`

```python
class ROIGuidedDiffusion:
    """
    Implements Algorithm 1 from proposal:
    - Takes MRI, CT, ROI masks as input
    - Generates fused image through guided reverse diffusion
    - Uses ROI-specific guidance (brain, bone, lesion)
    """

    def forward_sample(self, x_t, t, c, roi_masks, alpha, beta, gamma):
        """
        Modified DDPM sampling with ROI guidance:
        x_{t-1} = g(x_t, t | c) - η∇L_ROI - η_u∇U
        """
```

**Key Features**:
- Conditioning on MRI, CT encodings
- ROI mask integration
- Composite loss gradients during sampling
- Uncertainty-modulated guidance

### 2. Clinical Composite ROI Loss

**File**: `src/training/roi_losses.py`

```python
class ClinicalROILoss:
    """
    Implements Equation 2:
    L_ROI = α·L_brain + β·L_bone + γ·L_les

    where:
    - L_brain = 1 - SSIM(F, IM | M_brain)
    - L_bone = 1 - SSIM(F, IC | M_bone)
    - L_les = λ1·Dice + λ2·NSD + λ3·HD95
    """
```

**Components**:
- **Brain Loss**: SSIM between fused image and MRI in brain ROI
- **Bone Loss**: SSIM between fused image and CT in bone ROI
- **Lesion Loss**: Boundary metrics (Dice, NSD@τmm, HD95)
- **Controllable weights**: (α, β, γ) for disease-specific presets

### 3. Lesion Segmentation Head

**File**: `src/models/lesion_head.py`

```python
class LesionSegmentationHead:
    """
    Frozen pre-trained head for lesion segmentation.
    Used to compute boundary-aware metrics during fusion.

    Input: Fused image F
    Output: Lesion probability map Ŝ(F)
    """
```

**Purpose**:
- Extract lesion boundaries from fused image
- Compare with ground truth for geometric fidelity
- Guide diffusion to preserve lesion shape

### 4. Uncertainty-Carrying Diffusion

**File**: `src/models/uncertainty.py`

```python
class UncertaintyModule:
    """
    Implements Equation 4:
    η_eff(p) = η · σ(κ(1 - conf(p)))
    conf(p) = 1 - uncert(p)

    Computes per-voxel uncertainty and modulates guidance.
    """

    def estimate_uncertainty(self, samples):
        """Ensemble-based epistemic + aleatoric uncertainty"""

    def calibrate(self, predictions, targets):
        """ECE/Brier score calibration"""
```

**Features**:
- Ensemble sampling for uncertainty estimation
- Spatial uncertainty maps
- Calibration metrics (ECE, Brier)
- Uncertainty-modulated guidance strength

### 5. Registration-Aware Robustness

**File**: `src/data/registration.py`

```python
class RegistrationAugmentation:
    """
    Applies plausible warp jitter to MRI/CT during training.
    Makes fusion robust to small misregistration errors.
    """

    def apply_warp_jitter(self, mri, ct, amplitude=2.0):
        """Small random deformations (mm-level)"""
```

**Augmentations**:
- Small random warps (1-3mm)
- Simulated registration errors
- Tolerance bands in boundary metrics

### 6. ROI-Aware Metrics Suite

**File**: `src/utils/roi_metrics.py`

```python
# Primary ROI Metrics
def roi_ssim(pred, target, roi_mask)
def roi_fsim(pred, target, roi_mask)
def roi_psnr(pred, target, roi_mask)

# Lesion Boundary Metrics
def dice_score(pred, target)
def normalized_surface_dice(pred, target, tolerance_mm=2.0)
def hausdorff_95(pred, target)

# Uncertainty Calibration
def expected_calibration_error(predictions, targets, confidences)
def brier_score(predictions, targets, probabilities)
```

### 7. Image Fusion Dataset

**File**: `src/data/fusion_dataset.py`

```python
class ImageFusionDataset(Dataset):
    """
    Loads paired MRI-CT images + ROI masks for fusion task.

    Returns:
    - MRI image (H×W×D)
    - CT image (H×W×D)
    - Brain ROI mask
    - Bone ROI mask
    - Lesion ROI mask
    - (Optional) Reference fused image
    """
```

**Directory Structure**:
```
data/fusion/
  patient_001/
    mri.nii.gz          # MRI image
    ct.nii.gz           # CT image
    brain_mask.nii.gz   # From TotalSegmentator
    bone_mask.nii.gz    # From TotalSegmentator
    lesion_mask.nii.gz  # Expert annotation
  patient_002/
    ...
```

### 8. Few-Step Sampling & Distillation

**File**: `src/models/fast_sampling.py`

```python
class FastDDIMSampler:
    """DDIM sampling with fewer steps (10-50 vs. 1000)"""

class ScoreDistillation:
    """
    Distill multi-step teacher into few-step student.
    Preserves quality while reducing inference time.
    """
```

## Updated Configuration

**File**: `configs/cvpr2026_config.yaml`

```yaml
# CLIN-FuseDiff++ Configuration

model:
  # Image-to-image diffusion (not feature fusion)
  type: "image_fusion"

  # Lightweight encoders for conditioning
  encoder:
    mri_encoder:
      type: "resnet18"  # Light encoder
      out_channels: 256
    ct_encoder:
      type: "resnet18"
      out_channels: 256

  # U-Net denoiser for images
  diffusion:
    architecture: "unet3d"
    in_channels: 1  # Fused image
    cond_channels: 512  # MRI + CT encodings
    num_timesteps: 1000
    beta_schedule: "cosine"

  # Lesion segmentation head (frozen)
  lesion_head:
    pretrained: "path/to/lesion_model.pth"
    frozen: true

# ROI guidance weights (Equation 2)
roi_guidance:
  alpha: 1.0      # Brain region weight
  beta: 1.0       # Bone region weight
  gamma: 2.0      # Lesion region weight (higher priority)

  # Lesion loss sub-weights (λ1, λ2, λ3)
  lesion_weights:
    dice: 1.0
    nsd: 1.0
    hd95: 0.5

  # Guidance strengths
  eta: 0.1        # ROI loss guidance
  eta_u: 0.05     # Uncertainty guidance

# Registration robustness
registration:
  warp_jitter: true
  amplitude_mm: 2.0
  tolerance_mm: 2.0  # For NSD metric

# Uncertainty estimation
uncertainty:
  enabled: true
  num_samples: 5  # Ensemble size
  calibration: ["ece", "brier"]
  modulate_guidance: true
  kappa: 0.5

# Efficiency
sampling:
  ddim_steps: 50  # Few-step sampling
  use_distillation: true

# Disease-specific presets
presets:
  brain_tumor:
    alpha: 1.5  # Emphasize brain tissue
    beta: 0.5
    gamma: 2.0
  bone_tumor:
    alpha: 0.5
    beta: 2.0  # Emphasize bone
    gamma: 2.0
  metastasis:
    alpha: 1.0
    beta: 1.0
    gamma: 3.0  # Strong lesion preservation
```

## Implementation Priority

### **Phase 1: Core Image Fusion (Weeks 1-2)**
1. ✅ Adapt diffusion model for image-level fusion (U-Net architecture)
2. ✅ Create ImageFusionDataset for MRI-CT pairs
3. ✅ Implement basic image fusion (without ROI guidance)
4. ✅ Test end-to-end: MRI + CT → Fused image

### **Phase 2: ROI-Aware Guidance (Weeks 3-4)**
5. ✅ Implement ROI mask generation with TotalSegmentator
6. ✅ Implement ROIGuidedDiffusion (Algorithm 1)
7. ✅ Implement ClinicalROILoss (brain, bone, lesion)
8. ✅ Test ROI-guided sampling

### **Phase 3: Lesion Boundary Preservation (Week 5)**
9. ✅ Train/obtain frozen LesionSegmentationHead
10. ✅ Implement boundary metrics (Dice, NSD, HD95)
11. ✅ Integrate lesion loss into guidance

### **Phase 4: Uncertainty & Robustness (Week 6)**
12. ✅ Implement UncertaintyModule
13. ✅ Add registration augmentation
14. ✅ Implement calibration metrics

### **Phase 5: Efficiency & Evaluation (Week 7)**
15. ✅ Implement DDIM few-step sampling
16. ✅ Implement score distillation
17. ✅ Implement full ROI metric suite

### **Phase 6: Experiments & Baselines (Weeks 8-10)**
18. ✅ Compare against state-of-art fusion methods
19. ✅ Ablation studies
20. ✅ Disease-specific preset experiments

## Key Architectural Changes

### **1. Replace Feature Encoders with U-Net**

**OLD**:
```python
# Feature-level fusion
encoder = ResNet50()  # → 2048-d features
fusion = DiffusionFusion(feat_dim=2048)
predictor = MLP([512, 256, num_classes])
```

**NEW**:
```python
# Image-level fusion
cond_encoder = LightResNet18()  # Condition on MRI/CT
denoiser = UNet3D(in_ch=1, cond_ch=512)  # Denoise fused image
output = F  # Fused image (H×W×D)
```

### **2. Replace Prediction Loss with ROI Losses**

**OLD**:
```python
loss = cross_entropy(pred, target)
```

**NEW**:
```python
loss = (alpha * ssim_brain_loss +
        beta * ssim_bone_loss +
        gamma * (dice + nsd + hd95))
```

### **3. Add Guided Sampling Loop**

**NEW**:
```python
for t in reversed(range(T)):
    # Standard denoising
    x_pred = denoiser(x_t, t, cond)

    # Compute ROI losses
    F = decode(x_pred)
    L_roi = compute_roi_loss(F, MRI, CT, masks)

    # Guided update
    x_{t-1} = x_pred - eta * grad(L_roi) - eta_u * grad(U)
```

## Migration Strategy

### **Option A: Clean Slate (Recommended)**
- Create new branch `cvpr2026`
- Keep existing `main` for ICLR version
- Implement CVPR version separately
- Both versions use shared utils (TotalSegmentator, data loading)

### **Option B: Gradual Refactor**
- Extend existing diffusion to support image fusion mode
- Add ROI guidance as optional module
- Single codebase with mode flags

## Next Immediate Actions

1. **Decision**: Choose Option A or B
2. **Create image fusion U-Net model**
3. **Update dataset to return full images (not features)**
4. **Implement ROI loss functions**
5. **Implement guided sampling loop**

This plan transforms the project from feature-level multimodal fusion (ICLR) to image-level ROI-aware fusion (CVPR 2026).
