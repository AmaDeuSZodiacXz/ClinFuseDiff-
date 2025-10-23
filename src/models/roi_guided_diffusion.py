"""ROI-Aware Guided Diffusion (Algorithm 1 from CVPR 2026 Proposal)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion import GaussianDiffusion
from .unet3d import ImageFusionDiffusion


class ROIGuidedDiffusion(nn.Module):
    """
    Implements Algorithm 1: ROI-aware guided reverse diffusion

    Key features:
    - Guided sampling with ROI-specific losses
    - Brain ROI: SSIM with MRI
    - Bone ROI: SSIM with CT
    - Lesion ROI: Boundary metrics (Dice, NSD, HD95)
    - Uncertainty-modulated guidance
    """

    def __init__(
        self,
        image_fusion_model: ImageFusionDiffusion,
        num_timesteps=1000,
        beta_schedule='cosine',
        # ROI guidance weights (α, β, γ from Eq. 2)
        alpha=1.0,  # Brain weight
        beta=1.0,   # Bone weight
        gamma=2.0,  # Lesion weight
        # Lesion loss sub-weights (λ1, λ2, λ3)
        lambda_dice=1.0,
        lambda_nsd=1.0,
        lambda_hd95=0.5,
        # Guidance strengths
        eta=0.1,      # ROI guidance strength
        eta_u=0.05,   # Uncertainty guidance strength
        # Uncertainty modulation
        use_uncertainty_modulation=True,
        kappa=0.5
    ):
        super().__init__()

        self.model = image_fusion_model
        self.num_timesteps = num_timesteps

        # Setup noise schedule
        self._setup_schedule(beta_schedule, num_timesteps)

        # ROI guidance parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_dice = lambda_dice
        self.lambda_nsd = lambda_nsd
        self.lambda_hd95 = lambda_hd95

        # Guidance strengths
        self.eta = eta
        self.eta_u = eta_u

        # Uncertainty modulation
        self.use_uncertainty_modulation = use_uncertainty_modulation
        self.kappa = kappa

    def _setup_schedule(self, schedule, timesteps):
        """Setup diffusion schedule"""
        if schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, timesteps)
        elif schedule == 'cosine':
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * 3.14159 * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: add noise to image"""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(x_0.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_roi_loss(
        self,
        fused,
        mri,
        ct,
        brain_mask,
        bone_mask,
        lesion_mask=None,
        lesion_head=None
    ):
        """
        Compute composite ROI loss (Equation 2)

        L_ROI = α·L_brain + β·L_bone + γ·L_les

        Args:
            fused: Fused image (B, 1, D, H, W)
            mri: MRI reference (B, 1, D, H, W)
            ct: CT reference (B, 1, D, H, W)
            brain_mask: Brain ROI mask (B, 1, D, H, W)
            bone_mask: Bone ROI mask (B, 1, D, H, W)
            lesion_mask: Lesion ground truth (B, 1, D, H, W)
            lesion_head: Lesion segmentation network (frozen)

        Returns:
            Total ROI loss
        """
        loss = 0.0

        # L_brain: 1 - SSIM(F, MRI | M_brain)
        if brain_mask is not None and mri is not None:
            l_brain = 1 - self._masked_ssim(fused, mri, brain_mask)
            loss += self.alpha * l_brain

        # L_bone: 1 - SSIM(F, CT | M_bone)
        if bone_mask is not None and ct is not None:
            l_bone = 1 - self._masked_ssim(fused, ct, bone_mask)
            loss += self.beta * l_bone

        # L_les: λ1·Dice + λ2·NSD + λ3·HD95
        if lesion_mask is not None and lesion_head is not None:
            with torch.no_grad():
                lesion_pred = lesion_head(fused)  # (B, 1, D, H, W)

            l_lesion = self._compute_lesion_loss(lesion_pred, lesion_mask)
            loss += self.gamma * l_lesion

        return loss

    def _masked_ssim(self, pred, target, mask, window_size=11):
        """
        Compute SSIM within ROI mask (correct masking).

        Notes:
        - Use only ROI voxels for statistics (divide by mask.sum()).
        - Previous version averaged over the whole volume → gradients vanished
          and guidance became ineffective, yielding noisy samples.
        """
        # Ensure same shape
        assert pred.shape == target.shape == mask.shape, "Shapes must match"

        # ROI voxel count
        n = mask.sum() + 1e-8

        # Means on ROI
        mu_pred = (pred * mask).sum() / n
        mu_target = (target * mask).sum() / n

        # Centered values only inside ROI
        pred_c = (pred - mu_pred) * mask
        target_c = (target - mu_target) * mask

        # Variances and covariance on ROI
        sigma_pred = (pred_c ** 2).sum() / n
        sigma_target = (target_c ** 2).sum() / n
        sigma_pred_target = (pred_c * target_c).sum() / n

        # SSIM (scalar)
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2) + 1e-8)

        return ssim.clamp(0, 1)

    def _compute_lesion_loss(self, pred, target):
        """
        Compute lesion boundary loss:
        L_les = λ1·Dice + λ2·NSD + λ3·HD95

        For now, using Dice only (NSD and HD95 require more complex computation)
        """
        # Dice loss
        pred_binary = (pred > 0.5).float()
        target_binary = (target > 0.5).float()

        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum()

        dice = (2.0 * intersection + 1e-8) / (union + 1e-8)
        dice_loss = 1 - dice

        # TODO: Implement NSD and HD95
        nsd_loss = 0.0
        hd95_loss = 0.0

        total_lesion_loss = (
            self.lambda_dice * dice_loss +
            self.lambda_nsd * nsd_loss +
            self.lambda_hd95 * hd95_loss
        )

        return total_lesion_loss

    def _estimate_uncertainty(self, x_t, t, mri, ct):
        """
        Estimate per-voxel uncertainty

        Simple implementation: variance of predictions with dropout
        """
        # Enable dropout for uncertainty estimation
        self.model.train()

        samples = []
        num_samples = 5

        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.model(x_t, t, mri, ct)
                samples.append(pred)

        # Compute variance across samples
        samples = torch.stack(samples, dim=0)
        uncertainty = samples.var(dim=0)

        self.model.eval()

        return uncertainty

    def _modulate_guidance_by_uncertainty(self, eta, uncertainty):
        """
        Modulate guidance strength by uncertainty (Equation 4)

        η_eff(p) = η · σ(κ(1 - conf(p)))
        conf(p) = 1 - uncert(p)
        """
        # Normalize uncertainty to [0, 1]
        uncert_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)

        # Confidence = 1 - uncertainty
        conf = 1 - uncert_norm

        # Modulate: higher uncertainty → stronger guidance
        eta_eff = eta * torch.sigmoid(self.kappa * (1 - conf))

        return eta_eff

    @torch.no_grad()
    def sample(
        self,
        mri,
        ct,
        brain_mask,
        bone_mask,
        lesion_mask=None,
        lesion_head=None,
        sampling_timesteps=None,
        return_intermediates=False,
        verbose=False
    ):
        """
        Algorithm 1: ROI-aware guided reverse diffusion (inference)

        Args:
            mri: MRI image (B, 1, D, H, W)
            ct: CT image (B, 1, D, H, W)
            brain_mask: Brain ROI mask (B, 1, D, H, W)
            bone_mask: Bone ROI mask (B, 1, D, H, W)
            lesion_mask: Optional lesion mask (B, 1, D, H, W)
            lesion_head: Optional lesion segmentation network
            sampling_timesteps: Number of DDIM steps (default: use all timesteps)
            return_intermediates: Return intermediate denoising steps
            verbose: Print per-step progress

        Returns:
            Fused image F (B, 1, D, H, W)
        """
        batch_size = mri.shape[0]
        device = mri.device

        # DDIM: subsample timesteps
        if sampling_timesteps is None:
            sampling_timesteps = self.num_timesteps

        # Create DDIM timestep schedule
        step_size = self.num_timesteps // sampling_timesteps
        timesteps = list(range(0, self.num_timesteps, step_size))[:sampling_timesteps]
        timesteps = sorted(timesteps, reverse=True)

        if verbose:
            print(f"    DDIM sampling: {len(timesteps)} steps (subsampled from {self.num_timesteps})")

        # Start from pure noise (line 2)
        x_T = torch.randn_like(mri)

        x_t = x_T
        intermediates = []

        # Reverse diffusion (line 3)
        for step_idx, t in enumerate(timesteps):
            if verbose and step_idx % 5 == 0:
                print(f"      Step {step_idx+1}/{len(timesteps)} (t={t})")

            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict noise (line 4: F ← denoise(x_t))
            with torch.enable_grad():
                # Clone for gradient computation
                x_t_grad = x_t.clone().detach().requires_grad_(True)

                # Denoise
                noise_pred = self.model(x_t_grad, t_batch, mri, ct)

                # Predict x_0 (denoised image)
                alpha_t = self.alphas_cumprod[t]
                alpha_t = alpha_t.view(-1, 1, 1, 1, 1)

                x_0_pred = (x_t_grad - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

                # Compute ROI loss (lines 5-9)
                roi_loss = self.compute_roi_loss(
                    x_0_pred, mri, ct,
                    brain_mask, bone_mask,
                    lesion_mask, lesion_head
                )

                # Compute gradient (line 10: -η∇L_ROI)
                if roi_loss.requires_grad:
                    roi_grad = torch.autograd.grad(roi_loss, x_t_grad, retain_graph=False)[0]
                else:
                    roi_grad = torch.zeros_like(x_t_grad)

                # Uncertainty gradient (line 10: -η_u∇U)
                uncert_grad = torch.zeros_like(x_t_grad)
                if self.use_uncertainty_modulation:
                    uncertainty = self._estimate_uncertainty(x_t_grad, t_batch, mri, ct)
                    # Simple uncertainty loss: minimize variance
                    uncert_loss = uncertainty.mean()
                    if uncert_loss.requires_grad:
                        uncert_grad = torch.autograd.grad(uncert_loss, x_t_grad, retain_graph=False)[0]

            # Standard DDPM update
            beta_t = self.betas[t]
            alpha_t_val = self.alphas[t]

            # Mean prediction
            x_t_mean = (x_t - beta_t / torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t_val)

            # Add noise (except at t=0)
            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = x_t_mean + torch.sqrt(beta_t) * noise
            else:
                x_t = x_t_mean

            # Apply guided gradients (line 10) with simple normalization to
            # stabilize scale across cases
            def _norm(g):
                std = g.std()
                return g / (std + 1e-8)

            if torch.isfinite(roi_grad).all():
                roi_adj = _norm(roi_grad)
            else:
                roi_adj = torch.zeros_like(roi_grad)
            if torch.isfinite(uncert_grad).all():
                uncert_adj = _norm(uncert_grad)
            else:
                uncert_adj = torch.zeros_like(uncert_grad)

            x_t = x_t - self.eta * roi_adj - self.eta_u * uncert_adj

            if return_intermediates:
                intermediates.append(x_t.cpu())

        # REMOVED: Output clamping was killing gradients during training
        # Model will learn natural output range through training
        # x_t = torch.clamp(x_t, -3.0, 5.0)  # Commented out - was causing issues

        # Return final fused image (line 12)
        if return_intermediates:
            return x_t, intermediates
        else:
            return x_t

    def forward(self, x_0, t, mri, ct):
        """
        Training forward pass

        Args:
            x_0: Ground truth fused image (B, 1, D, H, W)
            t: Timestep (B,)
            mri: MRI conditioning (B, 1, D, H, W)
            ct: CT conditioning (B, 1, D, H, W)

        Returns:
            Predicted noise
        """
        # Sample noise
        noise = torch.randn_like(x_0)

        # Forward diffusion
        x_t = self.q_sample(x_0, t, noise)

        # Predict noise
        noise_pred = self.model(x_t, t, mri, ct)

        return noise_pred, noise
