"""Diffusion model for multimodal fusion"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional embeddings for timesteps"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffusionTransformer(nn.Module):
    """Transformer-based denoising network for diffusion"""

    def __init__(self, input_dim, hidden_dim=512, num_layers=6, num_heads=8, dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t, condition=None):
        """
        Args:
            x: (batch_size, input_dim) - noisy input
            t: (batch_size,) - timestep
            condition: (batch_size, cond_dim) - optional conditioning
        Returns:
            noise_pred: (batch_size, input_dim)
        """
        # Project input
        h = self.input_proj(x)  # (B, hidden_dim)

        # Add time embedding
        t_emb = self.time_embed(t)  # (B, hidden_dim)
        h = h + t_emb

        # Add conditioning if provided
        if condition is not None:
            h = h + condition

        # Add sequence dimension for transformer (B, 1, hidden_dim)
        h = h.unsqueeze(1)

        # Transformer
        h = self.transformer(h)  # (B, 1, hidden_dim)

        # Remove sequence dimension
        h = h.squeeze(1)  # (B, hidden_dim)

        # Project to output
        return self.output_proj(h)


class GaussianDiffusion(nn.Module):
    """Gaussian diffusion process for multimodal fusion"""

    def __init__(
        self,
        input_dim,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        num_timesteps=1000,
        beta_schedule='linear'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_timesteps = num_timesteps

        # Denoising network
        self.denoiser = DiffusionTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        # Noise schedule
        self.register_buffer('betas', self._get_betas(beta_schedule, num_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev',
                            torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]]))

        # Derived quantities for sampling
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                            torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / self.alphas))

    def _get_betas(self, schedule, timesteps):
        """Get beta schedule"""
        if schedule == 'linear':
            return torch.linspace(1e-4, 0.02, timesteps)
        elif schedule == 'cosine':
            # Cosine schedule from improved DDPM paper
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion process: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_0, t, condition=None, noise=None):
        """Compute denoising loss"""
        if noise is None:
            noise = torch.randn_like(x_0)

        # Forward diffusion
        x_t = self.q_sample(x_0, t, noise)

        # Predict noise
        noise_pred = self.denoiser(x_t, t, condition)

        # MSE loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        return loss

    @torch.no_grad()
    def p_sample(self, x_t, t, condition=None):
        """Single reverse diffusion step: p(x_{t-1} | x_t)"""
        # Predict noise
        noise_pred = self.denoiser(x_t, t, condition)

        # Compute x_{t-1}
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].reshape(-1, 1)
        betas_t = self.betas[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)

        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
        )

        if t[0] == 0:
            return model_mean
        else:
            noise = torch.randn_like(x_t)
            sqrt_betas_t = torch.sqrt(betas_t)
            return model_mean + sqrt_betas_t * noise

    @torch.no_grad()
    def sample(self, shape, condition=None, device='cuda'):
        """Generate sample through reverse diffusion"""
        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Reverse diffusion
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, condition)

        return x

    def forward(self, x_0, condition=None):
        """
        Training forward pass
        Args:
            x_0: (batch_size, input_dim) - clean input
            condition: optional conditioning information
        Returns:
            loss: diffusion loss
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)

        # Compute loss
        loss = self.p_losses(x_0, t, condition)

        return loss
