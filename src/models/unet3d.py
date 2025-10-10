"""3D U-Net for image-level diffusion denoising"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timesteps"""

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


class DoubleConv3D(nn.Module):
    """Double convolution block for U-Net"""

    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.GELU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return nn.functional.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down3D(nn.Module):
    """Downscaling block with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels, residual=True)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upscaling block with upsampling, concat, and double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2, residual=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle potential size mismatches
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2,
                                    diffZ // 2, diffZ - diffZ // 2])

        # Concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionBlock3D(nn.Module):
    """Self-attention block for U-Net"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv3d(channels, channels, 1)
        self.k = nn.Conv3d(channels, channels, 1)
        self.v = nn.Conv3d(channels, channels, 1)
        self.proj_out = nn.Conv3d(channels, channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        # Compute attention
        b, c, d, h, w = q.shape
        q = q.reshape(b, c, d * h * w).permute(0, 2, 1)  # (b, dhw, c)
        k = k.reshape(b, c, d * h * w)  # (b, c, dhw)
        v = v.reshape(b, c, d * h * w).permute(0, 2, 1)  # (b, dhw, c)

        attn = torch.bmm(q, k) * (c ** (-0.5))
        attn = torch.softmax(attn, dim=2)

        out = torch.bmm(attn, v)  # (b, dhw, c)
        out = out.permute(0, 2, 1).reshape(b, c, d, h, w)

        return x + self.proj_out(out)


class UNet3D(nn.Module):
    """
    3D U-Net for image-level diffusion denoising

    Args:
        in_channels: Number of input channels (usually 1 for fused image)
        out_channels: Number of output channels (usually 1)
        cond_channels: Number of conditioning channels (from MRI/CT encoders)
        base_channels: Base number of channels (default: 64)
        time_emb_dim: Dimension of time embedding (default: 256)
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        cond_channels=512,
        base_channels=64,
        time_emb_dim=256,
        channel_multipliers=(1, 2, 4, 8),
        attention_resolutions=(4,),
        num_res_blocks=2
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Conditioning projection (from MRI/CT encoders)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_channels, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Initial convolution
        self.inc = DoubleConv3D(in_channels, base_channels)

        # Downsampling path
        self.downs = nn.ModuleList()
        ch_in = base_channels
        for i, mult in enumerate(channel_multipliers):
            ch_out = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(Down3D(ch_in, ch_out))
                ch_in = ch_out

        # Middle block with attention
        self.mid_block1 = DoubleConv3D(ch_in, ch_in, residual=True)
        self.mid_attn = AttentionBlock3D(ch_in)
        self.mid_block2 = DoubleConv3D(ch_in, ch_in, residual=True)

        # Upsampling path
        self.ups = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_multipliers)):
            ch_out = base_channels * mult
            for _ in range(num_res_blocks):
                self.ups.append(Up3D(ch_in + ch_out, ch_out))
                ch_in = ch_out

        # Output convolution
        self.outc = nn.Sequential(
            DoubleConv3D(ch_in, base_channels),
            nn.Conv3d(base_channels, out_channels, kernel_size=1)
        )

        # Time embedding layers for each level
        self.time_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, base_channels * mult)
            )
            for mult in channel_multipliers for _ in range(num_res_blocks)
        ])

    def forward(self, x, t, cond=None):
        """
        Forward pass

        Args:
            x: Noisy image (batch_size, 1, D, H, W)
            t: Timestep (batch_size,)
            cond: Conditioning from MRI/CT encoders (batch_size, cond_channels)
                  Can be dict with 'mri' and 'ct' keys

        Returns:
            Predicted noise (batch_size, 1, D, H, W)
        """
        # Embed time
        t_emb = self.time_mlp(t)  # (B, time_emb_dim)

        # Add conditioning if provided
        if cond is not None:
            if isinstance(cond, dict):
                # Concatenate MRI and CT conditioning
                cond_emb = torch.cat([cond.get('mri', torch.zeros_like(t_emb)),
                                     cond.get('ct', torch.zeros_like(t_emb))], dim=-1)
                if cond_emb.shape[-1] != self.cond_proj[0].in_features:
                    # Handle size mismatch
                    cond_emb = cond_emb[..., :self.cond_proj[0].in_features]
            else:
                cond_emb = cond

            c_emb = self.cond_proj(cond_emb)  # (B, time_emb_dim)
            t_emb = t_emb + c_emb

        # Initial conv
        x = self.inc(x)

        # Downsampling
        skip_connections = []
        for i, down in enumerate(self.downs):
            x = down(x)

            # Add time embedding
            if i < len(self.time_emb_layers):
                t_proj = self.time_emb_layers[i](t_emb)[:, :, None, None, None]
                x = x + t_proj

            skip_connections.append(x)

        # Middle
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        # Upsampling
        skip_connections = skip_connections[::-1]
        for i, (up, skip) in enumerate(zip(self.ups, skip_connections)):
            x = up(x, skip)

        # Output
        return self.outc(x)


class ConditionalEncoder(nn.Module):
    """Lightweight encoder for MRI/CT conditioning"""

    def __init__(self, in_channels=1, out_channels=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),

            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),

            nn.AdaptiveAvgPool3d(1)
        )

        self.proj = nn.Linear(128, out_channels)

    def forward(self, x):
        """
        Args:
            x: Input image (B, 1, D, H, W)
        Returns:
            Encoding (B, out_channels)
        """
        feat = self.encoder(x)
        feat = feat.flatten(1)
        return self.proj(feat)


class ImageFusionDiffusion(nn.Module):
    """
    Complete image fusion diffusion model

    Input: MRI image, CT image
    Output: Fused image
    """

    def __init__(
        self,
        image_channels=1,
        cond_dim=256,
        unet_base_channels=64,
        time_emb_dim=256
    ):
        super().__init__()

        # Conditioning encoders
        self.mri_encoder = ConditionalEncoder(image_channels, cond_dim)
        self.ct_encoder = ConditionalEncoder(image_channels, cond_dim)

        # U-Net denoiser
        self.unet = UNet3D(
            in_channels=image_channels,
            out_channels=image_channels,
            cond_channels=cond_dim * 2,  # MRI + CT
            base_channels=unet_base_channels,
            time_emb_dim=time_emb_dim
        )

    def encode_condition(self, mri=None, ct=None):
        """Encode MRI and CT for conditioning"""
        cond = {}

        if mri is not None:
            cond['mri'] = self.mri_encoder(mri)
        if ct is not None:
            cond['ct'] = self.ct_encoder(ct)

        # Concatenate
        if 'mri' in cond and 'ct' in cond:
            return torch.cat([cond['mri'], cond['ct']], dim=-1)
        elif 'mri' in cond:
            return torch.cat([cond['mri'], torch.zeros_like(cond['mri'])], dim=-1)
        elif 'ct' in cond:
            return torch.cat([torch.zeros_like(cond['ct']), cond['ct']], dim=-1)
        else:
            return None

    def forward(self, x_t, t, mri=None, ct=None):
        """
        Denoise step

        Args:
            x_t: Noisy fused image (B, 1, D, H, W)
            t: Timestep (B,)
            mri: MRI image for conditioning (B, 1, D, H, W)
            ct: CT image for conditioning (B, 1, D, H, W)

        Returns:
            Predicted noise (B, 1, D, H, W)
        """
        # Encode conditioning
        cond = self.encode_condition(mri, ct)

        # Denoise
        return self.unet(x_t, t, cond)
