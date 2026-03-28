"""
Conditional Denoising Diffusion Probabilistic Model (DDPM) for virtual staining.

Architecture overview
=====================
This module implements a conditional DDPM that translates brightfield (BF)
microscopy images into immunofluorescence (IF) intensity maps (e.g. DAPI +
Phalloidin channels).  Unlike regression models that predict IF directly, the
diffusion model learns the *score function* of the conditional distribution
p(IF | BF) and generates IF images through iterative denoising.

Diffusion process
-----------------
**Forward (q):**  Given a ground-truth IF image x_0, we progressively add
Gaussian noise over T timesteps according to a variance schedule
{beta_1, ..., beta_T}:

    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

where alpha_bar_t = prod_{s=1}^{t} (1 - beta_s).

**Reverse (p):**  A neural network epsilon_theta learns to predict the noise
component added at each step, conditioned on the BF image c:

    epsilon_theta(x_t, t, c)  -->  estimate of epsilon ~ N(0, I)

**Training loss:**  Simplified MSE between true and predicted noise:

    L = E_{t, x_0, epsilon} [ || epsilon - epsilon_theta(x_t, t, c) ||^2 ]

**Sampling:**  Starting from x_T ~ N(0, I), we iteratively denoise:

    x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t))
              * epsilon_theta(x_t, t, c))  +  sigma_t * z

DDIM deterministic sampling is also provided for faster inference with fewer
steps while maintaining quality.

Key components
--------------
- ``ConditionEncoder``: Multi-scale CNN that extracts BF feature maps at each
  resolution level of the UNet, enabling the denoiser to attend to the BF
  structure at every spatial scale.

- ``UNetDenoiser``: A UNet with residual blocks, sinusoidal time embeddings,
  and self-attention at lower resolutions.  The BF condition features are
  concatenated with the noisy IF features at each encoder/decoder level.

- ``ConditionalDDPM``: Top-level module orchestrating the forward diffusion,
  noise prediction, training loss, and both DDPM and DDIM reverse sampling.

Input : 3-channel brightfield image   (B, 3, H, W)
Output: 2-channel IF intensity map    (B, 2, H, W)  -- DAPI + Phalloidin
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal positional embeddings for diffusion timesteps.

    Args:
        timesteps: (B,) integer timestep indices.
        dim: Embedding dimensionality (must be even).

    Returns:
        Embedding tensor of shape (B, dim).
    """
    assert dim % 2 == 0, "Embedding dim must be even."
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class GroupNorm32(nn.GroupNorm):
    """GroupNorm that casts to float32 for numerical stability then back."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


def norm_layer(channels: int, num_groups: int = 32) -> GroupNorm32:
    return GroupNorm32(min(num_groups, channels), channels)


class ResBlock(nn.Module):
    """Residual block with time-embedding injection.

    Two conv-norm-act layers with a learnable time-embedding projection added
    after the first normalisation.  A skip projection is used when the channel
    count changes.
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = norm_layer(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = norm_layer(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # Add time embedding (broadcast over spatial dims)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention over spatial positions (QKV attention)."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = norm_layer(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)               # (B, C, N)
        qkv = self.qkv(h).view(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]        # each (B, heads, d, N)
        q = q.permute(0, 1, 3, 2)                          # (B, heads, N, d)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        scale = (C // self.num_heads) ** -0.5
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)                        # (B, heads, N, d)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H * W)
        return x + self.proj_out(out).view(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Condition encoder
# ---------------------------------------------------------------------------

class ConditionEncoder(nn.Module):
    """Multi-scale CNN encoder for the brightfield condition image.

    Produces feature maps at each resolution level that will be concatenated
    with the corresponding level of the UNet denoiser, allowing the denoiser
    to leverage BF structural information at every spatial scale.
    """

    def __init__(self, in_channels: int, base_channels: int, channel_mults: Tuple[int, ...]):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
        )

        # One feature map per UNet encoder level (same resolution & channel
        # count as the UNet level) so they can be directly concatenated.
        self.level_blocks = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        ch = base_channels
        # Store channels for each level so the UNet knows the concat width
        self.feature_channels: List[int] = []
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.level_blocks.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_ch, 3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.SiLU(),
                )
            )
            self.feature_channels.append(out_ch)
            self.down_convs.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            ch = out_ch

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return one feature map per UNet level, from level-0 to level-(L-1).

        The i-th feature map has spatial resolution H/2^i x W/2^i and channel
        count ``feature_channels[i]``.
        """
        features: List[torch.Tensor] = []
        h = self.input_conv(x)
        for block, down in zip(self.level_blocks, self.down_convs):
            h = block(h)
            features.append(h)
            h = down(h)
        return features


# ---------------------------------------------------------------------------
# UNet denoiser
# ---------------------------------------------------------------------------

class UNetDenoiser(nn.Module):
    """UNet noise predictor conditioned on timestep and BF features.

    The architecture mirrors a standard diffusion UNet:
      - Encoder path with residual blocks and downsampling
      - Bottleneck with residual blocks and self-attention
      - Decoder path with residual blocks, skip connections, and upsampling
      - Self-attention at the two lowest resolution levels

    Condition features from the ``ConditionEncoder`` are concatenated channel-
    wise at every encoder and decoder level, giving the model direct access to
    BF information at matching spatial scales.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        channel_mults: Tuple[int, ...],
        cond_channels: List[int],
        num_res_blocks: int,
        time_dim: int,
        dropout: float = 0.0,
        attn_resolutions: Tuple[int, ...] = (2, 3),
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.time_dim = time_dim
        self.num_levels = len(channel_mults)

        # ---- Input projection ----
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ---- Encoder ----
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = base_channels
        encoder_channels: List[int] = [ch]

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            cc = cond_channels[level]
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch + cc, out_ch, time_dim, dropout))
                ch = out_ch
                if level in attn_resolutions:
                    attns.append(SelfAttention(ch))
                else:
                    attns.append(nn.Identity())
                encoder_channels.append(ch)
            self.encoder_blocks.append(blocks)
            self.encoder_attns.append(attns)
            if level < len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
                encoder_channels.append(ch)
            else:
                self.downsamples.append(nn.Identity())

        # ---- Bottleneck ----
        bot_ch = base_channels * channel_mults[-1]
        self.bot1 = ResBlock(ch, bot_ch, time_dim, dropout)
        self.bot_attn = SelfAttention(bot_ch)
        self.bot2 = ResBlock(bot_ch, bot_ch, time_dim, dropout)
        ch = bot_ch

        # ---- Decoder ----
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level in reversed(range(len(channel_mults))):
            out_ch = base_channels * channel_mults[level]
            cc = cond_channels[level]
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                skip_ch = encoder_channels.pop()
                blocks.append(ResBlock(ch + skip_ch + cc, out_ch, time_dim, dropout))
                ch = out_ch
                if level in attn_resolutions:
                    attns.append(SelfAttention(ch))
                else:
                    attns.append(nn.Identity())
            self.decoder_blocks.append(blocks)
            self.decoder_attns.append(attns)
            if level > 0:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(nn.Identity())

        # ---- Output ----
        self.out_norm = norm_layer(ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """Predict noise given noisy image, timestep, and condition features.

        Args:
            x: Noisy IF image (B, out_channels, H, W).
            t: Integer timestep (B,).
            cond_features: List of BF condition features (one per UNet level,
                finest to coarsest), produced by ``ConditionEncoder``.

        Returns:
            Predicted noise (B, out_channels, H, W).
        """
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        h = self.input_conv(x)

        # ---- Encoder ----
        skips: List[torch.Tensor] = [h]
        for level, (blocks, attns) in enumerate(
            zip(self.encoder_blocks, self.encoder_attns)
        ):
            cond_feat = cond_features[level]
            for blk, att in zip(blocks, attns):
                cf = F.interpolate(cond_feat, size=h.shape[2:], mode="bilinear", align_corners=False)
                h = torch.cat([h, cf], dim=1)
                h = blk(h, t_emb)
                h = att(h)
                skips.append(h)
            ds = self.downsamples[level]
            if not isinstance(ds, nn.Identity):
                h = ds(h)
                skips.append(h)

        # ---- Bottleneck ----
        h = self.bot1(h, t_emb)
        h = self.bot_attn(h)
        h = self.bot2(h, t_emb)

        # ---- Decoder ----
        for level_idx, (blocks, attns) in enumerate(
            zip(self.decoder_blocks, self.decoder_attns)
        ):
            # Decoder levels go from coarsest to finest
            level = self.num_levels - 1 - level_idx
            cond_feat = cond_features[level]
            for blk, att in zip(blocks, attns):
                skip = skips.pop()
                if h.shape[2:] != skip.shape[2:]:
                    h = F.interpolate(h, size=skip.shape[2:], mode="nearest")
                cf = F.interpolate(cond_feat, size=h.shape[2:], mode="bilinear", align_corners=False)
                h = torch.cat([h, skip, cf], dim=1)
                h = blk(h, t_emb)
                h = att(h)
            us = self.upsamples[level_idx]
            if not isinstance(us, nn.Identity):
                h = us(h)

        h = self.out_conv(F.silu(self.out_norm(h)))
        return h


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------

def _linear_beta_schedule(num_timesteps: int) -> torch.Tensor:
    return torch.linspace(1e-4, 0.02, num_timesteps, dtype=torch.float64)


def _cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(num_timesteps + 1, dtype=torch.float64) / num_timesteps
    alpha_bar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(max=0.999)


# ---------------------------------------------------------------------------
# ConditionalDDPM
# ---------------------------------------------------------------------------

class ConditionalDDPM(nn.Module):
    """Conditional DDPM for virtual staining: BF -> IF.

    This module wraps the full diffusion pipeline:
      - Condition encoder (BF)
      - UNet noise predictor
      - Forward diffusion (q_sample)
      - Training loss computation
      - DDPM reverse sampling (p_sample_loop)
      - DDIM deterministic sampling (ddim_sample)

    Args:
        config: Nested dict with keys under ``model``:
            - in_channels (int): BF input channels (default 3).
            - out_channels (int): IF output channels (default 2).
            - base_channels (int): Base feature width (default 64).
            - channel_mults (list[int]): Channel multipliers per level.
            - num_res_blocks (int): Residual blocks per level.
            - time_embed_dim (int): Timestep embedding dimension.
            - num_timesteps (int): Number of diffusion steps T.
            - beta_schedule (str): ``'linear'`` or ``'cosine'``.
    """

    def __init__(self, config: Dict):
        super().__init__()
        mcfg = config["model"]
        self.in_channels = mcfg.get("in_channels", 3)
        self.out_channels = mcfg.get("out_channels", 2)
        base_channels = mcfg.get("base_channels", 64)
        channel_mults = tuple(mcfg.get("channel_mults", (1, 2, 4, 8)))
        num_res_blocks = mcfg.get("num_res_blocks", 2)
        time_dim = mcfg.get("time_embed_dim", 256)
        self.num_timesteps = mcfg.get("num_timesteps", 1000)
        beta_schedule = mcfg.get("beta_schedule", "linear")

        # --- Noise schedule (registered as buffers, not parameters) ---
        if beta_schedule == "cosine":
            betas = _cosine_beta_schedule(self.num_timesteps)
        else:
            betas = _linear_beta_schedule(self.num_timesteps)

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

        # Posterior variance  q(x_{t-1} | x_t, x_0)
        posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alpha_bar", alpha_bar.float())
        self.register_buffer("alpha_bar_prev", alpha_bar_prev.float())
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar).float())
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar).float())
        self.register_buffer("sqrt_recip_alpha", torch.rsqrt(alphas).float())
        self.register_buffer("posterior_var", posterior_var.float())
        self.register_buffer(
            "posterior_log_var_clipped",
            torch.log(posterior_var.clamp(min=1e-20)).float(),
        )
        # Coefficient for mean prediction:  beta_t / sqrt(1 - alpha_bar_t)
        self.register_buffer(
            "noise_coeff",
            (betas / torch.sqrt(1.0 - alpha_bar)).float(),
        )

        # --- Networks ---
        self.condition_encoder = ConditionEncoder(
            in_channels=self.in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
        )

        # Attention at the two lowest resolution levels
        attn_resolutions = tuple(range(len(channel_mults) - 2, len(channel_mults)))

        self.denoiser = UNetDenoiser(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            time_dim=time_dim,
            attn_resolutions=attn_resolutions,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract(self, schedule: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """Gather schedule values for batch of timesteps and reshape for broadcasting."""
        out = schedule.gather(0, t.long())
        return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))

    # ------------------------------------------------------------------
    # Forward diffusion
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from q(x_t | x_0): add noise to clean IF image.

        Args:
            x_0: Clean IF image (B, C_out, H, W).
            t: Timestep indices (B,).
            noise: Optional pre-sampled noise.

        Returns:
            (x_t, noise) tuple.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self._extract(self.sqrt_alpha_bar, t, x_0.shape)
        sqrt_omab = self._extract(self.sqrt_one_minus_alpha_bar, t, x_0.shape)
        x_t = sqrt_ab * x_0 + sqrt_omab * noise
        return x_t, noise

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(
        self,
        bf_images: torch.Tensor,
        if_images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass: compute simplified noise-prediction MSE loss.

        Args:
            bf_images: Brightfield condition (B, C_in, H, W).
            if_images: Ground-truth IF target (B, C_out, H, W).

        Returns:
            Dict with ``'loss'`` and ``'pred_noise'`` keys.
        """
        B = bf_images.shape[0]
        device = bf_images.device

        # Random timesteps
        t = torch.randint(0, self.num_timesteps, (B,), device=device)

        # Forward diffusion
        noise = torch.randn_like(if_images)
        x_t, _ = self.q_sample(if_images, t, noise=noise)

        # Condition features
        cond_features = self.condition_encoder(bf_images)

        # Predict noise
        pred_noise = self.denoiser(x_t, t, cond_features)

        # Simplified loss
        loss = F.mse_loss(pred_noise, noise)

        return {"loss": loss, "pred_noise": pred_noise}

    # ------------------------------------------------------------------
    # DDPM reverse sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """Single DDPM reverse step: sample x_{t-1} from p(x_{t-1} | x_t, c).

        Args:
            x_t: Current noisy image (B, C_out, H, W).
            t: Current timestep (B,), all entries identical.
            cond_features: BF condition features.

        Returns:
            Denoised image x_{t-1}.
        """
        pred_noise = self.denoiser(x_t, t, cond_features)

        coeff = self._extract(self.noise_coeff, t, x_t.shape)
        recip = self._extract(self.sqrt_recip_alpha, t, x_t.shape)
        mean = recip * (x_t - coeff * pred_noise)

        if t[0].item() > 0:
            var = self._extract(self.posterior_var, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(var) * noise
        else:
            return mean

    @torch.no_grad()
    def p_sample_loop(
        self,
        bf_images: torch.Tensor,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Full DDPM reverse process: x_T -> x_0 conditioned on BF.

        Args:
            bf_images: Brightfield condition (B, C_in, H, W).
            shape: Optional output shape; defaults to
                (B, out_channels, H, W).

        Returns:
            Generated IF image (B, C_out, H, W).
        """
        device = bf_images.device
        B = bf_images.shape[0]
        if shape is None:
            shape = (B, self.out_channels, bf_images.shape[2], bf_images.shape[3])

        cond_features = self.condition_encoder(bf_images)
        x = torch.randn(shape, device=device)

        for i in reversed(range(self.num_timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, cond_features)

        return x

    # ------------------------------------------------------------------
    # DDIM sampling (faster, deterministic)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        bf_images: torch.Tensor,
        num_steps: int = 50,
        eta: float = 0.0,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """DDIM sampling for faster inference.

        Uses a sub-sequence of the full diffusion schedule to generate images
        in fewer steps.  When ``eta=0`` the process is fully deterministic;
        ``eta=1`` recovers the stochastic DDPM sampler.

        Args:
            bf_images: Brightfield condition (B, C_in, H, W).
            num_steps: Number of DDIM steps (<<T for speed).
            eta: Stochasticity parameter (0 = deterministic).
            shape: Optional output shape.

        Returns:
            Generated IF image (B, C_out, H, W).
        """
        device = bf_images.device
        B = bf_images.shape[0]
        if shape is None:
            shape = (B, self.out_channels, bf_images.shape[2], bf_images.shape[3])

        cond_features = self.condition_encoder(bf_images)

        # Sub-sequence of timesteps (evenly spaced)
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        x = torch.randn(shape, device=device)

        for i, t_cur in enumerate(timesteps):
            t = torch.full((B,), t_cur, device=device, dtype=torch.long)
            pred_noise = self.denoiser(x, t, cond_features)

            # Predicted x_0
            sqrt_ab = self._extract(self.sqrt_alpha_bar, t, x.shape)
            sqrt_omab = self._extract(self.sqrt_one_minus_alpha_bar, t, x.shape)
            pred_x0 = (x - sqrt_omab * pred_noise) / sqrt_ab
            pred_x0 = pred_x0.clamp(-1, 1)

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                t_prev_tensor = torch.full((B,), t_prev, device=device, dtype=torch.long)

                ab_cur = self._extract(self.alpha_bar, t, x.shape)
                ab_prev = self._extract(self.alpha_bar, t_prev_tensor, x.shape)

                # DDIM variance
                sigma = eta * torch.sqrt(
                    (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
                )

                # Direction pointing to x_t
                dir_xt = torch.sqrt(1 - ab_prev - sigma ** 2) * pred_noise

                noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
                x = torch.sqrt(ab_prev) * pred_x0 + dir_xt + sigma * noise
            else:
                # Last step: just return predicted x_0
                x = pred_x0

        return x

    # ------------------------------------------------------------------
    # High-level sampling API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        bf_images: torch.Tensor,
        method: str = "ddim",
        num_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Generate IF images from BF input (main inference entry point).

        Args:
            bf_images: Brightfield input (B, C_in, H, W).
            method: ``'ddpm'`` for full reverse process or ``'ddim'`` for
                accelerated deterministic sampling.
            num_steps: Number of DDIM steps (ignored for DDPM).
            eta: DDIM stochasticity (0 = deterministic).

        Returns:
            Predicted IF intensity map (B, C_out, H, W).
        """
        if method == "ddpm":
            return self.p_sample_loop(bf_images)
        elif method == "ddim":
            return self.ddim_sample(bf_images, num_steps=num_steps, eta=eta)
        else:
            raise ValueError(f"Unknown sampling method '{method}'. Use 'ddpm' or 'ddim'.")
