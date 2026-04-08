"""Conditional Flow Matching for virtual staining.

Uses a pretrained VelocityUNet (trained on 133M params, 448px human embryo
timelapse, unconditional flow matching) as the backbone for conditional
BF -> IF generation.

The flow matching model learned the embryo image manifold. We adapt it to
conditional generation by:
1. Expanding the input conv from 1 channel to (1 BF + out_channels) channels
2. Expanding the output conv from 1 channel to out_channels (DAPI + Phalloidin)
3. Loading all pretrained weights for the middle of the UNet
4. Training the flow to predict velocity from noise to IF, conditioned on BF

At inference, we start from Gaussian noise in the IF space, run the ODE
integration conditioned on BF, and get a predicted IF image.

Key advantages over regression:
- Generative: no washed-out means, sharp predictions
- Pretrained on embryo distribution: needs less paired data
- Trained on human timelapse: may generalize across species/imaging conditions
- Flow matching: ~20-50 inference steps (vs DDPM's 1000)

Normalization: all inputs/outputs in [-1, 1] during the flow matching pass,
converted to [0, 1] at the boundary for compatibility with the training
framework.
"""

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Velocity UNet (copied from the user's pretrained model for compatibility)
# ---------------------------------------------------------------------------

class SinusoidalPositionEmbeddings(nn.Module):
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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, t):
        h = self.activation(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(t)[:, :, None, None]
        h = self.dropout(self.activation(self.norm2(self.conv2(h))))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3).contiguous()
        k = k.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3).contiguous()
        v = v.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3).contiguous()
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        return self.proj(out) + x


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x, target_size=None):
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='nearest')
        else:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class VelocityUNet(nn.Module):
    """UNet that predicts velocity field v(x_t, t) for flow matching.

    Architecture identical to the pretrained model for weight loading
    compatibility. Image channel count is configurable.
    """

    def __init__(self, image_channels=3, base_channels=96,
                 channel_multipliers=(1, 2, 4, 8), num_res_blocks=2,
                 time_emb_dim=512, num_heads=12, dropout=0.1,
                 multi_scale_attention=True, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = image_channels
        self.in_channels = image_channels
        self.out_channels = out_channels

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.conv_in = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        self.downs = nn.ModuleList()
        self.down_attn = nn.ModuleDict()
        self.ups = nn.ModuleList()
        self.up_attn = nn.ModuleDict()
        self.num_resolutions = len(channel_multipliers)

        attn_levels = 2 if multi_scale_attention else 1
        attn_min_level = len(channel_multipliers) - attn_levels

        channels = [base_channels]
        now_channels = base_channels
        down_idx = 0
        for level, mult in enumerate(channel_multipliers):
            out_c = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(now_channels, out_c, time_emb_dim, dropout))
                now_channels = out_c
                channels.append(out_c)
                down_idx += 1
            if level >= attn_min_level:
                self.down_attn[str(down_idx - 1)] = AttentionBlock(now_channels, num_heads)
            if level != self.num_resolutions - 1:
                self.downs.append(nn.Conv2d(now_channels, now_channels, 3, stride=2, padding=1))
                channels.append(now_channels)
                down_idx += 1

        self.mid = nn.ModuleList([
            ResidualBlock(now_channels, now_channels, time_emb_dim, dropout),
            AttentionBlock(now_channels, num_heads),
            ResidualBlock(now_channels, now_channels, time_emb_dim, dropout)
        ])

        up_idx = 0
        for level, mult in list(enumerate(channel_multipliers))[::-1]:
            out_c = base_channels * mult
            for _ in range(num_res_blocks + 1):
                skip_channels = channels.pop()
                self.ups.append(
                    ResidualBlock(now_channels + skip_channels, out_c, time_emb_dim, dropout)
                )
                now_channels = out_c
                up_idx += 1
            if level >= attn_min_level:
                self.up_attn[str(up_idx - 1)] = AttentionBlock(now_channels, num_heads)
            if level != 0:
                self.ups.append(Upsample(now_channels))
                up_idx += 1

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, now_channels),
            nn.SiLU(),
            nn.Conv2d(now_channels, out_channels, 3, padding=1)
        )

    def forward(self, x, t):
        temb = self.time_mlp(t)
        h = self.conv_in(x)
        hs = [h]

        idx = 0
        for module in self.downs:
            if isinstance(module, ResidualBlock):
                h = module(h, temb)
                if str(idx) in self.down_attn:
                    h = self.down_attn[str(idx)](h)
            else:
                h = module(h)
            hs.append(h)
            idx += 1

        for module in self.mid:
            if isinstance(module, ResidualBlock):
                h = module(h, temb)
            else:
                h = module(h)

        idx = 0
        for module in self.ups:
            if isinstance(module, ResidualBlock):
                skip = hs.pop()
                if h.shape[2:] != skip.shape[2:]:
                    h = F.interpolate(h, size=skip.shape[2:], mode='nearest')
                h = torch.cat([h, skip], dim=1)
                h = module(h, temb)
                if str(idx) in self.up_attn:
                    h = self.up_attn[str(idx)](h)
            elif isinstance(module, Upsample):
                target = hs[-1].shape[2:] if hs else None
                h = module(h, target_size=target)
            else:
                h = module(h)
            idx += 1

        return self.conv_out(h)


# ---------------------------------------------------------------------------
# Conditional Flow Matching model (wraps UNet with conditioning)
# ---------------------------------------------------------------------------

class ConditionalFlowMatching(nn.Module):
    """Conditional Flow Matching for virtual staining.

    Configuration (config['model']):
        in_channels:         int, brightfield channels (default 3, we use first channel)
        out_channels:        int, IF channels to predict (default 2, DAPI + Phalloidin)
        base_channels:       int, UNet base width (default 96 — matches pretrained)
        channel_multipliers: list, width per stage (default [1,2,4,8])
        num_res_blocks:      int, blocks per stage (default 2)
        time_emb_dim:        int, time embedding dim (default 512)
        num_heads:           int, attention heads (default 12)
        dropout:             float, dropout rate (default 0.1)
        multi_scale_attention: bool (default True)
        pretrained_checkpoint: str, path to flow matching checkpoint (or None)
        sigma_min:           float, noise floor (default 1e-4)
    """

    def __init__(self, config: Dict):
        super().__init__()
        mcfg = config.get("model", {})

        self.in_channels = mcfg.get("in_channels", 3)
        self.out_channels = mcfg.get("out_channels", 2)
        # UNet input = 1 (BF, grayscale) + out_channels (noisy IF)
        self.bf_condition_channels = 1
        self.unet_in_channels = self.bf_condition_channels + self.out_channels

        self.sigma_min = mcfg.get("sigma_min", 1e-4)

        self.unet = VelocityUNet(
            image_channels=self.unet_in_channels,
            base_channels=mcfg.get("base_channels", 96),
            channel_multipliers=tuple(mcfg.get("channel_multipliers", [1, 2, 4, 8])),
            num_res_blocks=mcfg.get("num_res_blocks", 2),
            time_emb_dim=mcfg.get("time_emb_dim", 512),
            num_heads=mcfg.get("num_heads", 12),
            dropout=mcfg.get("dropout", 0.1),
            multi_scale_attention=mcfg.get("multi_scale_attention", True),
            out_channels=self.out_channels,
        )

        pretrained_path = mcfg.get("pretrained_checkpoint", None)
        if pretrained_path:
            self._load_pretrained(pretrained_path)

        # EMA for stable sampling
        self.ema_decay = mcfg.get("ema_decay", 0.995)
        self.ema_unet = copy.deepcopy(self.unet)
        for p in self.ema_unet.parameters():
            p.requires_grad = False
        self.ema_step_count = 0

    # -----------------------------------------------------------------
    def _load_pretrained(self, checkpoint_path: str):
        """Load pretrained VelocityUNet weights, expanding conv_in/conv_out.

        The pretrained model has:
        - conv_in: Conv2d(1, 96, 3, padding=1)  — grayscale BF input
        - conv_out[-1]: Conv2d(96, 1, 3, padding=1) — grayscale velocity output

        We want:
        - conv_in: Conv2d(1 + out_channels, 96, 3, padding=1)
        - conv_out[-1]: Conv2d(96, out_channels, 3, padding=1)

        Strategy:
        - Load all middle weights directly
        - For conv_in: copy pretrained weights into the BF condition slot (channel 0),
          tile into the IF noise channels (scaled down to preserve activation magnitude)
        - For conv_out: copy pretrained weight into channel 0 (DAPI),
          initialize remaining channels with Kaiming normal
        """
        print(f"Loading flow matching checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Extract state dict — prefer EMA weights (better quality for generation)
        if isinstance(ckpt, dict):
            if "ema_state_dict" in ckpt:
                state_dict = ckpt["ema_state_dict"]
                print("  Using EMA weights")
            elif "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
                print("  Using model weights")
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        # Strip module. prefix if present
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            cleaned[k] = v

        # Separate conv_in and conv_out weights — handle them specially
        own_state = self.unet.state_dict()
        loaded_count = 0
        skipped_count = 0

        for name, param in cleaned.items():
            if name not in own_state:
                skipped_count += 1
                continue

            own_param = own_state[name]
            if own_param.shape == param.shape:
                own_state[name].copy_(param)
                loaded_count += 1
            elif name == "conv_in.weight":
                # Pretrained: (base_channels, 1, 3, 3)
                # Target:     (base_channels, 1 + out_channels, 3, 3)
                # Strategy: replicate pretrained weight across all input channels,
                # scaled by 1/n_in so the initial activation matches the pretrained magnitude
                new_in_ch = own_param.shape[1]
                expanded = param.repeat(1, new_in_ch, 1, 1) / new_in_ch
                own_state[name].copy_(expanded)
                loaded_count += 1
                print(f"  Expanded conv_in: {param.shape} -> {own_param.shape} (replicated + scaled by 1/{new_in_ch})")
            elif name == "conv_in.bias":
                own_state[name].copy_(param)
                loaded_count += 1
            elif name == "conv_out.2.weight":
                # Pretrained: (1, base_channels, 3, 3) — final conv (index 2 after GroupNorm + SiLU)
                # Target:     (out_channels, base_channels, 3, 3)
                # Strategy: copy pretrained weight to ALL output channels so each
                # target channel starts from the same pretrained prior
                new_out = own_param.shape[0]
                expanded = param.repeat(new_out, 1, 1, 1)
                own_state[name].copy_(expanded)
                loaded_count += 1
                print(f"  Expanded conv_out: {param.shape} -> {own_param.shape} (replicated to {new_out} channels)")
            elif name == "conv_out.2.bias":
                new_out = own_param.shape[0]
                own_state[name].copy_(param.repeat(new_out))
                loaded_count += 1
            else:
                print(f"  Shape mismatch: {name} {own_param.shape} vs {param.shape}, skipping")
                skipped_count += 1

        self.unet.load_state_dict(own_state)
        print(f"  Loaded {loaded_count} params, skipped {skipped_count}")

    # -----------------------------------------------------------------
    @torch.no_grad()
    def ema_update(self):
        """Update EMA shadow weights. Called after each optimizer step."""
        self.ema_step_count += 1
        warmup = 100
        if self.ema_step_count <= warmup:
            decay = min(self.ema_decay, 1.0 - 1.0 / (self.ema_step_count + 1))
        else:
            decay = self.ema_decay
        for ema_p, p in zip(self.ema_unet.parameters(), self.unet.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)

    # -----------------------------------------------------------------
    @staticmethod
    def _to_flow_range(x: torch.Tensor) -> torch.Tensor:
        """Convert [0, 1] -> [-1, 1] for flow matching internal space."""
        return x * 2.0 - 1.0

    @staticmethod
    def _from_flow_range(x: torch.Tensor) -> torch.Tensor:
        """Convert [-1, 1] -> [0, 1] for external range."""
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0)

    @staticmethod
    def _bf_to_gray(bf: torch.Tensor) -> torch.Tensor:
        """Convert BF (B, 3, H, W) to grayscale (B, 1, H, W)."""
        if bf.size(1) == 1:
            return bf
        # Average 3 channels (BF was replicated from grayscale anyway in the loader)
        return bf.mean(dim=1, keepdim=True)

    # -----------------------------------------------------------------
    def training_loss(self, if_target: torch.Tensor, bf: torch.Tensor) -> torch.Tensor:
        """Compute flow matching loss.

        Args:
            if_target: Ground truth IF image in [0, 1], shape (B, out_channels, H, W)
            bf: Brightfield input in [0, 1], shape (B, in_channels, H, W)

        Returns:
            Scalar loss (MSE between predicted and target velocity).
        """
        # Convert to flow matching range [-1, 1]
        x1 = self._to_flow_range(if_target)  # target (IF)
        bf_cond = self._to_flow_range(self._bf_to_gray(bf))  # condition (BF, 1 channel)

        batch_size = x1.shape[0]
        device = x1.device

        # Sample random time and noise
        t = torch.rand(batch_size, device=device)
        noise = torch.randn_like(x1)  # x0 ~ N(0, I) in IF space

        # Linear interpolation between noise and target
        t_expand = t[:, None, None, None]
        x_t = (1 - t_expand) * noise + t_expand * x1

        # Concat BF condition with noisy state
        unet_input = torch.cat([bf_cond, x_t], dim=1)

        # Target velocity is constant: x1 - x0
        target_velocity = x1 - noise

        # Predict velocity
        predicted_velocity = self.unet(unet_input, t)

        return F.mse_loss(predicted_velocity, target_velocity)

    # -----------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        bf: torch.Tensor,
        method: str = "midpoint",
        num_steps: int = 50,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """Generate IF images from BF input.

        Args:
            bf: Brightfield input in [0, 1], shape (B, in_channels, H, W).
            method: 'euler' or 'midpoint' integration.
            num_steps: Number of ODE integration steps.
            use_ema: Use EMA shadow weights (recommended).

        Returns:
            Predicted IF images in [0, 1], shape (B, out_channels, H, W).
        """
        bf_cond = self._to_flow_range(self._bf_to_gray(bf))
        B, _, H, W = bf_cond.shape
        device = bf_cond.device

        unet = self.ema_unet if use_ema else self.unet
        unet.eval()

        # Start from noise in IF space
        x = torch.randn(B, self.out_channels, H, W, device=device)
        dt = 1.0 / num_steps

        if method == "euler":
            for i in range(num_steps):
                t = torch.full((B,), i / num_steps, device=device)
                unet_input = torch.cat([bf_cond, x], dim=1)
                v = unet(unet_input, t)
                x = x + v * dt
        else:  # midpoint
            for i in range(num_steps):
                t = torch.full((B,), i / num_steps, device=device)
                t_mid = torch.full((B,), (i + 0.5) / num_steps, device=device)

                unet_input = torch.cat([bf_cond, x], dim=1)
                v = unet(unet_input, t)
                x_mid = x + v * (dt / 2)

                unet_input_mid = torch.cat([bf_cond, x_mid], dim=1)
                v_mid = unet(unet_input_mid, t_mid)
                x = x + v_mid * dt

        return self._from_flow_range(x)

    # -----------------------------------------------------------------
    def forward(self, bf: torch.Tensor) -> torch.Tensor:
        """Inference entry point — matches other regression models' signature.

        Used by the training loop's validation step. Generates IF from BF
        using DDIM-like sampling.
        """
        return self.sample(bf, method="midpoint", num_steps=50, use_ema=True)
