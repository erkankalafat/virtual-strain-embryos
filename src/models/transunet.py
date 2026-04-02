"""TransUNet regression model for virtual staining.

Architecture overview
=====================
TransUNet combines a CNN encoder with a Vision Transformer (ViT) to capture
both local texture and global context, then fuses them through a U-Net-style
decoder with skip connections.

Forward path
------------
1. **CNN encoder** – Four ResNet-style stages progressively downsample the
   input brightfield image (H×W) while extracting multi-scale features:
     Stage 1: H/2  × W/2,   64 channels
     Stage 2: H/4  × W/4,  128 channels
     Stage 3: H/8  × W/8,  256 channels
     Stage 4: H/16 × W/16, 512 channels

2. **Patch embedding** – The Stage-4 feature map is projected to
   ``embed_dim``-dimensional tokens (one per spatial patch).

3. **Transformer encoder** – A stack of standard ViT blocks (multi-head
   self-attention + MLP) models long-range dependencies among tokens.

4. **Reshape** – Transformer output tokens are reshaped back to a 2-D
   feature map at H/16 × W/16.

5. **CNN decoder** – Four upsampling stages mirror the encoder.  Each stage
   concatenates the upsampled features with the corresponding encoder skip
   connection, then refines through two conv-BN-ReLU blocks:
     Up 1: H/8  × W/8   (skip from Stage 3)
     Up 2: H/4  × W/4   (skip from Stage 2)
     Up 3: H/2  × W/2   (skip from Stage 1)
     Up 4: H    × W     (no skip, final refinement)

6. **Head** – A 1×1 convolution followed by Sigmoid produces the output
   fluorescence-intensity map in [0, 1].

This is a **regression** model (not segmentation).  The Sigmoid final
activation constrains predicted intensities to [0, 1], matching normalised
IF target images.

Config keys (all nested under ``model.``):
    in_channels  – input channels (default 3, brightfield RGB)
    out_channels – output channels (default 2, configurable IF channels)
    img_size     – spatial resolution, must be divisible by 16 (default 512)
    patch_size   – patch side for ViT tokenisation (default 16)
    embed_dim    – transformer hidden dimension (default 768)
    depth        – number of transformer blocks (default 12)
    num_heads    – attention heads per block (default 12)
    mlp_ratio    – MLP expansion ratio in transformer (default 4.0)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _ConvBNReLU(nn.Sequential):
    """Conv 3×3 -> BatchNorm -> ReLU helper."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ResBlock(nn.Module):
    """Two-conv residual block with optional channel projection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut: nn.Module
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + self.shortcut(x))
        return out


# ---------------------------------------------------------------------------
# CNN encoder
# ---------------------------------------------------------------------------


class _CNNEncoder(nn.Module):
    """Four-stage ResNet-style encoder producing multi-scale feature maps."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # Stage 1: -> H/2, 64ch
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            _ResBlock(64, 64),
        )
        # Stage 2: -> H/4, 128ch
        self.stage2 = nn.Sequential(
            _ResBlock(64, 128, stride=2),
            _ResBlock(128, 128),
        )
        # Stage 3: -> H/8, 256ch
        self.stage3 = nn.Sequential(
            _ResBlock(128, 256, stride=2),
            _ResBlock(256, 256),
        )
        # Stage 4: -> H/16, 512ch
        self.stage4 = nn.Sequential(
            _ResBlock(256, 512, stride=2),
            _ResBlock(512, 512),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        s1 = self.stage1(x)   # H/2
        s2 = self.stage2(s1)  # H/4
        s3 = self.stage3(s2)  # H/8
        s4 = self.stage4(s3)  # H/16
        return [s1, s2, s3, s4]


# ---------------------------------------------------------------------------
# Vision Transformer components
# ---------------------------------------------------------------------------


class _PatchEmbedding(nn.Module):
    """Project CNN feature map to transformer token sequence."""

    def __init__(self, in_channels: int, embed_dim: int, grid_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, grid_size * grid_size, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H', W') where H'=W' may differ from training grid_size
        x = self.proj(x)  # (B, embed_dim, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)

        # Interpolate positional embedding if spatial size changed (e.g. 512->1024)
        N = H * W
        if N != self.pos_embed.shape[1]:
            pos = self.pos_embed  # (1, grid^2, embed_dim)
            grid_old = int(pos.shape[1] ** 0.5)
            pos = pos.reshape(1, grid_old, grid_old, -1).permute(0, 3, 1, 2)
            pos = nn.functional.interpolate(pos, size=(H, W), mode="bilinear", align_corners=False)
            pos = pos.permute(0, 2, 3, 1).reshape(1, N, -1)
            x = x + pos
        else:
            x = x + self.pos_embed[:, :N, :]
        return x


class _MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _TransformerEncoder(nn.Module):
    def __init__(
        self, depth: int, embed_dim: int, num_heads: int, mlp_ratio: float
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# CNN decoder (U-Net style)
# ---------------------------------------------------------------------------


class _DecoderBlock(nn.Module):
    """Upsample + concat skip + two convs."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            _ConvBNReLU(in_ch + skip_ch, out_ch),
            _ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            # Handle potential size mismatch from non-power-of-2 inputs
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# TransUNet regressor
# ---------------------------------------------------------------------------


class TransUNetRegressor(nn.Module):
    """TransUNet for virtual-staining intensity regression.

    Parameters
    ----------
    config : dict
        Nested config with keys under ``model.*``.  See module docstring for
        the full list.  Example::

            config = {
                "model": {
                    "name": "transunet",
                    "in_channels": 3,
                    "out_channels": 2,
                    "img_size": 512,
                }
            }
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        mcfg = config.get("model", {})
        in_channels: int = mcfg.get("in_channels", 3)
        out_channels: int = mcfg.get("out_channels", 2)
        img_size: int = mcfg.get("img_size", 512)
        patch_size: int = mcfg.get("patch_size", 16)
        embed_dim: int = mcfg.get("embed_dim", 768)
        depth: int = mcfg.get("depth", 12)
        num_heads: int = mcfg.get("num_heads", 12)
        mlp_ratio: float = mcfg.get("mlp_ratio", 4.0)

        assert img_size % patch_size == 0, (
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
        )

        self.grid_size = img_size // patch_size  # e.g. 32 for 512/16

        # --- CNN encoder ---
        self.encoder = _CNNEncoder(in_channels)

        # --- Patch embedding (operates on Stage-4 output: 512 channels) ---
        self.patch_embed = _PatchEmbedding(512, embed_dim, self.grid_size)

        # --- Transformer encoder ---
        self.transformer = _TransformerEncoder(depth, embed_dim, num_heads, mlp_ratio)

        # --- Reshape projection (embed_dim -> 512 channels for decoder) ---
        self.decoder_proj = nn.Sequential(
            nn.Conv2d(embed_dim, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # --- CNN decoder with skip connections ---
        # Up1: 512 + 256 (skip from stage3) -> 256   (H/16 -> H/8)
        self.dec1 = _DecoderBlock(512, 256, 256)
        # Up2: 256 + 128 (skip from stage2) -> 128   (H/8  -> H/4)
        self.dec2 = _DecoderBlock(256, 128, 128)
        # Up3: 128 + 64  (skip from stage1) -> 64    (H/4  -> H/2)
        self.dec3 = _DecoderBlock(128, 64, 64)
        # Up4: 64 + 0    (no skip)          -> 32    (H/2  -> H)
        self.dec4 = _DecoderBlock(64, 0, 32)

        # --- Regression head ---
        self.head = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self._init_weights()

        # Load DINO pretrained weights into transformer blocks if specified
        dino_path = mcfg.get("dino_checkpoint", None)
        if dino_path:
            self._load_dino_transformer(dino_path)

    # -----------------------------------------------------------------
    def _load_dino_transformer(self, checkpoint_path: str) -> None:
        """Load DINO ViT-S weights into the transformer encoder blocks.

        Maps DINO's block weights (norm1, attn.qkv, attn.proj, norm2, mlp)
        to TransUNet's _TransformerBlock structure. Only loads the transformer
        blocks and positional embedding — CNN encoder/decoder are untouched.
        """
        import os
        print(f"Loading DINO transformer weights: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if isinstance(ckpt, dict):
            # Clean state dict: strip common prefixes
            state_dict = ckpt
            cleaned = {}
            for k, v in state_dict.items():
                for prefix in ["backbone.", "module.", "encoder.", "model."]:
                    if k.startswith(prefix):
                        k = k[len(prefix):]
                        break
                cleaned[k] = v
        else:
            cleaned = ckpt

        loaded_blocks = 0
        # Load transformer blocks
        for i, block in enumerate(self.transformer.blocks):
            prefix = f"blocks.{i}."
            block_keys = {k[len(prefix):]: v for k, v in cleaned.items()
                          if k.startswith(prefix)}
            if not block_keys:
                continue

            # Map DINO block keys to TransUNet _TransformerBlock
            for key, value in block_keys.items():
                try:
                    parts = key.split(".")
                    module = block
                    for p in parts[:-1]:
                        module = getattr(module, p)
                    param = getattr(module, parts[-1])
                    if param.shape == value.shape:
                        param.data.copy_(value)
                    else:
                        print(f"  Shape mismatch at blocks.{i}.{key}: "
                              f"{param.shape} vs {value.shape}, skipping")
                except (AttributeError, RuntimeError):
                    pass
            loaded_blocks += 1

        # Load layer norm
        if "norm.weight" in cleaned:
            if self.transformer.norm.weight.shape == cleaned["norm.weight"].shape:
                self.transformer.norm.weight.data.copy_(cleaned["norm.weight"])
                self.transformer.norm.bias.data.copy_(cleaned["norm.bias"])

        # Load positional embedding (interpolate if size differs)
        if "pos_embed" in cleaned:
            old_pos = cleaned["pos_embed"]  # (1, 1+N_old, D)
            # Extract patch positions (skip CLS token)
            old_patch_pos = old_pos[:, 1:]
            old_N = old_patch_pos.shape[1]
            new_N = self.patch_embed.pos_embed.shape[1]

            if old_N == new_N and old_patch_pos.shape[-1] == self.patch_embed.pos_embed.shape[-1]:
                self.patch_embed.pos_embed.data.copy_(old_patch_pos)
            elif old_patch_pos.shape[-1] == self.patch_embed.pos_embed.shape[-1]:
                # Same embed_dim, different grid size — interpolate
                old_grid = int(old_N ** 0.5)
                new_grid = int(new_N ** 0.5)
                dim = old_patch_pos.shape[-1]
                old_patch_pos = old_patch_pos.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
                old_patch_pos = F.interpolate(old_patch_pos, size=(new_grid, new_grid),
                                              mode="bicubic", align_corners=False)
                old_patch_pos = old_patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
                self.patch_embed.pos_embed.data.copy_(old_patch_pos)
                print(f"  Interpolated pos_embed: {old_grid}x{old_grid} → {new_grid}x{new_grid}")
            else:
                print(f"  pos_embed dim mismatch ({old_patch_pos.shape[-1]} vs "
                      f"{self.patch_embed.pos_embed.shape[-1]}), skipping")

        print(f"  Loaded DINO weights into {loaded_blocks}/{len(self.transformer.blocks)} transformer blocks")

    # -----------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Zero-init the output head so model starts predicting all-black.
        # Sigmoid(0) = 0.5, so we bias to a large negative value for ~0 output.
        head_conv = self.head[0]
        nn.init.zeros_(head_conv.weight)
        nn.init.constant_(head_conv.bias, -5.0)  # Sigmoid(-5) ≈ 0.007

    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Brightfield input of shape ``(B, in_channels, H, W)``.

        Returns
        -------
        Tensor
            Predicted IF intensity map of shape ``(B, out_channels, H, W)``
            with values in [0, 1].
        """
        # CNN encoder --------------------------------------------------
        s1, s2, s3, s4 = self.encoder(x)
        # s1: (B,  64, H/2,  W/2)
        # s2: (B, 128, H/4,  W/4)
        # s3: (B, 256, H/8,  W/8)
        # s4: (B, 512, H/16, W/16)

        # Patch embedding + Transformer --------------------------------
        tokens = self.patch_embed(s4)                 # (B, N, embed_dim)
        tokens = self.transformer(tokens)             # (B, N, embed_dim)

        B, N, C = tokens.shape
        h = w = int(math.isqrt(N))
        feat = tokens.transpose(1, 2).reshape(B, C, h, w)  # (B, embed_dim, h, w)
        feat = self.decoder_proj(feat)                      # (B, 512, h, w)

        # CNN decoder with skip connections ----------------------------
        d = self.dec1(feat, s3)   # (B, 256, H/8,  W/8)
        d = self.dec2(d, s2)      # (B, 128, H/4,  W/4)
        d = self.dec3(d, s1)      # (B,  64, H/2,  W/2)
        d = self.dec4(d, None)    # (B,  32, H,    W)

        return self.head(d)       # (B, out_channels, H, W)
