"""
ConvNeXt U-Net for virtual staining regression.

A U-shaped encoder-decoder architecture built entirely from ConvNeXt blocks,
combining the modern design principles of ConvNeXt (depthwise convolutions,
LayerNorm, GELU, inverted bottleneck, layer scale) with the proven U-Net
topology for dense per-pixel prediction.

Why ConvNeXt U-Net for virtual staining:
    - **No window artifacts**: unlike Swin Transformer or other window-based
      attention models, pure convolution produces spatially smooth predictions
      with no tiling seams, which is critical for quantitative fluorescence
      intensity estimation.
    - **Resolution agnostic**: there are no positional embeddings or fixed
      token grids, so the same trained model handles arbitrary input sizes
      at inference time without interpolation hacks.
    - **Efficient at high resolution**: depthwise separable convolutions
      scale linearly with spatial size (vs. quadratic for global attention),
      making training on large microscopy crops practical.
    - **Strong local feature extraction**: 7x7 depthwise convolutions capture
      fine textural cues in brightfield images that correlate with
      immunofluorescence signal, without the information loss of aggressive
      patching.

Input : 3-channel brightfield image  (B, C_in, H, W)
Output: continuous IF intensity map  (B, C_out, H, W) in [0, 1]
"""

from __future__ import annotations

import math
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 0.02):
    """Truncated normal initialization (in-place)."""
    with torch.no_grad():
        nn.init.trunc_normal_(tensor, mean=mean, std=std, a=-2 * std, b=2 * std)


class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm (B, C, H, W) -- used throughout ConvNeXt."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization.

    During training, each residual branch is randomly dropped with probability
    ``drop_prob``, scaling the surviving branches by 1/(1 - drop_prob).
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        # Per-sample binary mask (broadcast over spatial + channel dims)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep


# ---------------------------------------------------------------------------
# ConvNeXt Block
# ---------------------------------------------------------------------------

class ConvNeXtBlock(nn.Module):
    """Single ConvNeXt block.

    Architecture:
        depthwise conv 7x7 -> LayerNorm -> 1x1 conv (expand 4x) -> GELU
        -> 1x1 conv (project back) -> Layer Scale -> (residual + Drop Path)
    """

    def __init__(self, dim: int, drop_path: float = 0.0,
                 layer_scale_init: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

        # Layer Scale: learnable per-channel scaling
        self.gamma = nn.Parameter(
            layer_scale_init * torch.ones(dim)
        ) if layer_scale_init > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma[:, None, None] * x
        x = shortcut + self.drop_path(x)
        return x


# ---------------------------------------------------------------------------
# Encoder / Decoder stages
# ---------------------------------------------------------------------------

class DownsampleLayer(nn.Module):
    """Spatial downsampling: LayerNorm -> 2x2 strided conv (no pooling)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = LayerNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


class UpsampleLayer(nn.Module):
    """Spatial upsampling: bilinear 2x upsample -> 3x3 conv (no transposed conv,
    avoids checkerboard artifacts)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = LayerNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(x)
        x = self.norm(x)
        return x


class EncoderStage(nn.Module):
    """Stack of ConvNeXt blocks for one encoder resolution level."""

    def __init__(self, dim: int, depth: int, drop_path_rates: List[float],
                 layer_scale_init: float = 1e-6):
        super().__init__()
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(dim, drop_path=drop_path_rates[i],
                          layer_scale_init=layer_scale_init)
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class DecoderStage(nn.Module):
    """One decoder level: upsample, concatenate skip, fuse, then ConvNeXt blocks."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 depth: int, drop_path_rates: List[float],
                 layer_scale_init: float = 1e-6):
        super().__init__()
        self.upsample = UpsampleLayer(in_channels, out_channels)
        # 1x1 conv to fuse concatenated features (upsample output + skip)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=1),
            LayerNorm2d(out_channels),
        )
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(out_channels, drop_path=drop_path_rates[i],
                          layer_scale_init=layer_scale_init)
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Handle size mismatches from odd input dimensions
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.blocks(x)
        return x


# ---------------------------------------------------------------------------
# ConvNeXt U-Net
# ---------------------------------------------------------------------------

class ConvNeXtUNet(nn.Module):
    """ConvNeXt U-Net for virtual staining regression.

    Brightfield (3-ch) -> continuous IF intensity (1 or 2 ch, configurable).
    Output is in [0, 1] (sigmoid activation).

    Config dict keys (all under ``model.*``):
        in_channels       : int, default 3
        out_channels      : int, default 2
        dims              : list[int], default [96, 192, 384, 768]
        depths            : list[int], default [3, 3, 9, 3]
        drop_path_rate    : float, default 0.0
        layer_scale_init  : float, default 1e-6
    """

    def __init__(self, config: dict):
        super().__init__()
        mcfg = config.get("model", {})
        in_channels: int = mcfg.get("in_channels", 3)
        out_channels: int = mcfg.get("out_channels", 2)
        dims: List[int] = list(mcfg.get("dims", [96, 192, 384, 768]))
        depths: List[int] = list(mcfg.get("depths", [3, 3, 9, 3]))
        drop_path_rate: float = mcfg.get("drop_path_rate", 0.0)
        layer_scale_init: float = mcfg.get("layer_scale_init", 1e-6)

        assert len(dims) == 4, f"Expected 4 stage dims, got {len(dims)}"
        assert len(depths) == 4, f"Expected 4 stage depths, got {len(depths)}"

        self.num_stages = 4

        # -- Stochastic depth schedule (linearly increasing across all blocks)
        total_blocks = sum(depths) + depths[-1]  # encoder + bottleneck
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        block_idx = 0

        # -- Stem: aggressive 4x4 stride-4 convolution (like ConvNeXt patchify)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )

        # -- Encoder stages + downsample layers
        self.encoder_stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        for i in range(self.num_stages):
            stage = EncoderStage(
                dim=dims[i],
                depth=depths[i],
                drop_path_rates=dp_rates[block_idx:block_idx + depths[i]],
                layer_scale_init=layer_scale_init,
            )
            block_idx += depths[i]
            self.encoder_stages.append(stage)

            if i < self.num_stages - 1:
                self.downsample_layers.append(
                    DownsampleLayer(dims[i], dims[i + 1])
                )

        # -- Bottleneck: extra ConvNeXt blocks at the coarsest resolution
        bottleneck_depth = depths[-1]
        self.bottleneck = nn.Sequential(*[
            ConvNeXtBlock(dims[-1], drop_path=dp_rates[block_idx + j],
                          layer_scale_init=layer_scale_init)
            for j in range(bottleneck_depth)
        ])
        block_idx += bottleneck_depth

        # -- Decoder stages (mirror encoder, with skip connections)
        # Decoder drop-path rates are kept at 0 for stable gradient flow
        self.decoder_stages = nn.ModuleList()
        decoder_depths = list(reversed(depths))  # mirror encoder depths
        decoder_dims = list(reversed(dims))       # [768, 384, 192, 96]

        for i in range(self.num_stages):
            dec_in = decoder_dims[i]
            skip_ch = decoder_dims[i + 1] if i < self.num_stages - 1 else decoder_dims[i]
            dec_out = decoder_dims[i + 1] if i < self.num_stages - 1 else decoder_dims[i]

            # Skip connections come from encoder stage at matching resolution
            # Encoder stage 0 -> decoder stage 3 (last), etc.
            enc_idx = self.num_stages - 1 - i
            skip_channels = dims[enc_idx]

            # First decoder stage takes bottleneck output
            stage_in = dec_in

            # Decoder dims: we upsample to the next (coarser->finer) channel count
            if i < self.num_stages - 1:
                stage_out = dims[self.num_stages - 2 - i]
            else:
                stage_out = dims[0]

            self.decoder_stages.append(
                DecoderStage(
                    in_channels=stage_in,
                    skip_channels=skip_channels,
                    out_channels=stage_out,
                    depth=decoder_depths[i],
                    drop_path_rates=[0.0] * decoder_depths[i],
                    layer_scale_init=layer_scale_init,
                )
            )

        # -- Upsample from stem resolution back to original resolution (4x)
        self.final_upsample = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], kernel_size=3, padding=1),
            LayerNorm2d(dims[0]),
            nn.GELU(),
        )

        # -- Output head: 1x1 conv + Sigmoid
        self.head = nn.Conv2d(dims[0], out_channels, kernel_size=1)

        # -- Weight initialization
        self.apply(self._init_weights)
        # Output head bias: Sigmoid(-2) ~ 0.12, mild negative for sparse targets
        nn.init.kaiming_normal_(self.head.weight, mode="fan_out", nonlinearity="sigmoid")
        nn.init.constant_(self.head.bias, -2.0)

    @staticmethod
    def _init_weights(m: nn.Module):
        """Initialize weights following ConvNeXt conventions."""
        if isinstance(m, nn.Conv2d):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C_in, H, W) brightfield input.

        Returns:
            (B, C_out, H, W) predicted IF intensity in [0, 1].
        """
        input_size = x.shape[-2:]

        # Stem: 4x downsample
        x = self.stem(x)

        # Encoder: collect skip features at each resolution
        skips: List[torch.Tensor] = []
        for i in range(self.num_stages):
            x = self.encoder_stages[i](x)
            skips.append(x)
            if i < self.num_stages - 1:
                x = self.downsample_layers[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder: upsample + skip connections (deepest skip is from last encoder stage)
        for i in range(self.num_stages):
            # Skip from matching encoder stage (reverse order)
            enc_idx = self.num_stages - 1 - i
            skip = skips[enc_idx]
            x = self.decoder_stages[i](x, skip)

        # Upsample 4x back to original resolution
        x = self.final_upsample(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        # Output head
        x = self.head(x)
        x = torch.sigmoid(x)
        return x
