"""
Swin-UNet for virtual staining regression.

A U-shaped architecture with Swin Transformer blocks in both encoder and
decoder.  Window-based multi-head self-attention (W-MSA) and shifted-window
MSA (SW-MSA) capture fine local details that are critical for detecting
subtle immunofluorescence signals from brightfield input.

Input : 3-channel brightfield image  (B, C_in, H, W)
Output: continuous IF intensity map  (B, C_out, H, W) in [0, 1]
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_2tuple(x):
    return (x, x) if isinstance(x, int) else tuple(x)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition feature map into non-overlapping windows.

    Args:
        x: (B, H, W, C)
        window_size: window height/width

    Returns:
        windows: (num_windows * B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse of :func:`window_partition`.

    Args:
        windows: (num_windows * B, window_size, window_size, C)
        window_size: window height/width
        H, W: original spatial dimensions

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ---------------------------------------------------------------------------
# Core Swin blocks
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(self, in_features: int, hidden_features: int | None = None,
                 out_features: int | None = None, drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention (W-MSA / SW-MSA).

    Supports both regular and shifted windows via an attention mask.
    """

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table: (2*Wh-1) * (2*Ww-1), nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute pair-wise relative position index for each token in window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 2
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # N, N
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (num_windows*B, N, C)  where N = window_size^2
            mask: (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_, nH, N, N)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class SwinTransformerBlock(nn.Module):
    """A single Swin Transformer block with W-MSA or SW-MSA."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 7,
                 shift_size: int = 0, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0,
                 attn_drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C)
            H, W: spatial dimensions
        """
        B, L, C = x.shape
        assert L == H * W, "Input feature size mismatch"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad feature map to be divisible by window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift for SW-MSA
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self._compute_mask(Hp, Wp, x.device)
        else:
            shifted_x = x
            attn_mask = None

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, ws, ws, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def _compute_mask(self, Hp: int, Wp: int, device: torch.device) -> torch.Tensor:
        """Compute attention mask for SW-MSA."""
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularisation."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


# ---------------------------------------------------------------------------
# Patch Embedding / Merging / Expanding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Image to patch embedding via convolution."""

    def __init__(self, img_size: int = 512, patch_size: int = 4,
                 in_channels: int = 3, embed_dim: int = 96):
        super().__init__()
        img_size = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = (img_size[0] // patch_size[0],
                                   img_size[1] // patch_size[1])
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x: (B, H'*W', embed_dim), H', W'
        """
        x = self.proj(x)  # (B, embed_dim, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', C)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    """Merge 2x2 neighbouring patches -> halve spatial, double channels."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        B, L, C = x.shape
        assert L == H * W

        x = x.view(B, H, W, C)
        # Pad if needed
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H += pad_h
            W += pad_w

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)

        H_out, W_out = H // 2, W // 2
        x = x.view(B, H_out * W_out, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, H_out, W_out


class PatchExpanding(nn.Module):
    """Expand patches: double spatial, halve channels (inverse of PatchMerging)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        B, L, C = x.shape
        assert L == H * W

        x = self.expand(x)  # (B, H*W, 2C)
        x = x.view(B, H, W, 2 * C)

        # Rearrange into 2x spatial with C/2 channels
        # (B, H, W, 2C) -> (B, 2H, 2W, C/2)
        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, 2 * H, 2 * W, C // 2)

        H_out, W_out = 2 * H, 2 * W
        x = self.norm(x)
        x = x.view(B, H_out * W_out, C // 2)
        return x, H_out, W_out


class FinalPatchExpand(nn.Module):
    """Final 4x up-sampling to recover original resolution from patch tokens."""

    def __init__(self, dim: int, patch_size: int = 4):
        super().__init__()
        self.up_scale = patch_size
        self.expand = nn.Linear(dim, (patch_size ** 2) * dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C)
        Returns:
            x: (B, H*up, W*up, C)
        """
        B, L, C = x.shape
        x = self.expand(x)  # (B, H*W, up^2 * C)
        x = x.view(B, H, W, self.up_scale, self.up_scale, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * self.up_scale, W * self.up_scale, C)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Encoder / Decoder stages
# ---------------------------------------------------------------------------

class SwinEncoderStage(nn.Module):
    """One encoder stage: stack of Swin blocks + optional down-sample."""

    def __init__(self, dim: int, depth: int, num_heads: int,
                 window_size: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0,
                 attn_drop: float = 0.0, drop_path: float | List[float] = 0.0,
                 downsample: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            )
            for i in range(depth)
        ])
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x: torch.Tensor, H: int, W: int):
        for blk in self.blocks:
            x = blk(x, H, W)
        skip = x
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, H, W, skip


class SwinDecoderStage(nn.Module):
    """One decoder stage: patch-expand up-sample + concat skip + Swin blocks."""

    def __init__(self, dim: int, depth: int, num_heads: int,
                 window_size: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0,
                 attn_drop: float = 0.0, drop_path: float | List[float] = 0.0):
        super().__init__()
        self.upsample = PatchExpanding(dim)
        # After expanding: channels = dim // 2
        # After concat with skip: channels = dim // 2 + dim // 2 = dim
        # Linear to fuse back to dim // 2
        self.concat_linear = nn.Linear(dim, dim // 2)
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim // 2, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor, skip: torch.Tensor,
                H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        x, H, W = self.upsample(x, H, W)
        x = torch.cat([x, skip], dim=-1)  # (B, H*W, dim)
        x = self.concat_linear(x)
        for blk in self.blocks:
            x = blk(x, H, W)
        return x, H, W


# ---------------------------------------------------------------------------
# Full Swin-UNet
# ---------------------------------------------------------------------------

class SwinUNetRegressor(nn.Module):
    """Swin-UNet for virtual staining regression.

    Brightfield (3-ch) -> continuous IF intensity (1 or 2 ch, configurable).
    Output is in [0, 1] (sigmoid activation).

    Config dict keys (all under ``model.*``):
        in_channels   : int, default 3
        out_channels  : int, default 2
        img_size      : int, default 512
        embed_dim     : int, default 96
        depths        : list[int], default [2, 2, 6, 2]
        num_heads     : list[int], default [3, 6, 12, 24]
        window_size   : int, default 8
        patch_size    : int, default 4
    """

    def __init__(self, config: dict):
        super().__init__()
        mcfg = config.get("model", {})
        in_channels: int = mcfg.get("in_channels", 3)
        out_channels: int = mcfg.get("out_channels", 2)
        img_size: int = mcfg.get("img_size", 512)
        embed_dim: int = mcfg.get("embed_dim", 96)
        depths: List[int] = list(mcfg.get("depths", [2, 2, 6, 2]))
        num_heads: List[int] = list(mcfg.get("num_heads", [3, 6, 12, 24]))
        window_size: int = mcfg.get("window_size", 8)
        patch_size: int = mcfg.get("patch_size", 4)

        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        mlp_ratio = 4.0
        drop_path_rate = 0.1

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ---- Patch embedding ----
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_channels=in_channels, embed_dim=embed_dim,
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.pos_drop = nn.Dropout(p=0.0)

        # ---- Encoder stages ----
        self.encoder_stages = nn.ModuleList()
        for i in range(self.num_stages):
            dim_i = embed_dim * (2 ** i)
            stage = SwinEncoderStage(
                dim=dim_i,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop_path=[dpr[sum(depths[:i]) + j] for j in range(depths[i])],
                downsample=(i < self.num_stages - 1),  # no downsample at bottleneck
            )
            self.encoder_stages.append(stage)

        # ---- Bottleneck norm ----
        bottleneck_dim = embed_dim * (2 ** (self.num_stages - 1))
        self.bottleneck_norm = nn.LayerNorm(bottleneck_dim)

        # ---- Decoder stages (mirror of encoder, excluding bottleneck) ----
        self.decoder_stages = nn.ModuleList()
        for i in range(self.num_stages - 1):
            # Decoder goes from deepest to shallowest
            dec_idx = self.num_stages - 2 - i  # corresponding encoder index
            dim_dec = embed_dim * (2 ** (dec_idx + 1))  # input dim at this decoder level
            stage = SwinDecoderStage(
                dim=dim_dec,
                depth=depths[dec_idx],
                num_heads=num_heads[dec_idx],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop_path=[dpr[sum(depths[:dec_idx]) + j] for j in range(depths[dec_idx])],
            )
            self.decoder_stages.append(stage)

        # ---- Final up-sample + projection ----
        self.final_expand = FinalPatchExpand(dim=embed_dim, patch_size=patch_size)
        self.output_proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W) brightfield image
        Returns:
            out: (B, C_out, H, W) predicted IF intensity in [0, 1]
        """
        # Patch embedding
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        # Encoder
        skips: List[torch.Tensor] = []
        skip_hw: List[Tuple[int, int]] = []
        for stage in self.encoder_stages:
            x, H, W, skip = stage(x, H, W)
            skips.append(skip)
            skip_hw.append((H * 2 if stage.downsample is not None else H,
                            W * 2 if stage.downsample is not None else W))

        x = self.bottleneck_norm(x)

        # Decoder (use skips in reverse, excluding the last/bottleneck skip)
        for i, stage in enumerate(self.decoder_stages):
            skip_idx = self.num_stages - 2 - i
            skip = skips[skip_idx]
            x, H, W = stage(x, skip, H, W)

        # Final expansion to full resolution
        x = self.final_expand(x, H, W)  # (B, H_full, W_full, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        x = self.output_proj(x)
        x = self.sigmoid(x)
        return x
