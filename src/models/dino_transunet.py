"""DINO-TransUNet: Pretrained DINO ViT-S backbone with U-Net decoder for virtual staining.

Uses a self-supervised DINO ViT-Small encoder (pretrained on live embryo
time-lapse images) as the feature extractor, paired with a CNN decoder
with skip connections for pixel-level IF intensity regression.

Architecture:
    1. DINO ViT-S encoder (frozen or fine-tuned):
       - Patch embedding: 16x16 patches → 384-dim tokens
       - 12 transformer blocks with 6 attention heads
       - Extracts multi-layer features for skip connections
    2. CNN projection layers:
       - Project transformer features at layers [3, 6, 9, 12] to spatial
         feature maps at different resolutions for skip connections
    3. U-Net decoder:
       - 4 upsampling stages with skip connections from encoder
       - Each stage: bilinear upsample → concat skip → conv blocks
    4. Output head: 1x1 conv + Sigmoid → [0, 1] intensity

The DINO backbone was pretrained on 15k live embryo brightfield images,
learning representations of cell boundaries, nuclei, and membrane
structures — exactly the features relevant for predicting IF staining.

This enables potential cross-domain generalization: the encoder understands
live-cell BF appearance even though the decoder was trained on fixed-cell
paired BF-IF data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from pathlib import Path


class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _DecoderBlock(nn.Module):
    """Upsample + concat skip + two conv blocks."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = _ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = _ConvBNReLU(out_ch, out_ch)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class _FeatureProjector(nn.Module):
    """Project transformer token features to spatial feature maps.

    Takes (B, N, D) transformer output and reshapes to (B, C, H, W),
    optionally projecting to a different channel dimension.
    """

    def __init__(self, embed_dim, out_channels, grid_size):
        super().__init__()
        self.grid_size = grid_size
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, out_channels),
            nn.LayerNorm(out_channels),
        )

    def forward(self, tokens, h, w):
        """
        Args:
            tokens: (B, N, D) — patch tokens (no CLS).
            h, w: spatial dimensions of the patch grid.
        """
        B, N, D = tokens.shape
        x = self.proj(tokens)  # (B, N, C)
        x = x.transpose(1, 2).reshape(B, -1, h, w)  # (B, C, h, w)
        return x


class DINOTransUNet(nn.Module):
    """DINO ViT-S backbone + U-Net decoder for virtual staining regression.

    Config dict keys (under ``model.*``):
        in_channels       : int, default 3
        out_channels      : int, default 2 (DAPI + Phalloidin)
        embed_dim         : int, default 384 (must match DINO checkpoint)
        patch_size        : int, default 16
        depth             : int, default 12 (ViT-S layers)
        num_heads         : int, default 6
        dino_checkpoint   : str, path to DINO checkpoint (.pth)
        freeze_encoder    : bool, default False (fine-tune encoder)
        freeze_epochs     : int, default 0 (freeze encoder for first N epochs,
                            then unfreeze — set via training loop)
        skip_layers       : list[int], default [3, 6, 9, 12]
                            Which transformer layers to tap for skip connections
        decoder_channels  : list[int], default [256, 128, 64, 32]
    """

    def __init__(self, config: dict):
        super().__init__()
        mcfg = config.get("model", {})

        self.in_channels = mcfg.get("in_channels", 3)
        self.out_channels = mcfg.get("out_channels", 2)
        self.embed_dim = mcfg.get("embed_dim", 384)
        self.patch_size = mcfg.get("patch_size", 16)
        depth = mcfg.get("depth", 12)
        num_heads = mcfg.get("num_heads", 6)
        self.skip_layers = list(mcfg.get("skip_layers", [3, 6, 9, 12]))
        decoder_channels = list(mcfg.get("decoder_channels", [256, 128, 64, 32]))
        freeze_encoder = mcfg.get("freeze_encoder", False)

        # --- Build DINO ViT-S encoder ---
        self.patch_embed = nn.Conv2d(
            self.in_channels, self.embed_dim,
            kernel_size=self.patch_size, stride=self.patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        # pos_embed will be resized dynamically
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        self.blocks = nn.ModuleList([
            _TransformerBlock(self.embed_dim, num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(self.embed_dim)

        # --- Load DINO checkpoint ---
        dino_path = mcfg.get("dino_checkpoint", None)
        if dino_path:
            self._load_dino_weights(dino_path)

        if freeze_encoder:
            self._freeze_encoder()

        # --- Feature projectors for skip connections ---
        # Skip connections at different transformer depths, projected to
        # decreasing channel dims to feed the decoder
        self.projectors = nn.ModuleList()
        for i, layer_idx in enumerate(self.skip_layers):
            out_ch = decoder_channels[i] if i < len(decoder_channels) else decoder_channels[-1]
            self.projectors.append(
                _FeatureProjector(self.embed_dim, out_ch, grid_size=0)
            )

        # --- Initial conv for high-res skip (before patch embedding) ---
        self.stem_conv = nn.Sequential(
            _ConvBNReLU(self.in_channels, 64, kernel_size=7, padding=3),
            _ConvBNReLU(64, 64),
        )

        # --- Decoder ---
        # Decoder takes the deepest skip and progressively upsamples
        self.decoder_blocks = nn.ModuleList()

        # First decoder: bottleneck (deepest projector output) → upsample
        # Subsequent decoders concatenate with shallower skip connections
        dec_in = decoder_channels[0]  # From deepest projector
        for i in range(len(decoder_channels)):
            if i == 0:
                # First block: no skip, just upsample from bottleneck
                skip_ch = decoder_channels[1] if len(decoder_channels) > 1 else 0
                self.decoder_blocks.append(
                    _DecoderBlock(dec_in, skip_ch, decoder_channels[1] if len(decoder_channels) > 1 else dec_in)
                )
                dec_in = decoder_channels[1] if len(decoder_channels) > 1 else dec_in
            elif i < len(decoder_channels) - 1:
                skip_ch = decoder_channels[i + 1]
                out_ch = decoder_channels[i + 1]
                self.decoder_blocks.append(_DecoderBlock(dec_in, skip_ch, out_ch))
                dec_in = out_ch
            else:
                # Last decoder block: skip from stem_conv (64 channels)
                self.decoder_blocks.append(_DecoderBlock(dec_in, 64, 32))
                dec_in = 32

        # Final upsample to original resolution
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _ConvBNReLU(32, 32),
        )

        # --- Output head ---
        self.head = nn.Sequential(
            nn.Conv2d(32, self.out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Initialize decoder and head
        self._init_decoder_weights()

    def _load_dino_weights(self, checkpoint_path: str):
        """Load DINO ViT-S weights from checkpoint."""
        print(f"Loading DINO checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if "student" in ckpt:
            state_dict = ckpt["student"]
        elif "teacher" in ckpt:
            state_dict = ckpt["teacher"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        # Strip prefix if present (e.g., "backbone.", "module.", "encoder.")
        cleaned = {}
        for k, v in state_dict.items():
            for prefix in ["backbone.", "module.", "encoder.", "model."]:
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            cleaned[k] = v

        # Load patch_embed
        if "patch_embed.proj.weight" in cleaned:
            self.patch_embed.weight.data.copy_(cleaned["patch_embed.proj.weight"])
            self.patch_embed.bias.data.copy_(cleaned["patch_embed.proj.bias"])

        # Load cls_token
        if "cls_token" in cleaned:
            self.cls_token.data.copy_(cleaned["cls_token"])

        # Load and resize pos_embed
        if "pos_embed" in cleaned:
            old_pos = cleaned["pos_embed"]
            self.pos_embed = nn.Parameter(old_pos)

        # Load transformer blocks
        block_keys = {}
        for k, v in cleaned.items():
            if k.startswith("blocks."):
                block_keys[k] = v

        if block_keys:
            msg = self.blocks.load_state_dict(
                {k.replace("blocks.", "", 1): v for k, v in block_keys.items()
                 if k.startswith("blocks.")},
                strict=False,
            )
            # Re-key properly
            block_sd = {}
            for k, v in block_keys.items():
                block_sd[k[len("blocks."):]] = v  # Remove "blocks." prefix once

            # Actually load block by block since ModuleList expects indexed keys
            for k, v in block_keys.items():
                parts = k.split(".")
                # blocks.0.norm1.weight -> index=0, rest=norm1.weight
                if len(parts) >= 2:
                    try:
                        idx = int(parts[1])
                        rest = ".".join(parts[2:])
                        param = self.blocks[idx]
                        for attr_name in rest.split(".")[:-1]:
                            param = getattr(param, attr_name)
                        leaf = rest.split(".")[-1]
                        if hasattr(param, leaf):
                            getattr(param, leaf).data.copy_(v)
                    except (ValueError, IndexError, AttributeError):
                        pass

        # Load norm
        if "norm.weight" in cleaned:
            self.norm.weight.data.copy_(cleaned["norm.weight"])
            self.norm.bias.data.copy_(cleaned["norm.bias"])

        print(f"  DINO weights loaded successfully")

    def _freeze_encoder(self):
        """Freeze all encoder parameters."""
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False
        for param in self.blocks.parameters():
            param.requires_grad = False
        for param in self.norm.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning."""
        for param in self.patch_embed.parameters():
            param.requires_grad = True
        self.cls_token.requires_grad = True
        self.pos_embed.requires_grad = True
        for param in self.blocks.parameters():
            param.requires_grad = True
        for param in self.norm.parameters():
            param.requires_grad = True

    def _init_decoder_weights(self):
        """Initialize decoder and output head."""
        for m in self.decoder_blocks.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        for m in self.projectors.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Output head: mild negative bias for sparse targets
        head_conv = self.head[0]
        nn.init.kaiming_normal_(head_conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(head_conv.bias, -2.0)  # Sigmoid(-2) ≈ 0.12

    def _interpolate_pos_embed(self, x, h, w):
        """Interpolate position embeddings for arbitrary input size."""
        num_patches = h * w
        N = self.pos_embed.shape[1] - 1  # Exclude CLS token

        if num_patches == N:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        dim = patch_pos.shape[-1]

        old_grid = int(N ** 0.5)
        patch_pos = patch_pos.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos, size=(h, w), mode="bicubic", align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # High-res skip from stem
        stem_skip = self.stem_conv(x)  # (B, 64, H, W)

        # Patch embedding
        patch_tokens = self.patch_embed(x)  # (B, embed_dim, h, w)
        h, w = patch_tokens.shape[2], patch_tokens.shape[3]
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2)  # (B, N, D)

        # Add CLS token and positional embedding
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patch_tokens], dim=1)
        tokens = tokens + self._interpolate_pos_embed(tokens, h, w)
        tokens = self.pos_drop(tokens)

        # Run transformer blocks and collect skip features
        skip_features = []
        for i, block in enumerate(self.blocks):
            tokens = block(tokens)
            if (i + 1) in self.skip_layers:
                # Get patch tokens (exclude CLS)
                patch_out = self.norm(tokens[:, 1:]) if (i + 1) == len(self.blocks) else tokens[:, 1:]
                proj_idx = self.skip_layers.index(i + 1)
                feat = self.projectors[proj_idx](patch_out, h, w)
                skip_features.append(feat)

        # Final norm on last layer output
        tokens = self.norm(tokens)

        # Decoder: deepest skip first, progressively upsample
        # skip_features: [layer3_feat, layer6_feat, layer9_feat, layer12_feat]
        # Reverse so we start from deepest
        skip_features = list(reversed(skip_features))

        x = skip_features[0]  # Deepest features

        for i, dec_block in enumerate(self.decoder_blocks):
            if i + 1 < len(skip_features):
                skip = skip_features[i + 1]
            elif i == len(self.decoder_blocks) - 1:
                # Last block: use stem skip, downsample to match
                skip = F.interpolate(
                    stem_skip, size=x.shape[2:] if x.shape[2] * 2 <= H else (H, W),
                    mode="bilinear", align_corners=False,
                )
                skip = F.adaptive_avg_pool2d(stem_skip, (x.shape[2] * 2, x.shape[3] * 2))
            else:
                skip = None
            x = dec_block(x, skip)

        # Final upsample to original resolution
        if x.shape[2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
            x = _ConvBNReLU(x.size(1), 32).to(x.device)(x)

        return self.head(x)


class _TransformerBlock(nn.Module):
    """Standard ViT block: LayerNorm → MHSA → LayerNorm → MLP."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)
