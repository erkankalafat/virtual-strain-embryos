"""
AdaIN-based style transfer model for virtual staining microscopy.

Two primary use cases in the virtual-strain-embryos pipeline:

1. **BF -> IF style transfer** (paired training)
   Train with brightfield content images and immunofluorescence style
   references.  The model learns a decoder that, given BF content
   features re-normalised to IF statistics, produces a plausible IF
   image.  Supervised with content + style losses on paired data.

2. **Fixed BF -> Live BF domain adaptation** (preprocessing)
   When the training distribution (e.g. fixed-sample BF) differs from
   deployment (live BF), this model can serve as a lightweight domain
   adapter that maps live-BF statistics onto the fixed-BF manifold,
   improving downstream virtual-stain quality.

Two feature-alignment strategies are provided:

* **AdaIN (Adaptive Instance Normalization)** -- Huang & Belongie 2017.
  Per-channel, shift the content features to have the same mean and
  standard deviation as the style features.  Fast and effective for
  global style, but only matches first- and second-order statistics.

* **EFDM (Exact Feature Distribution Matching)** -- Zhang et al. 2022.
  Sort the content and style feature vectors channel-wise, then
  replace content values with the style values at the same rank
  position (histogram matching in feature space).  This aligns the
  *full marginal distribution* per channel rather than just mean/var,
  giving higher-fidelity transfers at modest extra cost.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _adaptive_instance_norm(
    content_feat: torch.Tensor,
    style_feat: torch.Tensor,
) -> torch.Tensor:
    """Adaptive Instance Normalisation (AdaIN).

    For each sample in the batch and each channel, normalise *content_feat* to
    zero mean / unit variance, then re-scale and re-shift to match the
    per-instance channel statistics of *style_feat*.

    Parameters
    ----------
    content_feat : (B, C, H, W)
    style_feat   : (B, C, H', W')  -- spatial dims may differ.

    Returns
    -------
    Tensor of same shape as *content_feat*.
    """
    eps = 1e-5
    # content statistics
    c_mean = content_feat.mean(dim=(2, 3), keepdim=True)
    c_std = content_feat.std(dim=(2, 3), keepdim=True) + eps
    # style statistics
    s_mean = style_feat.mean(dim=(2, 3), keepdim=True)
    s_std = style_feat.std(dim=(2, 3), keepdim=True) + eps

    normalised = (content_feat - c_mean) / c_std
    return normalised * s_std + s_mean


def _exact_feature_distribution_matching(
    content_feat: torch.Tensor,
    style_feat: torch.Tensor,
) -> torch.Tensor:
    """Exact Feature Distribution Matching (EFDM).

    Instead of matching only mean and variance (as in AdaIN), EFDM matches the
    *full marginal distribution* per channel via sorting-based histogram
    matching in feature space.

    Algorithm (per channel, per sample):
        1. Flatten spatial dims of content and style features.
        2. Sort both vectors independently.
        3. If content and style have different spatial sizes, interpolate the
           sorted style vector to match the content length.
        4. Assign content pixels the style value at the same rank position
           (i.e., replace the content value that had rank *k* with the style
           value at rank *k*).

    This preserves the *spatial structure* of the content (rank order encodes
    relative intensity layout) while imposing the exact marginal distribution
    of the style.

    Parameters
    ----------
    content_feat : (B, C, H, W)
    style_feat   : (B, C, H', W')

    Returns
    -------
    Tensor of same shape as *content_feat*.
    """
    B, C, H, W = content_feat.shape
    # Flatten spatial dims -> (B, C, N)
    c_flat = content_feat.reshape(B, C, -1)
    s_flat = style_feat.reshape(B, C, -1)

    N_c = c_flat.shape[2]
    N_s = s_flat.shape[2]

    # Sort content and style along the spatial dimension
    c_sorted, c_indices = c_flat.sort(dim=2)
    s_sorted, _ = s_flat.sort(dim=2)

    # If spatial sizes differ, interpolate the sorted style to match content
    if N_c != N_s:
        # (B, C, N_s) -> (B*C, 1, N_s) for interpolation, then back
        s_sorted = s_sorted.reshape(B * C, 1, N_s)
        s_sorted = F.interpolate(s_sorted, size=N_c, mode="linear", align_corners=False)
        s_sorted = s_sorted.reshape(B, C, N_c)

    # Build output: place style values at the rank positions of the content
    out_flat = torch.zeros_like(c_flat)
    out_flat.scatter_(dim=2, index=c_indices, src=s_sorted)

    return out_flat.reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# VGG-19 multi-level encoder
# ---------------------------------------------------------------------------

# Default extraction points mirror the classic neural-style layers:
#   relu1_1, relu2_1, relu3_1, relu4_1  (indices into vgg19.features)
_VGG19_SLICE_INDICES: List[int] = [1, 6, 11, 20]


class VGGEncoder(nn.Module):
    """Frozen VGG-19 encoder that returns multi-level feature maps.

    Features are extracted after selected ReLU layers (by default
    relu1_1 / relu2_1 / relu3_1 / relu4_1).  All parameters are frozen
    so the encoder acts as a fixed perceptual feature extractor.

    The input is expected in [0, 1] and is internally normalised to
    ImageNet statistics.
    """

    def __init__(self, slice_indices: Optional[List[int]] = None) -> None:
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice_indices = slice_indices or _VGG19_SLICE_INDICES

        # Split VGG into sequential slices so we can tap intermediate outputs
        slices: List[nn.Sequential] = []
        prev = 0
        for idx in self.slice_indices:
            slices.append(nn.Sequential(*list(vgg.children())[prev: idx + 1]))
            prev = idx + 1
        self.slices = nn.ModuleList(slices)

        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalisation constants
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        )

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise [0,1] input to ImageNet statistics."""
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return a list of feature maps, one per extraction point.

        Parameters
        ----------
        x : (B, 3, H, W) in [0, 1].

        Returns
        -------
        List of tensors, shapes vary by spatial resolution.
        """
        # If input has fewer than 3 channels, replicate to 3
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 2:
            x = torch.cat([x, x[:, :1]], dim=1)

        h = self._normalise(x)
        features: List[torch.Tensor] = []
        for s in self.slices:
            h = s(h)
            features.append(h)
        return features


# ---------------------------------------------------------------------------
# Learnable decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """Mirror-of-VGG decoder that reconstructs an image from AdaIN/EFDM features.

    Architecture: a sequence of ``Upsample -> Conv3x3 -> ReLU`` blocks that
    progressively upsample from the deepest feature resolution back to the
    original image size, followed by a final 1x1 conv to map to the desired
    number of output channels.

    Reflection padding is used to avoid border artefacts.
    """

    def __init__(
        self,
        in_channels: int = 512,
        decoder_channels: Sequence[int] = (256, 128, 64),
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_ch = in_channels
        for ch in decoder_channels:
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.ReflectionPad2d(1),
                nn.Conv2d(prev_ch, ch, kernel_size=3),
                nn.ReLU(inplace=True),
            ]
            prev_ch = ch

        # Final projection to output channels (no activation -- sigmoid applied outside)
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(prev_ch, out_channels, kernel_size=3),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class AdaINStyleTransfer(nn.Module):
    """Style-transfer model with AdaIN / EFDM alignment for virtual staining.

    Parameters (via *config* dict)
    -------------------------------
    model.in_channels : int
        Number of input channels for content images (default 3).
    model.out_channels : int
        Number of output image channels (2 for two-channel IF, 3 for RGB).
    model.encoder_type : str
        Encoder backbone -- currently only ``'vgg19'`` is supported.
    model.decoder_channels : list[int]
        Channel widths for each decoder stage (default ``[256, 128, 64]``).
        The decoder mirrors the encoder in reverse, progressively upsampling.
    model.alignment : str
        ``'adain'`` (default) or ``'efdm'``.

    Usage
    -----
    >>> cfg = {"model": {"in_channels": 3, "out_channels": 3,
    ...                   "encoder_type": "vgg19",
    ...                   "decoder_channels": [256, 128, 64]}}
    >>> model = AdaINStyleTransfer(cfg)
    >>> out = model.transfer(content_img, style_img, alpha=1.0)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        mcfg = config["model"]

        self.in_channels: int = mcfg.get("in_channels", 3)
        self.out_channels: int = mcfg.get("out_channels", 3)
        encoder_type: str = mcfg.get("encoder_type", "vgg19")
        decoder_channels: List[int] = mcfg.get("decoder_channels", [256, 128, 64])
        self.alignment: str = mcfg.get("alignment", "adain")

        if encoder_type != "vgg19":
            raise ValueError(
                f"Unsupported encoder_type '{encoder_type}'. Only 'vgg19' is implemented."
            )

        # Frozen encoder
        self.encoder = VGGEncoder()

        # Decoder input channels = deepest encoder feature channels (512 for relu4_1)
        self.decoder = Decoder(
            in_channels=512,
            decoder_channels=decoder_channels,
            out_channels=self.out_channels,
        )

        # Choose alignment function
        if self.alignment == "efdm":
            self._align_fn = _exact_feature_distribution_matching
        else:
            self._align_fn = _adaptive_instance_norm

    # ----- alignment helpers ------------------------------------------------

    def align_features(
        self,
        content_feat: torch.Tensor,
        style_feat: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Align *content_feat* to *style_feat* statistics.

        Parameters
        ----------
        content_feat, style_feat : (B, C, H, W)
        alpha : float in [0, 1]
            Interpolation strength.  ``1.0`` = full style transfer,
            ``0.0`` = content only (identity).

        Returns
        -------
        Blended feature tensor of same shape as *content_feat*.
        """
        aligned = self._align_fn(content_feat, style_feat)
        if alpha < 1.0:
            aligned = (1.0 - alpha) * content_feat + alpha * aligned
        return aligned

    # ----- losses -----------------------------------------------------------

    def content_loss(
        self,
        output_feats: List[torch.Tensor],
        content_feats: List[torch.Tensor],
    ) -> torch.Tensor:
        """L1 content loss between encoder features of the generated output
        and the original content image.

        Only the deepest feature level (relu4_1) is used by default, matching
        the standard AdaIN formulation.
        """
        return F.l1_loss(output_feats[-1], content_feats[-1])

    def style_loss(
        self,
        output_feats: List[torch.Tensor],
        style_feats: List[torch.Tensor],
    ) -> torch.Tensor:
        """Multi-level style loss: match channel-wise mean and std between
        encoder features of the generated output and the style reference.

        Computed across *all* extraction levels so that both low- and
        high-level texture statistics are aligned.
        """
        loss = torch.tensor(0.0, device=output_feats[0].device)
        for o_feat, s_feat in zip(output_feats, style_feats):
            o_mean = o_feat.mean(dim=(2, 3))
            o_std = o_feat.std(dim=(2, 3))
            s_mean = s_feat.mean(dim=(2, 3))
            s_std = s_feat.std(dim=(2, 3))
            loss = loss + F.l1_loss(o_mean, s_mean) + F.l1_loss(o_std, s_std)
        return loss

    # ----- forward / training -----------------------------------------------

    def forward(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        alpha: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass.

        Parameters
        ----------
        content : (B, C_in, H, W) content image in [0, 1].
        style   : (B, C_in, H, W) style reference in [0, 1].
        alpha   : Style interpolation strength.

        Returns
        -------
        output        : (B, C_out, H, W) generated image clipped to [0, 1].
        content_loss  : scalar tensor.
        style_loss    : scalar tensor.
        """
        with torch.no_grad():
            content_feats = self.encoder(content)
            style_feats = self.encoder(style)

        # Align deepest features
        aligned = self.align_features(content_feats[-1], style_feats[-1], alpha=alpha)

        # Decode
        output = self.decoder(aligned)

        # Resize output to match content spatial size if needed
        if output.shape[2:] != content.shape[2:]:
            output = F.interpolate(
                output, size=content.shape[2:], mode="bilinear", align_corners=False
            )

        output = output.clamp(0.0, 1.0)

        # Compute losses -- encoder is already frozen (requires_grad=False on
        # all params), but we must NOT wrap this call in torch.no_grad() because
        # the computation graph through the encoder is needed to backpropagate
        # the content/style losses into the decoder weights.
        output_feats = self.encoder(output)

        c_loss = self.content_loss(output_feats, content_feats)
        s_loss = self.style_loss(output_feats, style_feats)

        return output, c_loss, s_loss

    # ----- inference --------------------------------------------------------

    @torch.no_grad()
    def transfer(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Run style transfer at inference time.

        Parameters
        ----------
        content_img : (B, C_in, H, W) in [0, 1].
        style_img   : (B, C_in, H, W) in [0, 1].
        alpha : float
            Interpolation between content (0.0) and full style (1.0).

        Returns
        -------
        (B, C_out, H, W) stylised image clipped to [0, 1].
        """
        content_feats = self.encoder(content_img)
        style_feats = self.encoder(style_img)

        aligned = self.align_features(content_feats[-1], style_feats[-1], alpha=alpha)
        output = self.decoder(aligned)

        if output.shape[2:] != content_img.shape[2:]:
            output = F.interpolate(
                output, size=content_img.shape[2:], mode="bilinear", align_corners=False
            )

        return output.clamp(0.0, 1.0)
