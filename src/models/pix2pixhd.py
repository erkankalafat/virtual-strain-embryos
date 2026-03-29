"""Pix2PixHD model for paired BF -> IF virtual staining.

Conditional GAN with high-resolution synthesis for regression output
(continuous fluorescence intensity prediction).

Reference: Wang et al., "High-Resolution Image Synthesis and Semantic
Manipulation with Conditional GANs", CVPR 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResNetBlock(nn.Module):
    """Residual block with instance normalization."""

    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ---------------------------------------------------------------------------
# Global Generator
# ---------------------------------------------------------------------------

class GlobalGenerator(nn.Module):
    """Coarse-to-fine generator: downsample -> ResNet blocks -> upsample.

    Architecture:
        c7s1-ngf -> d(ngf*2) -> ... -> d(ngf*2^n) ->
        R(ngf*2^n) x n_blocks ->
        u(ngf*2^(n-1)) -> ... -> u(ngf) -> c7s1-out_channels -> Sigmoid
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        ngf: int = 64,
        n_downsample: int = 4,
        n_blocks: int = 9,
    ):
        super().__init__()
        assert n_blocks >= 0

        # --- Initial convolution ---
        layers: List[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        # --- Downsampling ---
        for i in range(n_downsample):
            mult = 2 ** i
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]

        # --- ResNet blocks at bottleneck ---
        mult = 2 ** n_downsample
        for _ in range(n_blocks):
            layers.append(ResNetBlock(ngf * mult))

        # --- Upsampling ---
        for i in range(n_downsample):
            mult = 2 ** (n_downsample - i)
            layers += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    ngf * mult // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(inplace=True),
            ]

        # --- Output convolution ---
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7),
            nn.Sigmoid(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Local Enhancer
# ---------------------------------------------------------------------------

class LocalEnhancer(nn.Module):
    """Optional local enhancer that refines the global generator output.

    Adds a shallow front-end / back-end around the frozen (or jointly trained)
    global generator to operate at higher spatial resolution.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        ngf: int = 64,
        n_downsample_global: int = 4,
        n_blocks_global: int = 9,
        n_blocks_local: int = 3,
    ):
        super().__init__()

        # The inner global generator (operates at 2x-downsampled resolution).
        self.global_generator = GlobalGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            ngf=ngf,
            n_downsample=n_downsample_global,
            n_blocks=n_blocks_global,
        )

        # We need access to the global generator internals to extract
        # the feature map right before the final output layer.
        # Split the global model into encoder+resblocks+decoder_body and
        # the final output head.
        global_layers = list(self.global_generator.model.children())
        # Last 3 layers: ReflectionPad2d, Conv2d (to out_channels), Sigmoid
        self.global_body = nn.Sequential(*global_layers[:-3])
        self.global_head = nn.Sequential(*global_layers[-3:])

        # --- Local front-end (one extra downsample) ---
        self.local_downsample = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
        )

        # --- Local residual blocks ---
        local_blocks: List[nn.Module] = []
        for _ in range(n_blocks_local):
            local_blocks.append(ResNetBlock(ngf * 2))
        self.local_resblocks = nn.Sequential(*local_blocks)

        # --- Local back-end (one upsample) ---
        self.local_upsample = nn.Sequential(
            nn.ConvTranspose2d(
                ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1,
            ),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        )

        # --- Output head (shared channel width with global head) ---
        self.local_head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downsample input for the global path.
        x_down = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=True)
        global_feat = self.global_body(x_down)

        # Local front-end.
        local_feat = self.local_downsample(x)

        # Fuse: add global features (upsampled to local resolution) to
        # local features at the merging point.
        local_feat = local_feat + F.interpolate(
            global_feat, size=local_feat.shape[2:], mode="bilinear", align_corners=True,
        )

        local_feat = self.local_resblocks(local_feat)
        local_feat = self.local_upsample(local_feat)
        return self.local_head(local_feat)


# ---------------------------------------------------------------------------
# Discriminator (PatchGAN)
# ---------------------------------------------------------------------------

class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator (no sigmoid -- use with appropriate loss)."""

    def __init__(self, in_channels: int, ndf: int = 64, n_layers: int = 3):
        super().__init__()

        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Second-to-last layer (stride=1).
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Final prediction layer.
        layers.append(
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiscaleDiscriminator(nn.Module):
    """Multiple PatchGAN discriminators operating at different spatial scales.

    Each subsequent discriminator receives a 2x-downsampled version of the
    input, enabling coarse-to-fine adversarial feedback.
    """

    def __init__(
        self,
        in_channels: int,
        ndf: int = 64,
        n_layers: int = 3,
        n_discriminators: int = 2,
    ):
        super().__init__()
        self.n_discriminators = n_discriminators

        self.discriminators = nn.ModuleList()
        for _ in range(n_discriminators):
            self.discriminators.append(
                NLayerDiscriminator(in_channels=in_channels, ndf=ndf, n_layers=n_layers)
            )

        self.downsample = nn.AvgPool2d(
            kernel_size=3, stride=2, padding=1, count_include_pad=False,
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return list of discriminator outputs, one per scale."""
        outputs = []
        current = x
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(current))
            if i < self.n_discriminators - 1:
                current = self.downsample(current)
        return outputs


# ---------------------------------------------------------------------------
# Top-level Pix2PixHD wrapper
# ---------------------------------------------------------------------------

class Pix2PixHD(nn.Module):
    """Pix2PixHD conditional GAN for paired BF -> IF virtual staining.

    Parameters
    ----------
    config : dict
        Expected keys under ``config["model"]``:
            in_channels : int (default 3)
            out_channels : int (default 2)
            ngf : int (default 64)
            ndf : int (default 64)
            n_downsample : int (default 4)
            n_blocks : int (default 9)
            n_discriminators : int (default 2)
            use_local_enhancer : bool (default False)
    """

    def __init__(self, config: Dict):
        super().__init__()
        mcfg = config.get("model", {})

        self.in_channels: int = mcfg.get("in_channels", 3)
        self.out_channels: int = mcfg.get("out_channels", 2)
        self.ngf: int = mcfg.get("ngf", 64)
        self.ndf: int = mcfg.get("ndf", 64)
        self.n_downsample: int = mcfg.get("n_downsample", 4)
        self.n_blocks: int = mcfg.get("n_blocks", 9)
        self.n_discriminators: int = mcfg.get("n_discriminators", 2)
        self.use_local_enhancer: bool = mcfg.get("use_local_enhancer", False)

        # Build generator.
        if self.use_local_enhancer:
            self.generator = LocalEnhancer(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                ngf=self.ngf,
                n_downsample_global=self.n_downsample,
                n_blocks_global=self.n_blocks,
            )
        else:
            self.generator = GlobalGenerator(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                ngf=self.ngf,
                n_downsample=self.n_downsample,
                n_blocks=self.n_blocks,
            )

        # Build multi-scale discriminator.
        # Discriminator sees concatenation of input + output along channels.
        disc_in_channels = self.in_channels + self.out_channels
        self.discriminator = MultiscaleDiscriminator(
            in_channels=disc_in_channels,
            ndf=self.ndf,
            n_discriminators=self.n_discriminators,
        )

    # -- Convenience accessors ------------------------------------------------

    def get_generator(self) -> nn.Module:
        """Return the generator sub-network."""
        return self.generator

    def get_discriminator(self) -> MultiscaleDiscriminator:
        """Return the multi-scale discriminator sub-network."""
        return self.discriminator

    # -- Forward --------------------------------------------------------------

    def forward(
        self, bf_input: torch.Tensor, if_target: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        """Run generator (and optionally discriminator).

        Parameters
        ----------
        bf_input : Tensor [B, in_channels, H, W]
            Bright-field input image.
        if_target : Tensor [B, out_channels, H, W], optional
            Ground-truth immunofluorescence image.  When provided the
            discriminator is evaluated on both real and fake pairs.

        Returns
        -------
        dict with keys:
            ``"fake"`` : generated IF image [B, out_channels, H, W]
            ``"disc_real"`` : list of discriminator outputs on real pairs
                              (only when *if_target* is given)
            ``"disc_fake"`` : list of discriminator outputs on fake pairs
                              (only when *if_target* is given)
        """
        fake = self.generator(bf_input)
        result: Dict[str, torch.Tensor | List[torch.Tensor]] = {"fake": fake}

        if if_target is not None:
            real_pair = torch.cat([bf_input, if_target], dim=1)
            fake_pair = torch.cat([bf_input, fake.detach()], dim=1)
            result["disc_real"] = self.discriminator(real_pair)
            result["disc_fake"] = self.discriminator(fake_pair)

        return result
