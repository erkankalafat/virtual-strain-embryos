"""
Dataset loader for paired Brightfield-Immunofluorescence embryo images.

Expects folder structure per embryo project:
    project_<name>/
        BF/             # Brightfield images
        DAPI/           # DAPI channel (nuclear stain)
        Phalloidin/     # Phalloidin channel (actin/cytoskeleton)
        DAPI+Phalloidin/  # Combined IF channels
        Overlay/        # BF + IF composite
        MetaData/       # Imaging metadata (ignored by loader)

All IF images are from the same focal plane as BF, with matching filenames
(modulo channel suffix).
"""

import os
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


def load_image(path):
    """Load a TIFF or standard image file as a numpy float32 array in [0, 1]."""
    path = str(path)
    if path.lower().endswith((".tif", ".tiff")) and HAS_TIFFFILE:
        img = tifffile.imread(path).astype(np.float32)
    else:
        img = np.array(Image.open(path)).astype(np.float32)

    # Normalize to [0, 1]
    if img.max() > 1.0:
        if img.max() > 255:
            # 16-bit image
            img = img / 65535.0
        else:
            img = img / 255.0
    return img


def extract_z_index(filename):
    """Extract z-slice index from filename like ..._z00_ch01.tif -> 0."""
    match = re.search(r"_z(\d+)", filename)
    return int(match.group(1)) if match else -1


def extract_base_name(filename):
    """Extract base name without channel suffix for matching across folders.

    e.g. 'project_Embryo2_..._z00_ch01.tif' -> 'project_Embryo2_..._z00'
         'project_Embryo2_..._z00.tif' -> 'project_Embryo2_..._z00'
    """
    stem = Path(filename).stem
    # Remove channel suffix if present
    stem = re.sub(r"_ch\d+$", "", stem)
    return stem


class EmbryoDataset(Dataset):
    """Paired BF-IF dataset for virtual staining regression.

    Args:
        root_dir: Path to folder containing all project_* embryo directories.
        target_channels: Which IF channels to predict. Options:
            'dapi' -> single channel (DAPI)
            'phalloidin' -> single channel (Phalloidin)
            'both' -> 2-channel output (DAPI + Phalloidin)
            'combined' -> single channel (DAPI+Phalloidin composite)
            'overlay' -> 3-channel (full overlay)
        img_size: Resize images to this size (square).
        augment: Whether to apply data augmentation.
        z_slices: Optional list of z-indices to include. None = all.
    """

    # Subfolder names (case-insensitive matching)
    FOLDER_MAP = {
        "bf": "BF",
        "dapi": "DAPI",
        "phalloidin": "Phalloidin",
        "combined": "DAPI+Phalloidin",
        "overlay": "Overlay",
    }

    def __init__(
        self,
        root_dir,
        target_channels="both",
        img_size=512,
        augment=False,
        z_slices=None,
    ):
        self.root_dir = Path(root_dir)
        self.target_channels = target_channels
        self.img_size = img_size
        self.augment = augment
        self.z_slices = z_slices

        self.pairs = self._discover_pairs()
        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No image pairs found in {root_dir}. "
                "Expected project_* folders with BF/ and IF subfolders."
            )

    def _find_subfolder(self, project_dir, key):
        """Find subfolder by case-insensitive match."""
        expected = self.FOLDER_MAP[key]
        for d in project_dir.iterdir():
            if d.is_dir() and d.name.lower() == expected.lower():
                return d
        return None

    def _discover_pairs(self):
        """Walk all project folders and match BF images to IF targets."""
        pairs = []

        # Find all project directories
        project_dirs = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        # If root itself contains BF/, treat it as a single project
        if any((self.root_dir / name).is_dir()
               for name in ["BF", "bf", "Bf"]):
            project_dirs = [self.root_dir]

        for proj_dir in project_dirs:
            bf_dir = self._find_subfolder(proj_dir, "bf")
            if bf_dir is None:
                continue

            # Get target directories based on requested channels
            target_dirs = self._get_target_dirs(proj_dir)
            if not target_dirs:
                continue

            # Index BF files by base name
            bf_files = {}
            for f in sorted(bf_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
                    base = extract_base_name(f.name)
                    z_idx = extract_z_index(f.name)
                    if self.z_slices is not None and z_idx not in self.z_slices:
                        continue
                    bf_files[base] = f

            # Match with target files
            for base_name, bf_path in bf_files.items():
                target_paths = []
                matched = True
                for tgt_dir in target_dirs:
                    # Find matching file in target dir
                    found = None
                    for f in tgt_dir.iterdir():
                        if extract_base_name(f.name) == base_name:
                            found = f
                            break
                    if found is None:
                        matched = False
                        break
                    target_paths.append(found)

                if matched:
                    pairs.append({
                        "bf": bf_path,
                        "targets": target_paths,
                        "base_name": base_name,
                        "project": proj_dir.name,
                    })

        return pairs

    def _get_target_dirs(self, proj_dir):
        """Return list of target directories based on target_channels setting."""
        if self.target_channels == "both":
            dirs = []
            for key in ["dapi", "phalloidin"]:
                d = self._find_subfolder(proj_dir, key)
                if d is None:
                    return []
                dirs.append(d)
            return dirs
        elif self.target_channels in ("dapi", "phalloidin", "combined", "overlay"):
            d = self._find_subfolder(proj_dir, self.target_channels)
            return [d] if d else []
        else:
            raise ValueError(f"Unknown target_channels: {self.target_channels}")

    def _apply_augmentation(self, bf_tensor, target_tensor):
        """Apply synchronized augmentation to BF and target."""
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            bf_tensor = TF.hflip(bf_tensor)
            target_tensor = TF.hflip(target_tensor)

        # Random vertical flip
        if torch.rand(1) > 0.5:
            bf_tensor = TF.vflip(bf_tensor)
            target_tensor = TF.vflip(target_tensor)

        # Random rotation (0, 90, 180, 270)
        angle = float(torch.randint(0, 4, (1,)).item() * 90)
        if angle > 0:
            bf_tensor = TF.rotate(bf_tensor, angle)
            target_tensor = TF.rotate(target_tensor, angle)

        # Random brightness/contrast on BF only (simulates imaging variation)
        if torch.rand(1) > 0.5:
            factor = 0.8 + 0.4 * torch.rand(1).item()
            bf_tensor = TF.adjust_brightness(bf_tensor, factor)
        if torch.rand(1) > 0.5:
            factor = 0.8 + 0.4 * torch.rand(1).item()
            bf_tensor = TF.adjust_contrast(bf_tensor, factor)

        # Gaussian noise on BF
        if torch.rand(1) > 0.5:
            noise = torch.randn_like(bf_tensor) * 0.02
            bf_tensor = (bf_tensor + noise).clamp(0, 1)

        return bf_tensor, target_tensor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # Load BF image
        bf = load_image(pair["bf"])
        if bf.ndim == 2:
            bf = np.stack([bf] * 3, axis=-1)  # Grayscale -> 3-channel for pretrained encoders
        elif bf.shape[-1] == 1:
            bf = np.concatenate([bf] * 3, axis=-1)

        # Load target channels and stack
        targets = []
        for tgt_path in pair["targets"]:
            tgt = load_image(tgt_path)
            if tgt.ndim == 2:
                tgt = tgt[..., np.newaxis]
            targets.append(tgt)

        if self.target_channels == "both":
            # DAPI signal is in blue channel (ch2), Phalloidin in red channel (ch0)
            # targets[0] = DAPI image, targets[1] = Phalloidin image
            dapi = targets[0]
            phall = targets[1]
            if dapi.ndim == 3 and dapi.shape[-1] == 3:
                dapi = dapi[..., 2:3]   # Blue channel for DAPI
            elif dapi.ndim == 2:
                dapi = dapi[..., np.newaxis]
            else:
                dapi = dapi[..., :1]

            if phall.ndim == 3 and phall.shape[-1] == 3:
                phall = phall[..., 0:1]  # Red channel for Phalloidin (Texas Red)
            elif phall.ndim == 2:
                phall = phall[..., np.newaxis]
            else:
                phall = phall[..., :1]

            target = np.concatenate([dapi, phall], axis=-1)
        elif self.target_channels == "overlay":
            target = targets[0]  # Keep all 3 RGB channels
            if target.ndim == 2:
                target = np.stack([target] * 3, axis=-1)
        elif self.target_channels == "dapi":
            target = targets[0]
            if target.ndim == 3 and target.shape[-1] == 3:
                target = target[..., 2:3]   # Blue channel
            elif target.ndim == 2:
                target = target[..., np.newaxis]
            else:
                target = target[..., :1]
        elif self.target_channels == "phalloidin":
            target = targets[0]
            if target.ndim == 3 and target.shape[-1] == 3:
                target = target[..., 0:1]   # Red channel
            elif target.ndim == 2:
                target = target[..., np.newaxis]
            else:
                target = target[..., :1]
        else:
            target = targets[0]
            if target.ndim == 2:
                target = target[..., np.newaxis]
            elif target.shape[-1] > 1:
                target = target[..., :1]

        # Convert to tensors [C, H, W]
        bf_tensor = torch.from_numpy(bf).permute(2, 0, 1).float()
        target_tensor = torch.from_numpy(target).permute(2, 0, 1).float()

        # Resize
        bf_tensor = TF.resize(bf_tensor, [self.img_size, self.img_size],
                              interpolation=transforms.InterpolationMode.BILINEAR,
                              antialias=True)
        target_tensor = TF.resize(target_tensor, [self.img_size, self.img_size],
                                  interpolation=transforms.InterpolationMode.BILINEAR,
                                  antialias=True)

        # Augment
        if self.augment:
            bf_tensor, target_tensor = self._apply_augmentation(bf_tensor, target_tensor)

        return {
            "bf": bf_tensor,
            "target": target_tensor,
            "name": pair["base_name"],
        }


def get_dataloaders(config):
    """Create train/val/test dataloaders from config dict.

    IMPORTANT: Splits by embryo (project folder), NOT by individual image.
    All z-slices from the same embryo stay in the same split to prevent
    data leakage (adjacent z-slices are nearly identical).

    Config expected keys:
        data.root_dir: path to data folder
        data.target_channels: 'both', 'dapi', 'phalloidin', 'combined', 'overlay'
        data.img_size: int
        data.batch_size: int
        data.num_workers: int
        data.val_split: float (0-1)
        data.test_split: float (0-1)
        data.z_slices: list[int] or null
    """
    data_cfg = config["data"]

    full_dataset = EmbryoDataset(
        root_dir=data_cfg["root_dir"],
        target_channels=data_cfg.get("target_channels", "both"),
        img_size=data_cfg.get("img_size", 512),
        augment=False,
        z_slices=data_cfg.get("z_slices"),
    )

    # --- Split by embryo project, not by image ---
    # Group image indices by their project folder
    from collections import OrderedDict
    project_to_indices = OrderedDict()
    for idx, pair in enumerate(full_dataset.pairs):
        proj = pair["project"]
        if proj not in project_to_indices:
            project_to_indices[proj] = []
        project_to_indices[proj].append(idx)

    projects = list(project_to_indices.keys())
    n_projects = len(projects)

    # Shuffle projects deterministically
    rng = np.random.RandomState(data_cfg.get("seed", 42))
    perm = rng.permutation(n_projects)
    projects_shuffled = [projects[i] for i in perm]

    val_split = data_cfg.get("val_split", 0.15)
    test_split = data_cfg.get("test_split", 0.1)

    if n_projects >= 3:
        # Split by project count
        n_test = max(1, int(n_projects * test_split))
        n_val = max(1, int(n_projects * val_split))
        n_train = n_projects - n_val - n_test

        test_projects = projects_shuffled[:n_test]
        val_projects = projects_shuffled[n_test:n_test + n_val]
        train_projects = projects_shuffled[n_test + n_val:]
    elif n_projects == 2:
        # 2 embryos: one train, one val (no test)
        train_projects = [projects_shuffled[0]]
        val_projects = [projects_shuffled[1]]
        test_projects = []
    else:
        # Single embryo: fall back to random image split (z-slice leakage
        # is unavoidable with 1 embryo, but warn user)
        print("WARNING: Only 1 embryo found. Splitting by z-slice — "
              "adjacent slices may leak between splits. "
              "Add more embryos for proper validation.")
        n = len(full_dataset)
        val_size = int(n * val_split)
        test_size = int(n * test_split)
        train_size = n - val_size - test_size

        generator = torch.Generator().manual_seed(data_cfg.get("seed", 42))
        train_ds, val_ds, test_ds = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )

        train_ds.dataset.augment = True
        batch_size = data_cfg.get("batch_size", 4)
        num_workers = data_cfg.get("num_workers", 4)

        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True, drop_last=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True),
            DataLoader(test_ds, batch_size=1, shuffle=False,
                       num_workers=num_workers, pin_memory=True),
        )

    # Collect indices for each split
    train_indices = [i for p in train_projects for i in project_to_indices[p]]
    val_indices = [i for p in val_projects for i in project_to_indices[p]]
    test_indices = [i for p in test_projects for i in project_to_indices[p]]

    # Print split info
    print(f"Split by embryo ({n_projects} embryos):")
    print(f"  Train: {len(train_projects)} embryos, {len(train_indices)} images — {train_projects}")
    print(f"  Val:   {len(val_projects)} embryos, {len(val_indices)} images — {val_projects}")
    print(f"  Test:  {len(test_projects)} embryos, {len(test_indices)} images — {test_projects}")

    from torch.utils.data import Subset
    import copy

    # Create a shallow copy for training with augmentation enabled
    # (avoids re-scanning all project folders from Drive)
    train_dataset = copy.copy(full_dataset)
    train_dataset.augment = True

    train_ds = Subset(train_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)    # full_dataset has augment=False
    test_ds = Subset(full_dataset, test_indices)

    batch_size = data_cfg.get("batch_size", 4)
    num_workers = data_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
