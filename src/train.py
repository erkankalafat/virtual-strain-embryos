"""
Main training script for virtual staining models.

Usage:
    python -m src.train --config configs/transunet.yaml
    python -m src.train --config configs/swin_unet.yaml
    python -m src.train --config configs/pix2pixhd.yaml
    python -m src.train --config configs/diffusion.yaml
    python -m src.train --config configs/adain.yaml

Supports: TransUNet, SwinUNet (regression), Pix2PixHD (GAN), DDPM (diffusion), AdaIN (style transfer)
"""

import argparse
import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import numpy as np

from tqdm.auto import tqdm

from src.data import get_dataloaders
from src.models import build_model
from src.losses import build_loss, GANLoss, FeatureMatchingLoss
from src.utils import compute_metrics, MetricTracker


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def save_sample_images(pred, target, bf, epoch, output_dir):
    """Save a grid of BF | Predicted | Ground Truth for visual inspection."""
    try:
        from torchvision.utils import save_image
        os.makedirs(output_dir, exist_ok=True)

        # Take first sample from batch
        bf_img = bf[0].cpu()
        pred_img = pred[0].cpu().clamp(0, 1)
        tgt_img = target[0].cpu()

        # If pred/target are 1-2 channels, expand to 3 for visualization
        def to_3ch(x):
            if x.size(0) == 1:
                return x.repeat(3, 1, 1)
            elif x.size(0) == 2:
                # DAPI=blue, Phalloidin=red, green=zero
                return torch.stack([x[1], torch.zeros_like(x[0]), x[0]], dim=0)
            return x[:3]

        grid = torch.stack([
            bf_img[:3],
            to_3ch(pred_img),
            to_3ch(tgt_img),
        ])
        save_image(grid, os.path.join(output_dir, f"epoch_{epoch:03d}.png"), nrow=3)
    except Exception:
        pass  # Don't fail training over visualization


def show_inline_samples(pred, target, bf, epoch, n_samples=3):
    """Display BF | Predicted | Ground Truth inline in Colab during training."""
    try:
        import matplotlib.pyplot as plt
        from IPython.display import display

        def to_composite(t):
            if t.shape[0] == 2:
                rgb = np.zeros((*t.shape[1:], 3))
                rgb[:, :, 2] = t[0].numpy()
                rgb[:, :, 0] = t[1].numpy()
                return rgb
            elif t.shape[0] == 1:
                return np.stack([t[0].numpy()] * 3, axis=-1)
            return t.permute(1, 2, 0).numpy()

        n = min(n_samples, pred.size(0))
        fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        for i in range(n):
            axes[i, 0].imshow(bf[i, 0].cpu().numpy(), cmap='gray')
            axes[i, 0].set_title('BF Input')
            axes[i, 1].imshow(to_composite(target[i].cpu()))
            axes[i, 1].set_title('Ground Truth IF')
            axes[i, 2].imshow(to_composite(pred[i].cpu().clamp(0, 1)))
            axes[i, 2].set_title(f'Predicted (epoch {epoch})')
            for ax in axes[i]:
                ax.axis('off')

        plt.suptitle(f'Validation Samples — Epoch {epoch}', fontsize=13)
        plt.tight_layout()
        display(fig)
        plt.close(fig)
    except Exception:
        pass


class EarlyStopping:
    """Stop training when val loss stops improving.

    Args:
        patience: Epochs to wait after last improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class HistoryTracker:
    """Track and display training history across epochs."""

    def __init__(self):
        self.history = {}

    def log(self, epoch, **kwargs):
        for k, v in kwargs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)

    def plot(self):
        """Display training curves inline (works in Colab)."""
        try:
            import matplotlib.pyplot as plt
            from IPython.display import display, clear_output

            n_metrics = len(self.history)
            if n_metrics == 0:
                return

            clear_output(wait=True)
            fig, axes = plt.subplots(1, min(n_metrics, 4), figsize=(5 * min(n_metrics, 4), 4))
            if n_metrics == 1:
                axes = [axes]

            for ax, (name, values) in zip(axes, list(self.history.items())[:4]):
                ax.plot(range(1, len(values) + 1), values, linewidth=1.5)
                ax.set_title(name)
                ax.set_xlabel("Epoch")
                ax.grid(True, alpha=0.3)
                # Mark best
                if "loss" in name.lower() or "mae" in name.lower():
                    best_idx = np.argmin(values)
                else:
                    best_idx = np.argmax(values)
                ax.axvline(best_idx + 1, color="red", linestyle="--", alpha=0.5, label=f"Best: {values[best_idx]:.4f}")
                ax.legend(fontsize=8)

            plt.tight_layout()
            display(fig)
            plt.close(fig)
        except ImportError:
            pass  # Not in notebook environment

    def print_summary(self):
        """Print a compact summary table of the last epoch."""
        if not self.history:
            return
        last = {k: v[-1] for k, v in self.history.items()}
        parts = [f"{k}: {v:.4f}" for k, v in last.items()]
        print(" | ".join(parts))


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_regression(config):
    """Training loop for regression models (TransUNet, SwinUNet)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, _ = get_dataloaders(config)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # Model
    model = build_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config['model']['name']} | Params: {total_params:,}")

    # Loss
    criterion = build_loss(config)
    if isinstance(criterion, nn.Module):
        criterion = criterion.to(device)

    # Optimizer & scheduler
    opt_cfg = config.get("optimizer", {})
    lr = opt_cfg.get("lr", 2e-4)
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=opt_cfg.get("weight_decay", 1e-4))

    sched_cfg = config.get("scheduler", {})
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"],
        eta_min=sched_cfg.get("min_lr", 1e-6),
    )

    # Output
    output_dir = Path(config["training"].get("output_dir", "outputs")) / config["model"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs = config["training"]["epochs"]
    history = HistoryTracker()
    patience = config["training"].get("patience", 15)
    early_stop = EarlyStopping(patience=patience)

    epoch_pbar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        # --- Train ---
        model.train()
        train_metrics = MetricTracker()
        epoch_loss = 0.0

        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]",
                          leave=False, unit="batch")
        for batch in batch_pbar:
            bf = batch["bf"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            pred = model(bf)
            loss_val = criterion(pred, target)
            if isinstance(loss_val, tuple):
                loss, breakdown = loss_val
            else:
                loss, breakdown = loss_val, {"total": loss_val.item()}

            # Skip batch if loss is NaN (bad image / numerical issue)
            if torch.isnan(loss) or torch.isinf(loss):
                tqdm.write(f"  WARNING: NaN/Inf loss at batch, skipping")
                optimizer.zero_grad()
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            train_metrics.update(compute_metrics(pred, target))

            # Update batch progress bar
            batch_pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)

        # --- Validate ---
        model.eval()
        val_metrics = MetricTracker()
        val_loss = 0.0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]",
                        leave=False, unit="batch")
        with torch.no_grad():
            for batch in val_pbar:
                bf = batch["bf"].to(device)
                target = batch["target"].to(device)

                pred = model(bf)
                loss_val = criterion(pred, target)
                if isinstance(loss_val, tuple):
                    loss, _ = loss_val
                else:
                    loss = loss_val

                val_loss += loss.item()
                val_metrics.update(compute_metrics(pred, target))

                # Save sample on first batch
                if val_loss == loss.item():
                    save_sample_images(pred, target, bf, epoch, str(output_dir / "samples"))
                    # Show inline samples periodically
                    preview_every = config["training"].get("preview_every", 20)
                    if epoch % preview_every == 0 or epoch == 1:
                        show_inline_samples(pred, target, bf, epoch, n_samples=3)

        avg_val_loss = val_loss / max(len(val_loader), 1)

        # Log
        t_summary = train_metrics.summary()
        v_summary = val_metrics.summary()

        history.log(epoch,
                    train_loss=avg_train_loss,
                    val_loss=avg_val_loss,
                    val_psnr=v_summary.get("psnr", 0),
                    val_ssim=v_summary.get("ssim", 0))

        # Update epoch progress bar
        epoch_pbar.set_postfix(
            train_loss=f"{avg_train_loss:.4f}",
            val_loss=f"{avg_val_loss:.4f}",
            psnr=f"{v_summary.get('psnr', 0):.2f}",
            ssim=f"{v_summary.get('ssim', 0):.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

        # Checkpoint
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "val_metrics": v_summary,
                "config": config,
            }, str(output_dir / "best_model.pth"))
            tqdm.write(f"  Epoch {epoch}: Best model saved (val_loss: {avg_val_loss:.4f})")

        # Periodic checkpoint
        if epoch % config["training"].get("save_every", 10) == 0:
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config,
            }, str(output_dir / f"checkpoint_epoch_{epoch:03d}.pth"))

        # Plot curves every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            history.plot()

        # Early stopping
        if early_stop.step(avg_val_loss):
            tqdm.write(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            history.plot()
            break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Outputs saved to: {output_dir}")
    return history


def train_gan(config):
    """Training loop for Pix2PixHD (adversarial + L1 + feature matching)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, _ = get_dataloaders(config)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # Build model (contains both G and D)
    model = build_model(config)
    generator = model.get_generator().to(device)
    discriminator = model.get_discriminator().to(device)

    g_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"Generator params: {g_params:,} | Discriminator params: {d_params:,}")

    # Losses
    gan_loss = GANLoss(mode="lsgan").to(device)
    fm_loss = FeatureMatchingLoss()
    recon_loss = build_loss(config)
    if isinstance(recon_loss, nn.Module):
        recon_loss = recon_loss.to(device)

    lambda_gan = config["loss"].get("lambda_gan", 1.0)
    lambda_fm = config["loss"].get("lambda_fm", 10.0)
    lambda_recon = config["loss"].get("lambda_recon", 10.0)

    # Optimizers
    opt_cfg = config.get("optimizer", {})
    lr_g = opt_cfg.get("lr_g", 2e-4)
    lr_d = opt_cfg.get("lr_d", 2e-4)
    betas = (opt_cfg.get("beta1", 0.5), opt_cfg.get("beta2", 0.999))

    opt_G = optim.Adam(generator.parameters(), lr=lr_g, betas=betas)
    opt_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=betas)

    output_dir = Path(config["training"].get("output_dir", "outputs")) / "pix2pixhd"
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = config["training"]["epochs"]
    best_val_psnr = 0.0
    history = HistoryTracker()
    patience = config["training"].get("patience", 20)  # GANs need more patience
    early_stop = EarlyStopping(patience=patience)

    epoch_pbar = tqdm(range(1, epochs + 1), desc="Pix2PixHD Training", unit="epoch")
    for epoch in epoch_pbar:
        generator.train()
        discriminator.train()
        g_losses, d_losses = [], []

        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}",
                          leave=False, unit="batch")
        for batch in batch_pbar:
            bf = batch["bf"].to(device)
            target = batch["target"].to(device)

            fake = generator(bf)

            # --- Train Discriminator ---
            opt_D.zero_grad()
            real_input = torch.cat([bf, target], dim=1)
            fake_input = torch.cat([bf, fake.detach()], dim=1)

            pred_real = discriminator(real_input)
            pred_fake = discriminator(fake_input)

            d_loss = 0.0
            for pr, pf in zip(pred_real, pred_fake):
                d_loss += (gan_loss(pr[-1], True) + gan_loss(pf[-1], False)) * 0.5

            d_loss.backward()
            opt_D.step()
            d_losses.append(d_loss.item())

            # --- Train Generator ---
            opt_G.zero_grad()
            fake_input = torch.cat([bf, fake], dim=1)
            real_input = torch.cat([bf, target], dim=1)

            pred_fake = discriminator(fake_input)
            pred_real = discriminator(real_input)

            # GAN loss
            g_gan = 0.0
            for pf in pred_fake:
                g_gan += gan_loss(pf[-1], True)

            # Feature matching loss
            g_fm = 0.0
            for pr, pf in zip(pred_real, pred_fake):
                g_fm += fm_loss(pr[:-1], pf[:-1])

            # Reconstruction loss
            r_loss = recon_loss(fake, target)
            if isinstance(r_loss, tuple):
                r_loss = r_loss[0]

            g_loss = lambda_gan * g_gan + lambda_fm * g_fm + lambda_recon * r_loss
            g_loss.backward()
            opt_G.step()
            g_losses.append(g_loss.item())

            batch_pbar.set_postfix(G=f"{g_loss.item():.3f}", D=f"{d_loss.item():.3f}")

        # --- Validate ---
        generator.eval()
        val_metrics = MetricTracker()
        with torch.no_grad():
            for batch in val_loader:
                bf = batch["bf"].to(device)
                target = batch["target"].to(device)
                pred = generator(bf)
                val_metrics.update(compute_metrics(pred, target))

                if len(val_metrics.values["psnr"]) == 1:
                    save_sample_images(pred, target, bf, epoch, str(output_dir / "samples"))
                    preview_every = config["training"].get("preview_every", 20)
                    if epoch % preview_every == 0 or epoch == 1:
                        show_inline_samples(pred, target, bf, epoch, n_samples=3)

        v = val_metrics.summary()

        history.log(epoch,
                    g_loss=np.mean(g_losses),
                    d_loss=np.mean(d_losses),
                    val_psnr=v.get("psnr", 0),
                    val_ssim=v.get("ssim", 0))

        epoch_pbar.set_postfix(
            G=f"{np.mean(g_losses):.4f}",
            D=f"{np.mean(d_losses):.4f}",
            psnr=f"{v.get('psnr', 0):.2f}",
        )

        if v.get("psnr", 0) > best_val_psnr:
            best_val_psnr = v["psnr"]
            save_checkpoint({
                "epoch": epoch,
                "generator_state": generator.state_dict(),
                "discriminator_state": discriminator.state_dict(),
                "val_metrics": v,
                "config": config,
            }, str(output_dir / "best_model.pth"))
            tqdm.write(f"  Epoch {epoch}: Best model saved (PSNR: {best_val_psnr:.2f})")

        if epoch % 5 == 0 or epoch == epochs:
            history.plot()

        # Early stopping (monitor -PSNR so higher PSNR = lower "loss")
        if early_stop.step(-v.get("psnr", 0)):
            tqdm.write(f"\n  Early stopping at epoch {epoch} (no PSNR improvement for {patience} epochs)")
            history.plot()
            break

    print(f"\nTraining complete. Best val PSNR: {best_val_psnr:.2f}")
    return history


def train_diffusion(config):
    """Training loop for Conditional DDPM."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, _ = get_dataloaders(config)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    model = build_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Conditional DDPM | Params: {total_params:,}")

    opt_cfg = config.get("optimizer", {})
    optimizer = optim.AdamW(model.parameters(), lr=opt_cfg.get("lr", 2e-4),
                            weight_decay=opt_cfg.get("weight_decay", 1e-4))

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"],
        eta_min=opt_cfg.get("min_lr", 1e-6),
    )

    output_dir = Path(config["training"].get("output_dir", "outputs")) / "diffusion"
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = config["training"]["epochs"]
    best_val_loss = float("inf")
    history = HistoryTracker()
    patience = config["training"].get("patience", 25)  # Diffusion converges slowly
    early_stop = EarlyStopping(patience=patience)

    epoch_pbar = tqdm(range(1, epochs + 1), desc="DDPM Training", unit="epoch")
    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0.0

        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}",
                          leave=False, unit="batch")
        for batch in batch_pbar:
            bf = batch["bf"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            loss = model.training_loss(target, bf)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # Validate: compute noise prediction loss + generate samples
        model.eval()
        val_loss = 0.0
        val_metrics = MetricTracker()

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                bf = batch["bf"].to(device)
                target = batch["target"].to(device)

                loss = model.training_loss(target, bf)
                val_loss += loss.item()

                # Generate a few samples for metrics (expensive, so limit)
                if i == 0:
                    sampled = model.sample(bf[:2], method="ddim", num_steps=50)
                    val_metrics.update(compute_metrics(sampled, target[:2]))
                    save_sample_images(sampled, target[:2], bf[:2], epoch,
                                       str(output_dir / "samples"))
                    preview_every = config["training"].get("preview_every", 20)
                    if epoch % preview_every == 0 or epoch == 1:
                        show_inline_samples(sampled, target[:2], bf[:2], epoch, n_samples=2)

        avg_val_loss = val_loss / max(len(val_loader), 1)
        v = val_metrics.summary()

        history.log(epoch,
                    train_loss=avg_loss,
                    val_loss=avg_val_loss,
                    sample_psnr=v.get("psnr", 0))

        epoch_pbar.set_postfix(
            train=f"{avg_loss:.4f}",
            val=f"{avg_val_loss:.4f}",
            psnr=f"{v.get('psnr', 0):.2f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": avg_val_loss,
                "config": config,
            }, str(output_dir / "best_model.pth"))
            tqdm.write(f"  Epoch {epoch}: Best model saved")

        if epoch % 5 == 0 or epoch == epochs:
            history.plot()

        if early_stop.step(avg_val_loss):
            tqdm.write(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            history.plot()
            break

    print(f"\nDiffusion training complete. Best val loss: {best_val_loss:.4f}")
    return history


def train_style_transfer(config):
    """Training loop for AdaIN/EFDM style transfer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, _ = get_dataloaders(config)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    model = build_model(config).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"AdaIN Style Transfer | Trainable params: {trainable:,}")

    opt_cfg = config.get("optimizer", {})
    # Only train the decoder
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt_cfg.get("lr", 1e-4),
        weight_decay=opt_cfg.get("weight_decay", 0),
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=opt_cfg.get("step_size", 20), gamma=0.5
    )

    lambda_content = config["loss"].get("lambda_content", 1.0)
    lambda_style = config["loss"].get("lambda_style", 10.0)

    output_dir = Path(config["training"].get("output_dir", "outputs")) / "adain"
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = config["training"]["epochs"]
    best_val_loss = float("inf")
    history = HistoryTracker()
    patience = config["training"].get("patience", 15)
    early_stop = EarlyStopping(patience=patience)

    epoch_pbar = tqdm(range(1, epochs + 1), desc="AdaIN Training", unit="epoch")
    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0.0

        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}",
                          leave=False, unit="batch")
        for batch in batch_pbar:
            bf = batch["bf"].to(device)       # Content (BF)
            target = batch["target"].to(device)  # Style reference (IF)

            optimizer.zero_grad()
            output, content_loss, style_loss = model(bf, target)
            loss = lambda_content * content_loss + lambda_style * style_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        val_metrics = MetricTracker()

        with torch.no_grad():
            for batch in val_loader:
                bf = batch["bf"].to(device)
                target = batch["target"].to(device)

                output, c_loss, s_loss = model(bf, target)
                loss = lambda_content * c_loss + lambda_style * s_loss
                val_loss += loss.item()
                val_metrics.update(compute_metrics(output, target))

                if len(val_metrics.values["psnr"]) == 1:
                    save_sample_images(output, target, bf, epoch, str(output_dir / "samples"))
                    preview_every = config["training"].get("preview_every", 20)
                    if epoch % preview_every == 0 or epoch == 1:
                        show_inline_samples(output, target, bf, epoch, n_samples=3)

        avg_val_loss = val_loss / max(len(val_loader), 1)
        v = val_metrics.summary()

        history.log(epoch,
                    train_loss=avg_loss,
                    val_loss=avg_val_loss,
                    val_psnr=v.get("psnr", 0),
                    val_ssim=v.get("ssim", 0))

        epoch_pbar.set_postfix(
            train=f"{avg_loss:.4f}",
            val=f"{avg_val_loss:.4f}",
            psnr=f"{v.get('psnr', 0):.2f}",
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": avg_val_loss,
                "config": config,
            }, str(output_dir / "best_model.pth"))
            tqdm.write(f"  Epoch {epoch}: Best model saved")

        if epoch % 5 == 0 or epoch == epochs:
            history.plot()

        if early_stop.step(avg_val_loss):
            tqdm.write(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            history.plot()
            break

    print(f"\nStyle transfer training complete. Best val loss: {best_val_loss:.4f}")
    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TRAIN_FN_MAP = {
    "transunet": train_regression,
    "swin_unet": train_regression,
    "pix2pixhd": train_gan,
    "diffusion": train_diffusion,
    "adain": train_style_transfer,
}


def main():
    parser = argparse.ArgumentParser(description="Train virtual staining models")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = config["model"]["name"]

    print(f"\n{'='*60}")
    print(f"Virtual Staining Training: {model_name}")
    print(f"{'='*60}\n")

    # Save config
    output_dir = Path(config["training"].get("output_dir", "outputs")) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    train_fn = TRAIN_FN_MAP.get(model_name)
    if train_fn is None:
        raise ValueError(f"No training loop for model: {model_name}. "
                         f"Available: {list(TRAIN_FN_MAP.keys())}")

    train_fn(config)


if __name__ == "__main__":
    main()
