"""
train.py
========
Training logic for GN2.

Provides:
    - GN2Loss      : combined multi-task loss
    - lr_scheduler : cosine annealing + linear warmup
    - run_epoch    : single epoch train/val loop
    - train        : full training loop, callable from main.py

Outputs:
    Directory specified in ``output_dir``:

    .. code-block:: text

        outputs/checkpoints/
        ├── runs/
        │    └── <timestamp>/
        │        ├── best_model.pt
        │        ├── events.out.tfevents.xxxx
        │        └── learning_curves.pdf
        └── best_model/
            ├── best_model.pt
            └── learning_curves.pdf
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .model import GN2
from .plotting import plot_learning_curves
from .utils import get_optimizer

logger = logging.getLogger("GN2.train     ")


class GN2Loss(nn.Module):
    """
    Jet classification loss only.

    .. math::
        L = CE_{jet}
    """

    def __init__(self):
        """
        Initialize thr GN2 loss.
        """
        super().__init__()
        self.ce_jet = nn.CrossEntropyLoss(ignore_index = -1)

    def forward(self, outputs: dict, labels: dict) -> dict:
        """
        Forward pass to compute the loss.

        Args:
            outputs (dict): model outputs with keys ``"jet_outputs"``
                (torch.Tensor, shape ``(batch_size, n_classes)``) with raw outputs for jet
                classification.
            labels (dict): ground truth labels with keys ``"jet_label"``
                (torch.Tensor, shape ``(batch_size,)``) with integer class labels for each jet.
        """
        loss_jet = self.ce_jet(outputs["jet_outputs"], labels["jet_label"])

        return {
            "total": loss_jet,
            "jet": loss_jet,
        }


def lr_scheduler(
    optimizer,
    n_total_steps: int,
    warmup_frac: float = 0.01,
    lr_initial: float = 1.0e-07,
    lr_peak: float = 5.0e-04,
    lr_final: float = 1.0e-05,
) -> LambdaLR:
    """
    Build a learning rate scheduler with linear warmup followed by cosine annealing.

    The schedule consists of two phases:
        1. Linear warmup: the learning rate increases from ``lr_initial`` to ``lr_peak``
        over the first ``warmup_frac * n_total_steps`` steps.
        2. Cosine decay: the learning rate decreases from ``lr_peak`` to ``lr_final``
        following a cosine schedule for the remaining steps.

    Args:
        optimizer: optimizer instance (e.g. ``AdamW``).
        n_total_steps (int): total number of optimizer steps (``epochs x batches``).
        warmup_frac (float): fraction of steps used for warmup (default: ``0.01``).
        lr_initial (float): initial learning rate at step 0.
        lr_peak (float): peak learning rate reached after warmup.
        lr_final (float): minimum learning rate at the end of cosine decay.

    Returns:
        torch.optim.lr_scheduler.SequentialLR: Scheduler implementing
            warmup + cosine annealing.
    """
    n_warmup = max(1, int(warmup_frac * n_total_steps))

    base_lr = optimizer.param_groups[0]["lr"]

    start_factor = lr_initial / base_lr
    end_factor   = lr_peak / base_lr

    warmup = LinearLR(
        optimizer,
        start_factor = start_factor,
        end_factor   = end_factor,
        total_iters  = n_warmup,
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max   = n_total_steps - n_warmup,
        eta_min = lr_final,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers = [warmup, cosine],
        milestones = [n_warmup],
    )

    return scheduler


def run_epoch(
    model: GN2,
    loader: DataLoader,
    loss: GN2Loss,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    is_train: bool,
    scaler: torch.amp.GradScaler = None,
) -> dict:
    """
    Run one full epoch.

    Args:
        model (GN2): GN2 model instance.
        loader (DataLoader): train or val DataLoader.
        criterion (GN2Loss): GN2Loss instance.
        optimizer (torch.optim.Optimizer): optimizer instance (e.g. ``AdamW``).
        scheduler (LambdaLR): LambdaLR instance (stepped only during training).
        device (torch.device): Device to move tensors to.
        is_train (bool): ``True`` for training, ``False`` for validation.
        scaler (torch.amp.GradScaler): optional ``GradScaler`` for mixed precision training.
            (default: ``None``)

    Returns:
        dict with averaged loss.
    """
    if is_train:
        model.train()
    else:
        model.eval()

    totals    = {"total": 0.0, "jet": 0.0}
    n_batches = 0
    ctx       = torch.enable_grad if is_train else torch.no_grad
    model = model.to(device)

    with ctx():
        for batch in loader:
            jet_vars   = batch["jet_features"].to(device)
            track_vars = batch["track_features"].to(device)
            mask       = batch["mask"].to(device)

            labels = {"jet_label": batch["label"].to(device)}

            if is_train and scaler is not None:
                # torch.amp.autocast automatically chooses the precision based on
                # the device and runs the forward pass in that precision for better performance.
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(jet_vars, track_vars, mask)
                    losses  = loss(outputs, labels)
                optimizer.zero_grad()
                # underflow can cause loss to become zero in float16 scale it up before backward().
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # after unscaling, gradients may be inf or NaN -> skip the optimizer step.
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            elif is_train:
                outputs = model(jet_vars, track_vars, mask)
                losses  = loss(outputs, labels)
                optimizer.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
            else:
                outputs = model(jet_vars, track_vars, mask)
                losses  = loss(outputs, labels)

            for k in totals:
                totals[k] += losses[k].item()
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@dataclass      # decorator to automatically generate init, repr, etc.
class TrainingHistory:
    """
    Class for storing training history (losses and learning rates) during training.
    Useful for plotting learning curves after training.

    Attributes:
        train_loss (list[float]): list of training losses per epoch.
        val_loss (list[float]): list of validation losses per epoch.
        lr (list[float]): list of learning rates per epoch.
    """
    # field(default_factory=...) is used to create a new list for each instance
    train_loss: list[float] = field(default_factory=list)
    val_loss:   list[float] = field(default_factory=list)
    lr:         list[float] = field(default_factory=list)

    def append(
        self,
        train_loss: float,
        val_loss: float,
        lr: float,
    ) -> None:
        """
        Append a new entry to the training history.

        Args:
            train_loss (float): training loss for the current epoch.
            val_loss (float): validation loss for the current epoch.
            lr (float): learning rate for the current epoch.
        """
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.lr.append(lr)

    def to_dict(self) -> dict[str, list[float]]:
        """
        Convert the training history to a dictionary format suitable for plotting.

        Returns:
            dict[str, list[float]]: dictionary with keys ``"train_loss"``, ``"val_loss"``
                and ``"lr"``, each containing a list of values per epoch.
        """
        return {
            "train_loss": self.train_loss,
            "val_loss":   self.val_loss,
            "lr":         self.lr,
        }


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------
def train(
    model: GN2,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str | Path,
    device: torch.device,
    optimizer: str = "adam",
    max_epochs: int = 21,
    warmup_frac: float = 0.01,
    weight_decay: float = 1.0e-05,
    lr_initial: float = 1.0e-07,
    lr_peak: float = 5.0e-04,
    lr_final: float = 1.0e-05,
    config: dict | None = None,
) -> GN2:
    """
    Run the full training loop and return the best model.

    Saves two checkpoints:
        - ``runs/<timestamp>/best_model.pt`` - best model for this run.
        - ``best_model/best_model.pt``       - global best across all runs.

    After training, generates learning-curve.

    Args:
        model (GN2): GN2 instance already moved to *device*.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        output_dir (str | Path): root directory for checkpoints.
        device (torch device): device to run training on.
        optimizer (str): name of the optimizer to use (default: ``"adam"``).
        max_epochs (int): maximum number of epochs to train (default: ``21``).
        warmup_frac (float): fraction of total steps to use for linear warmup (default: ``0.01``).
        weight_decay (float): weight decay for the optimizer (default: ``1.0e-05``).
        lr_initial (float): initial learning rate at step 0 (default: ``1.0e-07``).
        lr_peak (float): peak learning rate reached after warmup (default: ``5.0e-04``).
        lr_final (float): final learning rate (default: ``1.0e-05``).
        config (dict): full configuration dictionary (optional, used for logging and plotting).

    Returns:
        GN2 model loaded with the best checkpoint weights from this run.

    Raises:
        OSError: if writing a checkpoint to disk fails.
    """
    output_dir = Path(output_dir)
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runs_dir = output_dir / "runs" / run_name
    runs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = runs_dir / "best_model.pt"
    best_dir = output_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_best_path = best_dir / "best_model.pt"

    if checkpoint_best_path.exists():
        best_model = torch.load(checkpoint_best_path, map_location="cpu")
        best_model_val_loss = best_model["val_loss"]
    else:
        best_model_val_loss = float("inf")

    loss = GN2Loss()

    optimizer = get_optimizer(model, optimizer, lr_peak, weight_decay)

    n_total_steps = max_epochs * len(train_loader)
    lr_decay = lr_scheduler(
        optimizer,
        n_total_steps,
        warmup_frac = warmup_frac,
        lr_initial  = lr_initial,
        lr_peak     = lr_peak,
        lr_final    = lr_final,
    )

    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    best_val_loss = float("inf")

    history = TrainingHistory()
    logger.info("Starting training ...")
    for epoch in range(1, max_epochs + 1):

        train_losses = run_epoch(model, train_loader, loss, optimizer,
                                 lr_decay, device, is_train=True,  scaler=scaler)
        val_losses   = run_epoch(model, val_loader,   loss, optimizer,
                                 lr_decay, device, is_train=False)

        lr_now = lr_decay.get_last_lr()[0]

        history.append(
            train_loss = train_losses["total"],
            val_loss   = val_losses["total"],
            lr         = lr_now,
        )

        logger.info("Epoch %s/%s | train loss=%s (jet=%s) | val=%s | lr=%s",
                    f"{epoch:4d}", max_epochs, f"{train_losses['total']:.4f}",
                    f"{train_losses['jet']:.4f}", f"{val_losses['total']:.4f}", f"{lr_now:.2e}")

        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            checkpoint = {
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss"   : best_val_loss,
                "config"     : config,
            }
            try:
                torch.save(checkpoint, checkpoint_path)
                logger.info("    New best (run) val_loss=%s - saved to %s",
                            f"{best_val_loss:.4f}", checkpoint_path)
            except OSError as e:
                logger.error("Failed to save checkpoint to %s: %s", checkpoint_path, e)

            if best_val_loss < best_model_val_loss:
                best_model_val_loss = best_val_loss
                try:
                    torch.save(checkpoint, checkpoint_best_path)
                    logger.info("    New best (global) model - saved to %s",
                                checkpoint_best_path)
                except OSError as e:
                    logger.error("Failed to save checkpoint to %s: %s", checkpoint_best_path, e)

    logger.info("Training complete.")

    # reload best weights before returning
    model = GN2.from_checkpoint(checkpoint_path, device)

    # plots
    plot_learning_curves(history.to_dict(), output_dir=runs_dir)
    if best_val_loss == best_model_val_loss:
        plot_learning_curves(history.to_dict(), output_dir=best_dir)

    return model


if __name__ == "__main__":
    pass
