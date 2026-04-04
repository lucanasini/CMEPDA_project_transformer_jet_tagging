"""
train.py
========
Training script for GN2.

Implements:
  - Combined loss: L = w_jet * CE_jet + w_origin * CE_origin + w_vertex * CE_vertex
  - AdamW optimiser with cosine annealing + linear warmup (as in the paper)
  - Class-weighted loss for track origin (inverse class frequencies)
  - Vertex loss computed only on real track pairs (mask applied)
  - Checkpoint saving (best val loss)
  - TensorBoard logging

Usage:
    python train.py --config configs/config.json
    python train.py --config configs/config.json --epochs 30 --lr 5e-4
"""

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import src.trasformer_jet_tagging.utils as utils
from src.trasformer_jet_tagging.dataset import GN2Dataset
from model import GN2

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("GN2.train")


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class GN2Loss(nn.Module):
    """
    Combined loss for the three GN2 tasks.

    L = w_jet    * CE(jet_logits, jet_labels)
      + w_origin * CE(origin_logits, origin_labels)   [valid tracks only]
      + w_vertex * CE(vertex_logits, vertex_labels)   [valid pairs only]

    Track-origin uses class-weighted CE to handle imbalance (paper §Methods).
    """

    def __init__(
        self,
        w_jet   : float = 1.0,
        w_origin: float = 0.5,
        w_vertex: float = 0.5,
        origin_class_weights: torch.Tensor = None,
        n_track_origin: int = 7,
    ):
        super().__init__()
        self.w_jet    = w_jet
        self.w_origin = w_origin
        self.w_vertex = w_vertex

        self.ce_jet    = nn.CrossEntropyLoss()
        self.ce_origin = nn.CrossEntropyLoss(
            weight=origin_class_weights,
            ignore_index=-1         # padded / missing labels → ignored
        )
        self.ce_vertex = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
        self,
        outputs: dict,
        labels : dict,
        mask   : torch.Tensor,      # (B, T) True = real track
    ) -> dict:
        """
        Args:
            outputs: dict from GN2.forward()
            labels : dict with keys:
                'jet_label'    : (B,)     long
                'origin_label' : (B, T)   long  (-1 = ignore)
                'vertex_label' : (B, T, T) long (-1 = ignore)
            mask   : (B, T) bool

        Returns:
            dict with 'total', 'jet', 'origin', 'vertex' losses.
        """
        # --- jet classification ---
        loss_jet = self.ce_jet(outputs["jet_logits"], labels["jet_label"])

        # --- track origin (only on real tracks; padded positions have label -1) ---
        B, T, C = outputs["origin_logits"].shape
        loss_origin = self.ce_origin(
            outputs["origin_logits"].reshape(B * T, C),
            labels["origin_label"].reshape(B * T),
        )

        # --- vertex (only on real track pairs) ---
        B, T, _, C2 = outputs["vertex_logits"].shape
        # pair mask: both tracks must be real
        pair_mask   = mask.unsqueeze(2) & mask.unsqueeze(1)   # (B, T, T)
        vtx_logits  = outputs["vertex_logits"].reshape(B * T * T, C2)
        vtx_labels  = labels["vertex_label"].reshape(B * T * T)
        # set label to -1 where pair is padded so CrossEntropyLoss ignores it
        vtx_labels  = vtx_labels.masked_fill(~pair_mask.reshape(B * T * T), -1)
        loss_vertex = self.ce_vertex(vtx_logits, vtx_labels)

        total = (self.w_jet    * loss_jet
               + self.w_origin * loss_origin
               + self.w_vertex * loss_vertex)

        return {
            "total"  : total,
            "jet"    : loss_jet,
            "origin" : loss_origin,
            "vertex" : loss_vertex,
        }


# ---------------------------------------------------------------------------
# LR scheduler: linear warmup + cosine annealing (paper §Methods)
# ---------------------------------------------------------------------------

def build_scheduler(optimiser, n_total_steps: int, warmup_frac: float = 0.01) -> LambdaLR:
    """
    Cosine annealing with linear warmup.
    - LR starts at lr_min = 1e-7 (approximated as 0)
    - Warms up linearly to peak LR over first 1% of steps
    - Cosine decays from peak to 1e-5 / peak over the remainder
    """
    n_warmup = max(1, int(warmup_frac * n_total_steps))

    def lr_lambda(step: int) -> float:
        if step < n_warmup:
            return step / n_warmup                           # linear warmup
        progress = (step - n_warmup) / max(1, n_total_steps - n_warmup)
        cosine   = 0.5 * (1 + math.cos(math.pi * progress))
        # scale so minimum is ~1e-5 / peak  (paper uses 1e-5 final lr)
        return max(cosine, 1e-5 / 5e-4)

    return LambdaLR(optimiser, lr_lambda)


# ---------------------------------------------------------------------------
# Helper: load preprocessing artifacts
# ---------------------------------------------------------------------------

def load_artifacts(preprocess_dir: Path):
    idx_dir   = preprocess_dir / "indices"
    norm_path = preprocess_dir / "norm_stats.json"

    for p in [idx_dir / "train_indices.npy",
              idx_dir / "val_indices.npy",
              norm_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Artifact not found: {p}\nRun preprocess.py first."
            )

    train_idx = np.load(idx_dir / "train_indices.npy")
    val_idx   = np.load(idx_dir / "val_indices.npy")

    with open(norm_path) as f:
        norm_stats = {k: np.array(v) for k, v in json.load(f).items()}

    logger.info(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}")
    return norm_stats, train_idx, val_idx


# ---------------------------------------------------------------------------
# Training & validation steps
# ---------------------------------------------------------------------------

def run_epoch(
    model      : GN2,
    loader     : DataLoader,
    criterion  : GN2Loss,
    optimiser  : AdamW,
    scheduler  : LambdaLR,
    device     : torch.device,
    is_train   : bool,
    scaler     : torch.cuda.amp.GradScaler = None,
) -> dict:
    """Run one full epoch. Returns averaged loss dict."""
    model.train() if is_train else model.eval()

    totals = {"total": 0.0, "jet": 0.0, "origin": 0.0, "vertex": 0.0}
    n_batches = 0

    ctx = torch.enable_grad if is_train else torch.no_grad

    with ctx():
        for batch in loader:
            jet_f  = batch["jet_features"].to(device)
            trk_f  = batch["track_features"].to(device)
            mask   = batch["mask"].to(device)

            labels = {
                "jet_label"    : batch["label"].to(device),
                # If auxiliary labels are not in the batch, use -1 (ignored by loss)
                "origin_label" : batch.get(
                    "origin_label",
                    torch.full((jet_f.size(0), trk_f.size(1)), -1, dtype=torch.long)
                ).to(device),
                "vertex_label" : batch.get(
                    "vertex_label",
                    torch.full(
                        (jet_f.size(0), trk_f.size(1), trk_f.size(1)), -1, dtype=torch.long
                    )
                ).to(device),
            }

            if is_train and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(jet_f, trk_f, mask)
                    losses  = criterion(outputs, labels, mask)
                optimiser.zero_grad()
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimiser)
                scaler.update()
                scheduler.step()
            elif is_train:
                outputs = model(jet_f, trk_f, mask)
                losses  = criterion(outputs, labels, mask)
                optimiser.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()
                scheduler.step()
            else:
                outputs = model(jet_f, trk_f, mask)
                losses  = criterion(outputs, labels, mask)

            for k in totals:
                totals[k] += losses[k].item()
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str, overrides: dict) -> None:

    # 1. Config
    config = utils.load_config_json(config_path)
    for key, value in overrides.items():
        section, _, param = key.partition(".")
        if param:
            config[section][param] = value

    preprocess_dir = Path(config["output"]["preprocess_dir"])
    output_dir     = Path(config["output"].get("checkpoints_dir", "outputs/checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load artifacts  (run preprocess if missing)
    if not (preprocess_dir / "norm_stats.json").exists():
        logger.info("Preprocessing artifacts not found — running preprocess.py …")
        import src.trasformer_jet_tagging.preprocess as preprocess
        preprocess.main(config_path)

    norm_stats, train_idx, val_idx = load_artifacts(preprocess_dir)

    # 3. DataLoaders
    file_path   = config["data"]["h5_path"]
    max_tracks  = config["data"].get("max_tracks", 40)
    jet_vars    = config["data"]["jet_features"]
    track_vars  = config["data"]["track_features"]
    label_var   = config["data"]["label"]
    label_map   = config["data"]["label_map"]
    batch_size  = config["training"].get("batch_size", 1024)
    num_workers = config["training"].get("num_workers", 4)
    shuffle     = config["data"].get("shuffle", True)

    common = dict(
        file_path=file_path, n_tracks=max_tracks,
        jet_vars=jet_vars, track_vars=track_vars,
        jet_flavour=label_var, jet_flavour_map=label_map,
        norm_stats=norm_stats,
    )
    loader_kw = dict(
        batch_size=batch_size, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(GN2Dataset(indices=train_idx, **common), shuffle=shuffle,  **loader_kw)
    val_loader   = DataLoader(GN2Dataset(indices=val_idx,   **common), shuffle=False, **loader_kw)

    # 4. Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model_cfg = config.get("model", {})
    model = GN2(
        n_jet_vars    = len(jet_vars),
        n_track_vars  = len(track_vars),
        n_classes     = model_cfg.get("n_classes", 4),
        n_track_origin= model_cfg.get("n_track_origin", 7),
        embed_dim     = model_cfg.get("embed_dim", 256),
        n_heads       = model_cfg.get("n_heads", 8),
        n_layers      = model_cfg.get("n_layers", 4),
        ff_dim        = model_cfg.get("ff_dim", 512),
        pool_dim      = model_cfg.get("pool_dim", 128),
        dropout       = model_cfg.get("dropout", 0.0),
    ).to(device)

    # 5. Loss, optimiser, scheduler
    criterion = GN2Loss(
        w_jet    = config["training"].get("w_jet",    1.0),
        w_origin = config["training"].get("w_origin", 0.5),
        w_vertex = config["training"].get("w_vertex", 0.5),
    )

    lr      = config["training"].get("lr", 5e-4)
    wd      = config["training"].get("weight_decay", 1e-5)
    optimiser = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    n_epochs      = config["training"].get("epochs", 20)
    n_total_steps = n_epochs * len(train_loader)
    scheduler     = build_scheduler(optimiser, n_total_steps,
                                    warmup_frac=config["training"].get("warmup_frac", 0.01))

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # 6. TensorBoard
    writer = SummaryWriter(log_dir=str(output_dir / "runs"))

    # 7. Training loop
    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        train_losses = run_epoch(model, train_loader, criterion, optimiser,
                                 scheduler, device, is_train=True, scaler=scaler)
        val_losses   = run_epoch(model, val_loader,   criterion, optimiser,
                                 scheduler, device, is_train=False)

        lr_now = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch:3d}/{n_epochs} | "
            f"train_loss={train_losses['total']:.4f}  "
            f"(jet={train_losses['jet']:.4f} "
            f"orig={train_losses['origin']:.4f} "
            f"vtx={train_losses['vertex']:.4f})  |  "
            f"val_loss={val_losses['total']:.4f}  |  "
            f"lr={lr_now:.2e}"
        )

        # TensorBoard
        for k, v in train_losses.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_losses.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        writer.add_scalar("lr", lr_now, epoch)

        # Checkpoint (best val loss)
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            ckpt_path = output_dir / "best_model.pt"
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optim_state": optimiser.state_dict(),
                "val_loss"   : best_val_loss,
                "config"     : config,
            }, ckpt_path)
            logger.info(f"  ↳ New best val_loss={best_val_loss:.4f} — saved to {ckpt_path}")

    writer.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GN2 training")
    parser.add_argument("--config", type=str,
                        default="src/trasformer_jet_tagging/configs/config.json")
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--batch-size",  type=int,   default=None)
    parser.add_argument("--num-workers", type=int,   default=None)
    args = parser.parse_args()

    overrides = {}
    if args.epochs      is not None: overrides["training.epochs"]      = args.epochs
    if args.lr          is not None: overrides["training.lr"]           = args.lr
    if args.batch_size  is not None: overrides["training.batch_size"]   = args.batch_size
    if args.num_workers is not None: overrides["training.num_workers"]  = args.num_workers

    main(args.config, overrides)