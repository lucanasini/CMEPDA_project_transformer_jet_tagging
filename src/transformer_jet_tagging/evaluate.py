"""
evaluate.py
===========
Evaluation script for the GN2 jet flavour tagging pipeline.

Computes and saves:
    - Per-class softmax score distributions (one figure per output node)
    - D_b and D_c discriminant distributions
    - ROC curves (D_b and D_c)
    - Classification metrics (accuracy, precision, recall, F1)
    - Normalised confusion matrix

Outputs (under ``config["output"]["eval_dir"]``, default ``outputs/eval``):

    .. code-block:: text

        outputs/eval/
        ├── metrics.json
        ├── confusion_matrix.pdf
        ├── score_distributions.pdf      - softmax P(class) per true-label class
        ├── discriminant_db.pdf          - D_b distribution per flavour
        ├── discriminant_dc.pdf          - D_c distribution per flavour
        ├── roc_db.pdf                   - b-tag ROC
        └── roc_dc.pdf                   - c-tag ROC
"""

import json
import logging
from pathlib import Path

import matplotlib
import mplhep as hep
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

from ._constants import FLAVOUR_LABELS
from .model import GN2
from .plotting import (
    plot_all_roc,
    plot_confusion_matrix,
    plot_discriminant,
    plot_score_distributions,
)

matplotlib.use("Agg")
hep.style.use(hep.style.ATLAS)
logger = logging.getLogger("GN2.evaluate  ")


@torch.no_grad()
def run_inference(
    model: GN2,
    loader: DataLoader,
    device: torch.device,
    fc_b: float = 0.2,
    ftau_b: float = 0.05,
    fb_c: float = 0.3,
    ftau_c: float = 0.01,
    flavour_map: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Run inference on the full test loader and collect all outputs.

    Args:
        model (GN2): trained GN2 model in eval mode.
        loader (DataLoader): test-set DataLoader.
        device (torch.device): torch device.
        fc_b (float): c-fraction for D_b discriminant.
        ftau_b (float): tau-fraction for D_b discriminant.
        fb_c (float): b-fraction for D_c discriminant.
        ftau_c (float): tau-fraction for D_c discriminant.
        flavour_map (dict): ``{class_index: class_name}`` for the discriminants.

    Returns:
        dict with keys:
            ``"proba"`` (np.ndarray): shape ``(N, n_classes)``, softmax probabilities
            ``"preds"`` (np.ndarray): shape ``(N,)``, argmax predictions
            ``"labels"`` (np.ndarray): shape ``(N,)``, true class labels
            ``"db"`` (np.ndarray): shape ``(N,)``, D_b discriminant values
            ``"dc"`` (np.ndarray): shape ``(N,)``, D_c discriminant values
    """
    logger.info("Running inference on test set ...")
    all_prob, all_labels, all_db, all_dc = [], [], [], []

    model.eval()
    for batch in loader:
        jet_feats   = batch["jet_features"].to(device)
        track_feats = batch["track_features"].to(device)
        mask        = batch["mask"].to(device)

        prob = model.predict_proba(jet_feats, track_feats, mask)
        db   = model.discriminant_db(prob, fc=fc_b, ftau=ftau_b, flavour_map=flavour_map)
        dc   = model.discriminant_dc(prob, fb=fb_c, ftau=ftau_c, flavour_map=flavour_map)

        all_prob.append(prob.cpu().numpy())
        all_labels.append(batch["label"].numpy())
        all_db.append(db.cpu().numpy())
        all_dc.append(dc.cpu().numpy())

    proba  = np.concatenate(all_prob,   axis=0)
    labels = np.concatenate(all_labels, axis=0)
    db     = np.concatenate(all_db,     axis=0)
    dc     = np.concatenate(all_dc,     axis=0)
    preds  = proba.argmax(axis=1)

    # drop jets with label -1 (unmapped flavour)
    valid = labels != -1
    if not valid.all():
        logger.warning("Dropping %d jets with unmapped label (-1).", int((~valid).sum()))
    proba  = proba[valid]
    preds  = preds[valid]
    labels = labels[valid]
    db     = db[valid]
    dc     = dc[valid]

    return {
        "proba":  proba,
        "preds":  preds,
        "labels": labels,
        "db":     db,
        "dc":     dc,
    }


def save_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    output_dir: Path,
) -> dict:
    """
    Compute classification metrics and save them to ``metrics.json``.

    Args:
        labels (np.ndarray): true class labels.
        preds (np.ndarray): predicted class labels.
        output_dir (Path): output directory.

    Returns:
        dict: the computed metrics.
    """
    logger.info("Computing classification metrics ...")

    class_names = [FLAVOUR_LABELS[c] for c in sorted(FLAVOUR_LABELS.keys())]
    report      = classification_report(
        labels, preds,
        target_names  = class_names,
        output_dict   = True,
        zero_division = 0,
    )
    metrics = {
        "accuracy":  float(accuracy_score(labels, preds)),
        "per_class": report,
    }
    out = output_dir / "metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Saved: %s", out)
    logger.info("Accuracy: %s", f"{metrics["accuracy"]:.2%}")

    return metrics


def evaluate(
    test_loader: DataLoader,
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
    flavour_map: dict[str, int],
    fc: float = 0.2,
    ftau_b: float = 0.05,
    fb: float = 0.3,
    ftau_c: float = 0.01,
) -> dict:
    """
    Run the full evaluation pipeline on the test set.

    Args:
        test_loader (DataLoader): DataLoader for the test set.
        checkpoint_path (Path): path to the best model checkpoint (.pt).
        output_dir (Path): directory where all evaluation outputs are saved.
        device (torch.device): torch device.
        flavour_map (dict): ``{class_name: class_index}`` for the discriminants.
        fc (float): c-fraction for D_b discriminant.
        ftau_b (float): tau-fraction for D_b discriminant.
        fb (float): b-fraction for D_c discriminant.
        ftau_c (float): tau-fraction for D_c discriminant.

    Returns:
        dict: computed classification metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    flavour_map = {v: int(k) for k, v in flavour_map.items()}

    logger.info("Loading model from %s ...", checkpoint_path)
    model = GN2.from_checkpoint(str(checkpoint_path), device)
    model.eval()

    results = run_inference(
        model        = model,
        loader       = test_loader,
        device       = device,
        fc_b         = fc,
        ftau_b       = ftau_b,
        fb_c         = fb,
        ftau_c       = ftau_c,
        flavour_map  = flavour_map,
    )

    metrics = save_metrics(results["labels"], results["preds"], output_dir)

    plot_confusion_matrix(results["labels"], results["preds"], output_dir)

    plot_score_distributions(results["proba"], results["labels"], output_dir)

    plot_discriminant(
        discriminant_scores=results["db"],
        labels=results["labels"],
        discriminant_type="b",
        output_dir=output_dir,
    )
    plot_discriminant(
        discriminant_scores=results["dc"],
        labels=results["labels"],
        discriminant_type="c",
        output_dir=output_dir,
    )

    plot_all_roc(results, output_dir)

    logger.info("Evaluation complete. Outputs saved to '%s/'.", output_dir)

    return metrics


if __name__ == "__main__":
    pass
