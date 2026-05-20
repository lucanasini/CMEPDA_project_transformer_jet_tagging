"""
utils.py
========
Utility functions for the transformer jet tagging project.
"""

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import SGD, Adam, AdamW

from ._constants import (
    JET_VARS_DEFAULT,
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_DEVICES,
    SUPPORTED_OPTIMIZERS,
    TRACK_VARS_DEFAULT,
)

logger = logging.getLogger("GN2.utils     ")


def load_config_json(filepath: str | Path) -> dict:
    """
    Load and return a JSON configuration file.

    Args:
        filepath (str | Path): path to the JSON configuration file.

    Returns:
        dict: parsed configuration.

    Raises:
        FileNotFoundError: if the file does not exist.
        json.JSONDecodeError: if the file is not valid JSON.
    """
    filepath = Path(filepath)
    try:
        with open(filepath, encoding="utf-8") as f:
            config = json.load(f)
        logger.info("Config loaded: %s", filepath)
        return config
    except FileNotFoundError:
        logger.error("Config file not found: %s", filepath)
        raise
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in config file %s: %s", filepath, exc)
        raise


def parse_label_map(raw: dict) -> dict[int, int]:
    """
    Convert a ``label_map`` whose keys are JSON strings to int keys.

    Args:
        raw: mapping as loaded from JSON, e.g. ``{"5": 0, "4": 1, ...}``.

    Returns:
        dict[int, int]: e.g. ``{5: 0, 4: 1, ...}``.

    Raises:
        ValueError: if any key cannot be converted to int.
    """
    try:
        return {int(k): int(v) for k, v in raw.items()}
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"label_map keys and values must be integers, got: {raw!r}"
        ) from e


def get_device(device_str: str = "auto") -> torch.device:
    """
    Resolve a device string from config into a ``torch.device``.

    Accepted values:
        - ``"auto"``  - CUDA if available, otherwise CPU.
        - ``"cuda"``  - NVIDIA GPU.
        - ``"cpu"``   - always CPU.

    Args:
        device_str (str): device string from config (default ``"auto"``).

    Returns:
        torch.device: resolved device.

    Raises:
        ValueError: if ``device_str`` is not one of the supported values.
        RuntimeError: if the requested device is not available on this machine.
    """
    key = device_str.lower()
    if key not in SUPPORTED_DEVICES:
        raise ValueError(
            f"Unknown device '{device_str}'. Supported: {SUPPORTED_DEVICES}"
        )

    if key == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    elif key == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' requested but CUDA is not available on this machine."
            )
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.debug("Using device: %s", device)
    return device


def get_activation(name: str) -> torch.nn:
    """
    Build and return an activation from the activation name.

    Supported activation names: ``"relu"``, ``"leakyrelu"``,
    ``"sigmoid"``, ``"tanh"``, ``"softplus"``.

    Args:
        name (str): activation name from config.

    Returns:
        torch.nn: configured activation instance.

    Raises:
        ValueError: if ``activation`` is not one of the supported names.
    """
    name = name.lower()
    if name not in SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Supported: {SUPPORTED_ACTIVATIONS}"
        )

    if name == "relu":
        activation = torch.nn.ReLU()
    elif name == "leakyrelu":
        activation = torch.nn.LeakyReLU()
    elif name == "sigmoid":
        activation = torch.nn.Sigmoid()
    elif name == "tanh":
        activation = torch.nn.Tanh()
    elif name == "softplus":
        activation = torch.nn.Softplus()

    logger.debug("Activation: %s", name.upper())
    return activation


def get_optimizer(
    model: nn.Module,
    name: str = "adam",
    lr: float = 5e-4,
    wd: float = 1e-5,
) -> torch.optim.Optimizer:
    """
    Build and return an optimizer from the config.

    Supported optimizer names: ``"adamw"``, ``"adam"``, ``"sgd"``.

    Config keys read:
        - `optimizer` (str,   default ``"adamw"``)
        - `lr_peak` (float, default ``5e-4``, used as the initial lr)
        - `weight_decay` (float, default ``1e-5``)

    Args:
        model (nn.Module): the model whose parameters will be optimized.
        name (str): the name of the optimizer to use.
        lr (float): learning rate (default ``5e-4``).
        wd (float): weight decay (default ``1e-5``).

    Returns:
        torch.optim.Optimizer: configured optimizer instance.

    Raises:
        ValueError: if ``optimizer`` is not one of the supported names.
    """
    name = name.lower()
    if name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{name}'. Supported: {SUPPORTED_OPTIMIZERS}"
        )

    params = model.parameters()

    if name == "adamw":
        optimizer = AdamW(params, lr=lr, weight_decay=wd)
    elif name == "adam":
        optimizer = Adam(params, lr=lr, weight_decay=wd)
    elif name == "sgd":
        optimizer = SGD(params, lr=lr, weight_decay=wd)

    logger.debug(
        "Optimizer: %s | lr_peak=%.2e | weight_decay=%.2e",
        name.upper(), lr, wd,
    )
    return optimizer


def check_artifacts(paths: list[str | Path]) -> bool:
    """
    Return ``True`` if every path in *paths* exists, ``False`` otherwise.

    Args:
        paths: list of ``Path`` objects to check.

    Returns:
        bool: ``True`` if all paths exist.
    """
    missing = [Path(p) for p in paths if not p.exists()]
    if missing:
        for p in missing:
            logger.warning("Missing artifact: %s", p)
        return False
    return True


def artifact_paths(preprocess_dir: str | Path) -> dict[str, Path]:
    """
    Return the paths for all preprocessing artifacts.

    Args:
        preprocess_dir (str | Path): root preprocessing output directory
            (``config["output"]["preprocess_dir"]``).

    Returns:
        dict with keys ``"train_indices"``, ``"val_indices"``,
        ``"test_indices"``, ``"norm_stats"``.
    """
    idx_dir = Path(preprocess_dir) / "indices"
    return {
        "train_indices": idx_dir / "train_indices.npy",
        "val_indices":   idx_dir / "val_indices.npy",
        "test_indices":  idx_dir / "test_indices.npy",
        "norm_stats":    preprocess_dir / "norm_stats.json",
    }


def load_indices(preprocess_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the train / val / test index arrays saved by the preprocessing step.

    Args:
        preprocess_dir (str | Path): root preprocessing output directory.

    Returns:
        tuple: ``(train_indices, val_indices, test_indices)`` as ``np.ndarray``.

    Raises:
        FileNotFoundError: if any index file is missing.
    """
    preprocess_dir = Path(preprocess_dir)
    paths = artifact_paths(preprocess_dir)
    for key in ("train_indices", "val_indices", "test_indices"):
        if not paths[key].exists():
            raise FileNotFoundError(
                f"Index file not found: {paths[key]}. Run preprocessing first."
            )
    train = np.load(paths["train_indices"])
    val   = np.load(paths["val_indices"])
    test  = np.load(paths["test_indices"])
    logger.info("Indices loaded - Train: %s, Val: %s, Test: %s",
                f"{len(train):,}", f"{len(val):,}", f"{len(test):,}")
    return train, val, test


def load_norm_stats(preprocess_dir: str | Path) -> dict[str, np.ndarray]:
    """
    Load normalization statistics from ``norm_stats.json``.

    Args:
        preprocess_dir (str | Path): root preprocessing output directory.

    Returns:
        dict mapping stat name to ``np.ndarray``
        (keys: ``"jet_mu"``, ``"jet_sigma"``, ``"track_mu"``, ``"track_sigma"``).

    Raises:
        FileNotFoundError: if ``norm_stats.json`` does not exist.
    """
    preprocess_dir = Path(preprocess_dir)
    path = artifact_paths(preprocess_dir)["norm_stats"]
    if not path.exists():
        raise FileNotFoundError(
            f"Norm stats not found: {path}. Run preprocessing first."
        )
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    stats = {k: np.array(v) for k, v in raw.items()}
    logger.info("Norm stats loaded from %s", path)
    return stats


def compute_normalization_stats(
    file_path: str | Path,
    train_indices: np.ndarray,
    jet_vars: list[str] | None = None,
    track_vars: list[str] | None = None,
    batch_size: int = 10_000,
) -> dict[str, np.ndarray]:
    """
    Compute per-feature mean and std on the **training set only**.

    Uses ``sklearn.preprocessing.StandardScaler.partial_fit`` to accumulate
    statistics iterating in batches. Statistics are computed
    exclusively on the training set to prevent data leakage.

    The ``pt`` jet variable is log-transformed before fitting.

    Args:
        file_path (str | Path): path to the HDF5 file.
        train_indices (np.ndarray): sorted array of training jet indices.
        jet_vars (list, optional): jet variable names to include
            (default: ``JET_VARS_DEFAULT``).
        track_vars (list, optional): track variable names to include
            (default: ``TRACK_VARS_DEFAULT``).
        batch_size (int, optional): jets per partial-fit batch (default: ``10 000``).

    Returns:
        dict with keys ``"jet_mu"``, ``"jet_sigma"``, ``"track_mu"``, ``"track_sigma"``
        (each a ``np.ndarray``).

    Raises:
        FileNotFoundError: if ``file_path`` does not exist.
        KeyError: if ``"jets"`` or ``"tracks"`` datasets are missing from the HDF5 file.
    """
    jet_vars   = list(jet_vars   if jet_vars   is not None else JET_VARS_DEFAULT)
    track_vars = list(track_vars if track_vars is not None else TRACK_VARS_DEFAULT)

    # h5py requires indices in strictly increasing order
    sorted_indices = np.sort(train_indices)
    jet_scaler   = StandardScaler()
    track_scaler = StandardScaler()

    file_path = Path(file_path)

    with h5py.File(file_path, "r") as f:
        jets_ds   = f["jets"]
        tracks_ds = f["tracks"]
        jet_dtype_names   = jets_ds.dtype.names   or ()
        track_dtype_names = tracks_ds.dtype.names or ()

        # warn and drop missing variables
        for var in [v for v in jet_vars if v not in jet_dtype_names]:
            logger.warning("Jet variable '%s' not found in HDF5 - skipping.", var)
            jet_vars.remove(var)

        for var in [v for v in track_vars if v not in track_dtype_names]:
            logger.warning("Track variable '%s' not found in HDF5 - skipping.", var)
            track_vars.remove(var)

        has_valid = "valid" in track_dtype_names
        if not has_valid:
            logger.warning(
                "'valid' field not found in tracks dataset. "
                "Assuming all tracks are valid for normalization stats."
            )

        for start in range(0, len(sorted_indices), batch_size):
            batch_idx = sorted_indices[start : start + batch_size]

            # jet features
            jets_raw  = jets_ds[batch_idx]
            jet_batch = np.empty((len(batch_idx), len(jet_vars)), dtype=np.float32)
            for i, var in enumerate(jet_vars):
                col = jets_raw[var].astype(np.float32)
                if var == "pt":
                    col = np.log(np.clip(col, 1e-8, None))
                jet_batch[:, i] = col
            jet_scaler.partial_fit(jet_batch)

            # track features
            tracks_raw  = tracks_ds[batch_idx]
            track_batch = (
                np.stack([tracks_raw[v] for v in track_vars], axis=-1)
                .astype(np.float32)
                .reshape(-1, len(track_vars))
            )
            valid_mask = (
                np.asarray(tracks_raw["valid"], dtype=bool).reshape(-1)
                if has_valid
                else np.ones(track_batch.shape[0], dtype=bool)
            )
            valid_tracks = track_batch[valid_mask]
            if valid_tracks.shape[0] == 0:
                valid_tracks = np.zeros((1, len(track_vars)), dtype=np.float32)
            track_scaler.partial_fit(valid_tracks)

            logger.debug("partial_fit: jets %d - %d", start, start + len(batch_idx))

    stats = {
        "jet_mu":      jet_scaler.mean_,
        "jet_sigma":   jet_scaler.scale_,
        "track_mu":    track_scaler.mean_,
        "track_sigma": track_scaler.scale_,
    }
    logger.info("Normalization stats computed on %s jets.", f"{len(train_indices):,}")
    return stats
