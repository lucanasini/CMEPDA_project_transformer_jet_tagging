"""
preprocess.py
=============
Standalone preprocessing script for the GN2 jet flavour tagging pipeline.

Run this script ONCE before training. It will:
    1. Load jet kinematics from the HDF5 file.
    2. Apply kinematic selection (pT, eta cuts).
    3. Split valid indices into train / val / test sets.
    4. Compute normalization statistics (mu, sigma) on the training set only.
    5. Save indices and norm stats to disk.

Outputs (under ``config["output"]["preprocess_dir"]``):

.. code-block:: text

    preprocess_dir/
    ├── indices/
    │   ├── train_indices.npy
    │   ├── val_indices.npy
    │   └── test_indices.npy
    └── norm_stats.json
"""

import json
import logging
from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

from . import utils

logger = logging.getLogger("GN2.preprocess")


def save_indices(
    output_dir: str | Path,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> None:
    """
    Save train / val / test index arrays as ``.npy`` files.

    Args:
        output_dir (str | Path): Base directory to save indices.
        train (np.ndarray): Array of training indices.
        val (np.ndarray): Array of validation indices.
        test (np.ndarray): Array of test indices.
    """
    idx_dir = Path(output_dir) / "indices"
    idx_dir.mkdir(parents=True, exist_ok=True)

    np.save(idx_dir / "train_indices.npy", train)
    np.save(idx_dir / "val_indices.npy",   val)
    np.save(idx_dir / "test_indices.npy",  test)

    logger.info("Indices saved to %s", idx_dir)
    logger.debug("    Train : %s jets", f"{len(train):>8,}")
    logger.debug("    Val   : %s jets", f"{len(val):>8,}")
    logger.debug("    Test  : %s jets", f"{len(test):>8,}")


def save_norm_stats(output_dir: str | Path, norm_stats: dict[str, np.ndarray]) -> None:
    """
    Serialize normalization statistics (numpy arrays) to ``norm_stats.json``.

    Args:
        output_dir (str | Path): Directory in which ``norm_stats.json`` will be written.
        norm_stats (dict): Dict mapping stat name to numpy array
            (``"jet_mu"``, ``"jet_sigma"``, ``"track_mu"``, ``"track_sigma"``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "norm_stats.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({k: v.tolist() for k, v in norm_stats.items()}, f, indent=4)

    logger.info("Normalization stats saved to %s", out_path)


def run_preprocess(config: dict) -> None:
    """
    Run the full preprocessing pipeline.

    Reads all settings from ``config`` (already-parsed)
    and writes the preprocessing artifacts to disk.

    Config keys read:
        - `data`:
            - `h5_path` (str)
            - `pt_min_mev` (float)
            - `pt_max_mev` (float)
            - `eta_max` (float)
            - `jet_features` (list)
            - `track_features` (list)
            - `train_fraction` (float)
            - `val_fraction` (float)
            - `test_fraction` (float)
            - `shuffle` (bool, default ``False``)
            - `split_seed` (int, default ``42``)
            - `norm_batch_size` (int, default ``10000``)
        - `output`:
            - `preprocess_dir` (str)

    Args:
        config (dict): Full configuration dict.

    Raises:
        FileNotFoundError: If the HDF5 file is not found.
        KeyError: If expected datasets are missing from the HDF5 file.
        ValueError: If the train/val/test fractions do not sum to 1.
    """
    # 1. load configuration
    data_config = config["data"]
    output_dir = Path(config["output"]["preprocess_dir"])

    file_path  = Path(data_config["h5_path"])
    pt_min     = data_config["pt_min_mev"]
    pt_max     = data_config["pt_max_mev"]
    eta_max    = data_config["eta_max"]
    jet_vars   = data_config["jet_features"]
    track_vars = data_config["track_features"]

    train_frac = data_config["train_fraction"]
    val_frac   = data_config["val_fraction"]
    test_frac  = data_config["test_fraction"]

    total = train_frac + val_frac + test_frac
    if abs(total - 1.) > 1e-6:
        logger.warning(
            "Fractions sum to %.6f, normalizing to 1.0", total
        )
        train_frac /= total
        val_frac   /= total
        test_frac  /= total

    shuffle    = data_config.get("shuffle", False)
    seed       = data_config.get("split_seed", 42)
    batch_size = data_config.get("norm_batch_size", 10_000)

    # 2. kinematic selection
    logger.debug("Reading kinematics from %s ...", file_path)
    try:
        with h5py.File(file_path, "r") as f:
            pt  = f["jets"]["pt"][:]
            eta = f["jets"]["eta"][:]
    except FileNotFoundError:
        logger.error("HDF5 file not found: %s", file_path)
        raise
    except KeyError as e:
        logger.error("Expected dataset not found in HDF5: %s", e)
        raise

    kinematic_mask = (pt > pt_min) & (pt < pt_max) & (np.abs(eta) < eta_max)
    valid_indices  = np.where(kinematic_mask)[0]
    logger.debug("Jets passing kinematic selection: %s / %s",
                f"{len(valid_indices):,}", f"{len(pt):,}")

    # 3. train / val / test split
    train_val_indices, test_indices = train_test_split(
        valid_indices,
        train_size   = train_frac + val_frac,
        random_state = seed,
        shuffle      = shuffle,
    )
    train_indices, val_indices = train_test_split(
        train_val_indices,
        train_size   = train_frac / (train_frac + val_frac),
        random_state = seed,
        shuffle      = shuffle,
    )
    save_indices(output_dir, train_indices, val_indices, test_indices)

    # 4. normalization statistics (training set only)
    logger.info("Computing normalization statistics on training set ...")
    norm_stats = utils.compute_normalization_stats(
        file_path     = file_path,
        train_indices = train_indices,
        jet_vars      = jet_vars,
        track_vars    = track_vars,
        batch_size    = batch_size,
    )
    save_norm_stats(output_dir, norm_stats)

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    pass
