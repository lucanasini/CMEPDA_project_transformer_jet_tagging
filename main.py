'''
main.py
'''

import json
import logging
from pathlib import Path

import src.trasformer_jet_tagging.utils as utils
from src.trasformer_jet_tagging.dataset import GN2Dataset

import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GN2")

if __name__ == "__main__":
    # load configuration
    config = utils.load_config_json("/home/lnasini/Desktop/PROGETTO_CMEPDA/CMEPDA_project_transformer_jet_tagging/src/trasformer_jet_tagging/configs/config.json")

    # extract configuration parameters
    file_path      = config["data"]["h5_path"]
    preprocess_dir = Path(config["output"]["preprocess_dir"])

    jet_vars   = config["data"]["jet_features"]
    track_vars = config["data"]["track_features"]
    label_vars = config["data"]["label"]
    label_map  = config["data"]["label_map"]

    training_batch_size = config["training"].get("batch_size", 1024)
    shuffle_var         = config["data"].get("shuffle", False)

    # 1. preprocessing
    idx_dir   = preprocess_dir / "indices"
    norm_path = preprocess_dir / "norm_stats.json"

    artifacts_dir = [
        idx_dir / "train_indices.npy",
        idx_dir / "val_indices.npy",
        idx_dir / "test_indices.npy",
        norm_path,
    ]

    if not all(p.exists() for p in artifacts_dir):
        logger.info("Preprocessing artifacts not found. Running preprocess.py ...")
        import src.trasformer_jet_tagging.preprocess as preprocess
        preprocess.main(config_path="src/trasformer_jet_tagging/configs/config.json")

    for path in [idx_dir / "train_indices.npy",
                 idx_dir / "val_indices.npy",
                 idx_dir / "test_indices.npy",
                 norm_path]:
        if not path.exists():
            raise FileNotFoundError(f"Preprocessing artifact not found: {path}\nRun preprocess.py first.")

    train_indices = np.load(idx_dir / "train_indices.npy")
    val_indices   = np.load(idx_dir / "val_indices.npy")
    test_indices  = np.load(idx_dir / "test_indices.npy")

    with open(norm_path, "r") as f:
        norm_stats = {k: np.array(v) for k, v in json.load(f).items()}

    logger.info(f"Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    # 2. initialize datasets and dataloaders
    common_kwargs = dict(
        file_path       = file_path,
        n_tracks        = config["data"].get("max_tracks", 40),
        jet_vars        = jet_vars,
        track_vars      = track_vars,
        jet_flavour     = label_vars,
        jet_flavour_map = label_map,
        norm_stats      = norm_stats,
    )
    loader_kwargs = dict(
        batch_size  = training_batch_size,
        num_workers = config["training"].get("num_workers", 4),
        pin_memory  = torch.cuda.is_available(),
    )

    train_dataset = GN2Dataset(indices=train_indices, **common_kwargs)
    val_dataset   = GN2Dataset(indices=val_indices,   **common_kwargs)
    test_dataset  = GN2Dataset(indices=test_indices,  **common_kwargs)

    train_loader = DataLoader(train_dataset, shuffle=shuffle_var, **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False,       **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False,       **loader_kwargs)

    batch = next(iter(train_loader))
    logger.debug(f"Jets shape:   {batch['jet_features'].shape}")
    logger.debug(f"Tracks shape: {batch['track_features'].shape}")
    logger.debug(f"Labels shape: {batch['label'].shape}")