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

Outputs (under the directory specified in config["output"]["preprocess_dir"]):
  preprocess_dir/
  ├── indices/
  │   ├── train_indices.npy
  │   ├── val_indices.npy
  │   └── test_indices.npy
  └── norm_stats.json

Usage:
    python preprocess.py --config configs/config.json
"""

import logging

import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("GN2.plotting")




def plot_var (dataloader: DataLoader):
    # all_labels = []
    # all_values = []

    # for batch_data, batch_labels in dataloader:
    #     all_values.append(batch_data)
    #     all_labels.append(batch_labels)

    # # Concatena tutti i batch
    # all_values = torch.cat(all_values).numpy()
    # all_labels = torch.cat(all_labels).numpy()

    # plt.hist(all_values.flatten(), bins=50)
    # plt.show()

    # Concatena direttamente durante l'iterazione
    data = np.concatenate([batch[0].numpy() for batch in dataloader])

    plt.figure(figsize=(10, 4))
    plt.hist(data.flatten(), bins=50, edgecolor='black')
    plt.title("Distribuzione dei dati")
    plt.xlabel("Valore")
    plt.ylabel("Frequenza")
    plt.show()





    


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="GN2 plot variables pipeline")
#     parser.add_argument(
#         "--config",
#         type=str,
#         default="src/trasformer_jet_tagging/configs/config.json",
#         help="Path to the JSON configuration file.",
#     )
#     args = parser.parse_args()
#     config_path = args.config
#     main(config_path)