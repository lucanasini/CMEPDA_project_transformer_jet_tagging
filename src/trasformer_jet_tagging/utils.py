"""
utils.py
"""

import logging
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("GN2DataLoader")

JET_VARS_DEFAULT = ['pt', 'eta']
TRACK_VARS_DEFAULT = [
    # tracks in the perigee repn
    'qOverP', 'deta', 'dphi', 'd0', 'z0SinTheta',
    # diagonal of the track cov matrix (first 3 els)
    'qOverPUncertainty', 'thetaUncertainty', 'phiUncertainty',
    # lifetime signed s(d0) and s(z0*sin(theta))
    'lifetimeSignedD0Significance', 'lifetimeSignedZ0SinThetaSignificance',
    # hit level variables
    'numberOfPixelHits', 'numberOfSCTHits',
    'numberOfInnermostPixelLayerHits', 'numberOfNextToInnermostPixelLayerHits',
    'numberOfInnermostPixelLayerSharedHits', 'numberOfInnermostPixelLayerSplitHits',
    'numberOfPixelSharedHits', 'numberOfPixelSplitHits', 'numberOfSCTSharedHits'
]

def compute_normalization_stats(
    file_path: str,
    train_indices: np.ndarray,
    jet_vars: list = None,
    track_vars: list = None,
    batch_size: int = 10_000
) -> dict:
    """Compute mean and std normalization statistics on the training set only.

    Uses sklearn's StandardScaler with partial_fit to accumulate statistics
    iterating in batches, avoiding loading the entire dataset into RAM.
    Statistics are computed exclusively on the training set to prevent
    data leakage into validation and test sets.

    Args:
        file_path (str): Path to the HDF5 file containing jets and tracks data.
        train_indices (np.ndarray): Array of integer indices identifying the
            training jets within the HDF5 file.
        jet_vars (list, optional): List of jet-level variable names to include.
            Defaults to GN2Dataset.JET_VARS_DEFAULT if not provided.
        track_vars (list, optional): List of track-level variable names to include.
            Defaults to GN2Dataset.TRACK_VARS_DEFAULT if not provided.
        batch_size (int, optional): Number of jets to process per batch during
            the partial_fit loop. Controls peak memory usage. Defaults to 10_000.

    Returns:
        dict: Normalization statistics with the following keys:

            - jet_mu (*np.ndarray*, shape ``(n_jet_vars,)``): Per-feature mean
              for jet-level variables. Index 0 corresponds to log(pT),
              index 1 to eta.
            - jet_sigma (*np.ndarray*, shape ``(n_jet_vars,)``): Per-feature
              standard deviation for jet-level variables.
            - track_mu (*np.ndarray*, shape ``(n_track_vars,)``): Per-feature
              mean computed over all valid tracks in the training set.
            - track_sigma (*np.ndarray*, shape ``(n_track_vars,)``): Per-feature
              standard deviation over all valid tracks in the training set.
    """
    jet_vars   = jet_vars   or JET_VARS_DEFAULT
    track_vars = track_vars or TRACK_VARS_DEFAULT

    jet_scaler   = StandardScaler()
    track_scaler = StandardScaler()

    # h5py requires indices in strictly increasing order
    sorted_indices = np.sort(train_indices)

    with h5py.File(file_path, 'r') as f:
        for start in range(0, len(sorted_indices), batch_size):
            batch_idx = sorted_indices[start : start + batch_size]

            # --- Jet features ---
            jets_raw   = f['jets'][batch_idx]
            jet_pt_log = np.log(jets_raw['pt']).reshape(-1, 1)
            jet_eta    = jets_raw['eta'].reshape(-1, 1)
            jet_batch  = np.hstack([jet_pt_log, jet_eta])
            jet_scaler.partial_fit(jet_batch)

            # --- Track features ---
            tracks_raw   = f['tracks'][batch_idx]
            valid_mask   = tracks_raw['valid'].astype(bool)
            track_batch  = np.stack(
                [tracks_raw[var] for var in track_vars], axis=-1
            )
            valid_tracks = track_batch[valid_mask]

            if len(valid_tracks) > 0:
                track_scaler.partial_fit(valid_tracks)

            logger.debug(f"partial_fit: batch {start}–{start + len(batch_idx)} done.")

    norm_stats = {
        'jet_mu':      jet_scaler.mean_,
        'jet_sigma':   jet_scaler.scale_,
        'track_mu':    track_scaler.mean_,
        'track_sigma': track_scaler.scale_,
    }

    logger.info(f"Normalization stats computed on {len(train_indices)} jets.")
    logger.info(f"jet_mu:    {norm_stats['jet_mu']}")
    logger.info(f"jet_sigma: {norm_stats['jet_sigma']}")
    return norm_stats