"""
dataset.py
==========
High-performance GN2 data pipeline for HDF5 datasets. 
Optimized for Large-Scale Jet Flavour Tagging at ATLAS.

HDF5 Structure:
  /jets          — Jet-level features (1 row per jet).
  /tracks        — Track-level features (Jagged or fixed-size array indexed by jet).
  /eventwise     — Event-level metadata (e.g., eventNumber, mu).
  /truth_hadrons — Simulation truth for hadron labeling and performance studies.

The pipeline utilizes 'Lazy Loading' via h5py to handle datasets that exceed 
available RAM, using NumPy vectorization for high-speed feature extraction.

Pipeline Workflow:
  1. Index Splitting: Generate Train/Val/Test indices using scikit-learn's 
     train_test_split to ensure zero data leakage.
  2. Data Loading: On-the-fly extraction of jet and track features using 
     global indices to map to the HDF5 file structure.
  3. Track Quality Filtering: Active filtering of track candidates using 
      the 'valid' boolean flag before processing.
  4. Padding & Masking: Enforce a fixed-size track array (default max_tracks=40). 
     Generate a boolean padding mask to inform the Transformer's Self-Attention 
     mechanism which inputs to ignore.
  5. Feature Engineering: 
     - Jet-level: Log-transformation of pT and Z-score standardization.
     - Track-level: Z-score standardization (mu and sigma computed ONLY on training set).
  6. Class Balancing: Optional 2D re-sampling (pT, eta) to flatten the 
     background distributions and match the reference class (c-jets).
  7. Integration: Wraps the logic into a PyTorch DataLoader with 
     multi-process worker support and pinned memory for GPU acceleration.
"""

import logging
from typing import Dict, Tuple, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# logging configuration
logger = logging.getLogger("GN2.dataset")

JET_FLAVOUR_MAP = {0: 0, 4: 1, 5: 2, 15: 3}  # light, c, b, tau
JET_FLAVOUR = 'HadronConeExclTruthLabelID'
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

class GN2Dataset(Dataset):
    """
    Dataset for flavour tagger using HDF5 file.
    Lazy loading of data for large datasets with Numpy vectorization.
    Includes filtering of invalid tracks and feature normalization. 

    Attributes:
        file_path (str): path of .h5 file.
        indices (np.ndarray): indices of jets to include in the dataset.
        n_tracks (int, optional): maximum number of tracks for each jet (padding/cropping).
        jet_vars (list, optional): list of jet variables.
        track_vars (list, optional): list of tracks variables.
        jet_flavour (str, optional): name of the jet flavour variable in the HDF5 file.
        jet_flavour_map (dict, optional): mapping from raw hadron labels to target classes.
        norm_stats (dict, optional): normalization statistics for jet and track features.
    """

    def __init__(
        self, 
        file_path: str,
        indices: np.ndarray,
        n_tracks: int = 40,
        jet_vars: Optional[list] = JET_VARS_DEFAULT,
        track_vars: Optional[list] = TRACK_VARS_DEFAULT,
        jet_flavour : Optional[str] = JET_FLAVOUR,
        jet_flavour_map: Optional[Dict[int, int]] = JET_FLAVOUR_MAP,
        norm_stats: Optional[Dict] = None
    ):
        """
        Initialize the dataset.

        Args:
            file_path (str): path of .h5 file.
            indices (np.ndarray): indices of jets to include in the dataset.
            n_tracks (int, optional): maximum number of tracks for each jet (padding/cropping).
            jet_vars (list, optional): list of jet variables. Defaults to JET_VARS_DEFAULT if not provided.
            track_vars (list, optional): list of tracks variables. Defaults to TRACK_VARS_DEFAULT if not provided.
            jet_flavour (str, optional): name of the jet flavour variable in the HDF5 file. Defaults to JET_FLAVOUR if not provided.
            jet_flavour_map (dict, optional): mapping from raw hadron labels to target classes. Defaults to JET_FLAVOUR_MAP if not provided.
            norm_stats (dict, optional): normalization statistics for jet and track features. Defaults to None (no normalization) if not provided.

        Raises:
            FileNotFoundError: if the specified HDF5 file does not exist.
            KeyError: if the expected datasets ('jets', 'tracks') are not found in the HDF5 file.
        """
        self.file_path  = file_path
        self.indices    = indices
        self.n_tracks   = n_tracks
        self.jet_vars   = jet_vars
        self.track_vars = track_vars
        self.jet_flavour = jet_flavour
        self.jet_flavour_map = jet_flavour_map
        self.norm_stats = norm_stats
        if norm_stats is None:
            logger.warning("No norm_stats provided — raw values will be used.")

        # initialize h5py file handler as None; will be opened lazily in _get_handler()
        self.handler = None
        
        # initial check
        try:
            with h5py.File(self.file_path, 'r') as f:
                self.n_jets = len(f['jets'])
                logger.info(f"Success loading {file_path}: {self.n_jets} jets found.")
                logger.debug(f"Original shape 'tracks': {f['tracks'].shape}")
        except (FileNotFoundError, KeyError) as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise

    def _get_handler(self) -> h5py.File:
        """
        Manage the h5py file handler for multiprocessing.
        
        Returns:
            h5py.File: h5py file object open.
        """
        if self.handler is None:
            try:
                self.handler = h5py.File(self.file_path, 'r', swmr=True)        # swmr=True allows multiple readers (for num_workers > 0)
            except OSError:
                logger.warning("SWMR mode not supported on this filesystem. Standard read.")
                self.handler = h5py.File(self.file_path, 'r')
        return self.handler

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Returns the shape of the dataset.

        Returns:
            Tuple[int, int, int]: (n_jets, n_tracks, n_track_features)
        """
        return (self.n_jets, self.n_tracks, len(self.track_vars))

    def __len__(self) -> int:
        """
        Calculate the number of selected jets in the dataset.

        Returns:
            int: number of jets in the dataset.
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Extract a single jet and its associated tracks.

        Optimization: uses h5py slicing to avoid Python loops.
        The tracks are extracted as a subset and then padded to n_tracks with zeros.
        The mask indicates which tracks are valid (True) or padded (False).

        Args:
            idx: index of the jet to extract.

        Returns:
            dict: {
                "jet_features"      (torch.Tensor, shape (n_jet_vars,)):            normalized jet-level features,
                "track_features"    (torch.Tensor, shape (n_tracks, n_track_vars)): normalized track-level features,
                "mask"              (torch.Tensor, shape (n_tracks,)):              boolean mask indicating valid tracks,
                "label"             (torch.Tensor, shape ()):                       ground truth label
            }
        """
        f = self._get_handler()

        real_idx = self.indices[idx]  # maps the dataset index to the actual jet index in the file

        # 1. Loading Jet Features and normalization
        jet_data = f['jets'][real_idx]
        
        jet_pt  = jet_data['pt']
        jet_eta = jet_data['eta']
        # pt log-trasformation
        jet_pt_log = np.log(jet_pt)

        if self.norm_stats and 'jet_mu' in self.norm_stats and 'jet_sigma' in self.norm_stats:
            if len(self.norm_stats['jet_mu']) < 2 or len(self.norm_stats['jet_sigma']) < 2:
                logger.warning("Normalization stats for jets are incomplete. Expected at least 2 values for 'jet_mu' and 'jet_sigma'. Using raw values.")
            else:
                jet_pt_log = (jet_pt_log - self.norm_stats['jet_mu'][0]) / self.norm_stats['jet_sigma'][0]
                jet_eta    = (jet_eta - self.norm_stats['jet_mu'][1]) / self.norm_stats['jet_sigma'][1]
        
        jet_features = np.array([jet_pt_log, jet_eta], dtype=np.float32)
        
        # 2. Loading Label
        raw_label = jet_data[self.jet_flavour]
        target = self.jet_flavour_map.get(int(raw_label), 0)

        # 3. Loading Tracks with 'valid' Filter (Optimized with slicing)
        tracks_all = f['tracks'][real_idx]
        # bool mask
        valid_tracks = tracks_all[tracks_all['valid'] == True]
        
        n_available = len(valid_tracks)
        n_to_read   = min(n_available, self.n_tracks)

        # pre-allocate arrays for track features and mask
        track_features = np.zeros((self.n_tracks, len(self.track_vars)), dtype=np.float32)
        padding_mask = np.zeros(self.n_tracks, dtype=bool)

        # vectorized extraction: read features for available tracks
        if n_to_read > 0:
            for i, var in enumerate(self.track_vars):
                raw_values = valid_tracks[var][:n_to_read]
                # normalization
                if self.norm_stats and 'track_mu' in self.norm_stats and 'track_sigma' in self.norm_stats:
                    mu    = self.norm_stats['track_mu'][i]
                    sigma = self.norm_stats['track_sigma'][i]
                    track_features[:n_to_read, i] = (raw_values - mu) / sigma
                else:
                    logger.info(f"Normalization stats not provided for track variable '{var}'. Using raw values.")
                    track_features[:n_to_read, i] = raw_values
            
            padding_mask[:n_to_read] = True

        return {
            'jet_features':   torch.from_numpy(jet_features),
            'track_features': torch.from_numpy(track_features),
            'mask':           torch.from_numpy(padding_mask),
            'label':          torch.tensor(target, dtype=torch.long)
        }

if __name__ == "__main__":

    import argparse
    import sys
    from utils import compute_normalization_stats
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser(
        description="Test for GN2Dataset: loads a sample, checks shapes and normalization."
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the HDF5 file.",
    )
    parser.add_argument(
        "--n-tracks",
        type=int, 
        default=40,
        help="Maximum number of tracks per jet (default: 40)."
    )
    parser.add_argument(
        "--n-jets",
        type=int,
        default=None,
        help="Limit the test to the first N jets (default: all)."
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Fraction of jets used for the training split (default: 0.7)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/test split (default: 42)."
    )
    args = parser.parse_args()

    try:
        with h5py.File(args.file, 'r') as f:
            n_total = len(f['jets'])
    except (FileNotFoundError, KeyError) as e:
        logger.error(f"Cannot open HDF5 file: {e}")
        sys.exit(1)

    n_jets  = n_total if args.n_jets is None else min(args.n_jets, n_total)
    indices = np.arange(n_jets)
    logger.info(f"Total jets in file: {n_total:,}")
    logger.info(f"Jets used for test: {n_jets:,}")

    train_indices, test_indices = train_test_split(
        indices,
        train_size=args.train_frac,
        random_state=args.seed,
        shuffle=True
    )
    logger.info(f"Split - train: {len(train_indices):,}  test: {len(test_indices):,}")

    logger.info("Computing normalization statistics on training set ...")
    norm_stats = compute_normalization_stats(args.file, train_indices)

    train_dataset = GN2Dataset(
        args.file,
        indices=train_indices,
        n_tracks=args.n_tracks,
        norm_stats=norm_stats
    )
    test_dataset  = GN2Dataset(
        args.file,
        indices=test_indices,
        n_tracks=args.n_tracks,
        norm_stats=norm_stats
    )


    sample = train_dataset[0]
    expected = {
        "jet_features":   (len(JET_VARS_DEFAULT),),
        "track_features": (args.n_tracks, len(TRACK_VARS_DEFAULT)),
        "mask":           (args.n_tracks,),
        "label":          (),
    }
    all_ok = True
    for key, exp_shape in expected.items():
        got = tuple(sample[key].shape)
        status = "OK" if got == exp_shape else "FAIL"
        if status == "FAIL":
            all_ok = False
        logger.info(f"  {key:<20} expected {str(exp_shape):<25} got {str(got):<25} [{status}]")

    n_valid = sample["mask"].sum().item()
    logger.info(f"  valid tracks in sample[0] : {n_valid} / {args.n_tracks}")
    logger.info(f"  label in sample[0]        : {sample['label'].item()}"
                f" ({[k for k,v in JET_FLAVOUR_MAP.items() if v == sample['label'].item()]})")

    # ------------------------------------------------------------------
    # 6. Normalization sanity check on a small batch
    # ------------------------------------------------------------------
    logger.info("--- Normalization sanity check (first 1000 training jets) ---")

    n_check = min(1_000, len(train_dataset))
    jet_pts, jet_etas = [], []
    for i in range(n_check):
        s = train_dataset[i]
        jet_pts.append(s["jet_features"][0].item())
        jet_etas.append(s["jet_features"][1].item())

    jet_pts  = np.array(jet_pts)
    jet_etas = np.array(jet_etas)
    logger.info(f"  jet pt  (normalized) — mean: {jet_pts.mean():.4f}  std: {jet_pts.std():.4f}  (expect ~0, ~1)")
    logger.info(f"  jet eta (normalized) — mean: {jet_etas.mean():.4f}  std: {jet_etas.std():.4f}  (expect ~0, ~1)")

    # ------------------------------------------------------------------
    # 7. __len__ consistency
    # ------------------------------------------------------------------
    logger.info("--- Length consistency ---")
    assert len(train_dataset) == len(train_indices), "train __len__ mismatch"
    assert len(test_dataset)  == len(test_indices),  "test  __len__ mismatch"
    logger.info(f"  train_dataset.__len__() = {len(train_dataset):,}  OK")
    logger.info(f"  test_dataset.__len__()  = {len(test_dataset):,}  OK")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if all_ok:
        logger.info("All checks passed.")
    else:
        logger.error("One or more shape checks failed. See above.")
        sys.exit(1)