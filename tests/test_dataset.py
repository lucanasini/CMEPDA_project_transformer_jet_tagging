"""
test_dataset.py
"""

import numpy as np
from src.transformer_jet_tagging.dataset import GN2Dataset, _BatchCollator
from src.transformer_jet_tagging.constants import TRACK_VARS_DEFAULT


def test_dataset_len(fake_hdf5_file):
    ds = GN2Dataset(
        file_path=fake_hdf5_file,
        indices=np.arange(10),
        n_tracks=4,
        track_vars=["x", "y"],
    )

    assert len(ds) == 10


def test_getitem_output_shapes(fake_hdf5_file):
    ds = GN2Dataset(
        file_path=fake_hdf5_file,
        indices=np.arange(10),
        n_tracks=4,
        track_vars=["x", "y"],
    )

    sample = ds[0]

    assert sample["track_features"].shape == (4, 2)
    assert sample["mask"].shape == (4,)
    assert "jet_features" in sample
    assert "label" in sample


def test_getitem_finite_values(fake_hdf5_file):
    ds = GN2Dataset(
        file_path=fake_hdf5_file,
        indices=np.arange(10),
        n_tracks=4,
        track_vars=["x", "y"],
    )

    sample = ds[0]

    assert np.isfinite(sample["track_features"].numpy()).all()
    assert np.isfinite(sample["jet_features"].numpy()).all()


def test_process_jet_no_norm():
    ds = GN2Dataset.__new__(GN2Dataset)
    ds.norm_stats = None

    pt = np.array([10.0], dtype=np.float32)
    eta = np.array([2.0], dtype=np.float32)

    out = GN2Dataset._process_jet(ds, pt, eta)

    assert out.shape == (1, 2)
    assert np.isfinite(out).all()


def test_process_tracks_padding_mask():
    ds = GN2Dataset.__new__(GN2Dataset)
    ds.n_tracks = 5
    ds.track_vars = ["x", "y"]
    ds.norm_stats = None

    tracks = np.array([
        (1, 1.0, 2.0),
        (1, 2.0, 3.0),
        (0, 3.0, 4.0),
    ], dtype=[("valid","i1"),("x","f4"),("y","f4")])

    feats, mask = GN2Dataset._process_tracks(ds, tracks)

    assert feats.shape == (5, 2)
    assert mask.shape == (5,)
    assert mask.sum() == 2


def test_label_mapping():
    ds = GN2Dataset.__new__(GN2Dataset)
    ds.jet_flavour_map = {1: 0, 2: 1}

    assert ds.jet_flavour_map.get(1) == 0
    assert ds.jet_flavour_map.get(999, -1) == -1


def test_batch_collator_shapes(fake_hdf5_file):
    ds = GN2Dataset(
        file_path=fake_hdf5_file,
        indices=np.arange(10),
        n_tracks=4,
        track_vars=["x", "y"],
    )

    collator = _BatchCollator(ds)

    batch = collator([0, 1, 2, 3])

    assert batch["jet_features"].shape == (4, 2)
    assert batch["track_features"].shape == (4, 4, 2)
    assert batch["mask"].shape == (4, 4)
    assert batch["label"].shape == (4,)


def test_batch_collator_finite(fake_hdf5_file):
    ds = GN2Dataset(
        file_path=fake_hdf5_file,
        indices=np.arange(10),
        n_tracks=4,
        track_vars=["x", "y"],
    )

    collator = _BatchCollator(ds)
    batch = collator([0, 1, 2])

    assert np.isfinite(batch["jet_features"].numpy()).all()
    assert np.isfinite(batch["track_features"].numpy()).all()