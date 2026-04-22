"""
test_utils.py
"""

import numpy as np
import h5py
import tempfile
from src.transformer_jet_tagging.utils import load_config_json, compute_normalization_stats
from src.transformer_jet_tagging.constants import TRACK_VARS_DEFAULT


def test_load_config(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text('{"a": 1}')

    cfg = load_config_json(str(p))
    assert cfg["a"] == 1


def test_compute_norm_stats_minimal():
    with tempfile.NamedTemporaryFile(suffix=".h5") as f:
        with h5py.File(f.name, "w") as h5:
            jets = h5.create_dataset(
                "jets",
                shape=(10,),
                dtype=[("pt", "f4"), ("eta", "f4")]
            )
            tracks = h5.create_dataset(
                "tracks",
                shape=(10,),
                dtype=[("x", "f4"), ("y", "f4"), ("valid", "i1")]
            )

            jets["pt"] = np.arange(10) + 1
            jets["eta"] = np.ones(10)

            tracks["x"] = np.random.randn(10)
            tracks["y"] = np.random.randn(10)
            tracks["valid"] = np.ones(10)

        stats = compute_normalization_stats(
            file_path=f.name,
            train_indices=np.arange(10),
            jet_vars=["pt", "eta"],
            track_vars=["x", "y"]
        )

        assert "jet_mu" in stats
        assert "track_mu" in stats