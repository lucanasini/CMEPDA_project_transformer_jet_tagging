"""
conftest.py
"""

import pytest
import numpy as np
import h5py
import tempfile
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.transformer_jet_tagging.constants import (
    JET_FLAVOUR_LABEL,
    TRACK_VARS_DEFAULT,
    JET_VARS_DEFAULT,
)


@pytest.fixture
def fake_hdf5_file(tmp_path):
    path = tmp_path / "test.h5"
    with h5py.File(path, "w") as f:
        # Create jets dataset
        f.create_dataset("jets", data=np.zeros(10, dtype=[("pt", "f4"), ("eta", "f4")]))
        
        # Define the fields expected by track_vars=["x", "y"]
        track_dtype = np.dtype([
            ("x", "f4"),
            ("y", "f4"),
            ("valid", "i1")
        ])
        f.create_dataset("tracks", data=np.zeros((10, 5), dtype=track_dtype))
    return str(path)