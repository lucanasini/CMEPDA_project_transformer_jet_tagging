"""
test_model.py
"""

import torch
import pytest
from src.transformer_jet_tagging.model import GN2, _get_activation

@pytest.fixture
def dummy_input():
    B, T = 2, 5
    return {
        "jet_features": torch.randn(B, 3),
        "track_features": torch.randn(B, T, 4),
        "mask": torch.ones(B, T, dtype=torch.bool),
    }


def test_model_forward_shape(dummy_input):
    model = GN2(
        n_jet_vars=3,
        n_track_vars=4,
        n_classes=4,
        embed_dim=32,
        n_heads=4,
        n_layers=2,
        ff_dim=64,
        pool_dim=16,
    )

    out = model(**dummy_input)
    assert "jet_outputs" in out
    assert out["jet_outputs"].shape == (2, 4)


def test_predict_proba(dummy_input):
    model = GN2(3, 4)

    p = model.predict_proba(**dummy_input)
    assert p.shape == (2, 4)
    assert torch.allclose(p.sum(dim=1), torch.ones(2), atol=1e-5)


def test_activation_factory():
    assert isinstance(_get_activation("relu"), torch.nn.ReLU)
    with pytest.raises(ValueError):
        _get_activation("not_existing")