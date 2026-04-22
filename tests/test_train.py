"""
test_train.py
"""

import torch
import pytest
from torch.optim import AdamW

from src.transformer_jet_tagging.train import GN2Loss, lr_scheduler
from src.transformer_jet_tagging.model import GN2


def test_loss_forward():
    loss = GN2Loss()

    outputs = {"jet_outputs": torch.randn(4, 4)}
    labels = {"jet_label": torch.tensor([0, 1, 2, 3])}

    out = loss(outputs, labels)

    assert "total" in out
    assert out["total"].shape == torch.Size([])


def test_scheduler_steps():
    model = GN2(3, 4)
    opt = AdamW(model.parameters(), lr=1e-3)

    sched = lr_scheduler(opt, n_total_steps=100)

    lr_before = opt.param_groups[0]["lr"]
    sched.step()
    lr_after = opt.param_groups[0]["lr"]

    assert lr_after != lr_before