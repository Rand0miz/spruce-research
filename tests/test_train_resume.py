from unittest.mock import Mock

import torch

import pytest

from selector.train import load_resume_checkpoint, validate_resume_recipe


def test_resume_moves_cuda_rng_states_back_to_cpu(monkeypatch):
    gate = Mock()
    optimizer = Mock()
    config = {"num_layers": 2, "head_dim": 4, "proj_dim": 4}
    checkpoint = {
        "config": config,
        "state_dict": {"gate": torch.tensor([1.0])},
        "optimizer": {"state": {}},
        "epoch": 25,
        "history": {"epochs": [1, 25]},
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": [
            torch.arange(16, dtype=torch.uint8),
        ],
    }
    restored = []

    monkeypatch.setattr(
        "selector.train.torch.load",
        lambda *args, **kwargs: checkpoint,
    )
    monkeypatch.setattr(
        "selector.train.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "selector.train.torch.cuda.set_rng_state_all",
        lambda states: restored.extend(states),
    )

    epoch, history = load_resume_checkpoint(
        "resume.pt", gate, optimizer, config, "cuda")

    assert epoch == 25
    assert history == {"epochs": [1, 25]}
    assert len(restored) == 1
    assert restored[0].device.type == "cpu"
    assert restored[0].dtype == torch.uint8


def test_resume_recipe_rejects_tree_objective_drift():
    saved = {
        "lr": 2e-4,
        "lambda_topk": 0.5,
        "lambda_boundary": 0.5,
        "lambda_needle": 1.0,
        "topk": 8,
        "topk_margin": 0.25,
        "needle_topk": 8,
        "needle_margin": 0.25,
        "needle_objective": "union",
        "tree_supervision": True,
        "tree_radix": 2,
        "tree_beam": 8,
        "natural_fraction": 0.8,
        "shuffle_targets": True,
    }
    expected = dict(saved)
    expected["lambda_needle"] = 0.25

    with pytest.raises(SystemExit, match="lambda_needle"):
        validate_resume_recipe(saved, expected)


def test_legacy_flat_resume_gets_safe_new_objective_defaults():
    validate_resume_recipe(
        {"lr": 5e-4, "lambda_topk": 0.75},
        {
            "lr": 5e-4,
            "lambda_topk": 0.75,
            "lambda_boundary": 0.0,
            "topk_margin": 0.0,
            "needle_margin": 0.0,
            "needle_objective": "group",
            "tree_supervision": False,
            "tree_radix": 2,
            "tree_beam": 8,
        },
    )
