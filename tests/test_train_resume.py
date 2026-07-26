from unittest.mock import Mock

import torch

from selector.train import load_resume_checkpoint


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
