import argparse

import pytest

from sparse.config import (
    ResidualSummaryConfig,
    add_residual_summary_arguments,
    residual_summary_config_from_args,
)


def test_residual_summary_cli_defaults_and_explicit_values():
    parser = argparse.ArgumentParser()
    add_residual_summary_arguments(parser)
    defaults = residual_summary_config_from_args(parser.parse_args([]))
    assert defaults == ResidualSummaryConfig()

    configured = residual_summary_config_from_args(
        parser.parse_args(
            [
                "--residual-summaries",
                "--summary-prototypes",
                "4",
                "--summary-mode",
                "learned",
                "--summary-checkpoint",
                "compressor.pt",
            ]
        )
    )
    assert configured == ResidualSummaryConfig(
        enabled=True,
        prototypes=4,
        mode="learned",
        checkpoint="compressor.pt",
    )


def test_learned_enabled_mode_requires_checkpoint():
    with pytest.raises(ValueError, match="requires --summary-checkpoint"):
        ResidualSummaryConfig(enabled=True, mode="learned")

