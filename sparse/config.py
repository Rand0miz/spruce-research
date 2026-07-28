"""Configuration shared by residual-summary CLI and attention paths."""

from __future__ import annotations

from dataclasses import dataclass


SUMMARY_PROTOTYPES = (1, 2, 4)
SUMMARY_MODES = ("mean", "learned")


@dataclass(frozen=True)
class ResidualSummaryConfig:
    enabled: bool = False
    prototypes: int = 1
    mode: str = "mean"
    checkpoint: str | None = None

    def __post_init__(self) -> None:
        if self.prototypes not in SUMMARY_PROTOTYPES:
            raise ValueError(
                f"summary prototypes must be one of {SUMMARY_PROTOTYPES}, "
                f"got {self.prototypes}"
            )
        if self.mode not in SUMMARY_MODES:
            raise ValueError(
                f"summary mode must be one of {SUMMARY_MODES}, got {self.mode!r}"
            )
        if self.mode == "learned" and self.enabled and not self.checkpoint:
            raise ValueError("learned summary mode requires --summary-checkpoint")

    def attention_kwargs(self) -> dict:
        return {
            "residual_summaries": self.enabled,
            "summary_prototypes": self.prototypes,
            "summary_mode": self.mode,
            "summary_checkpoint": self.checkpoint,
        }


def add_residual_summary_arguments(parser) -> None:
    parser.add_argument(
        "--residual-summaries",
        action="store_true",
        help="replace omitted causal blocks with deterministic tree-node K/V summaries",
    )
    parser.add_argument(
        "--summary-prototypes",
        type=int,
        choices=SUMMARY_PROTOTYPES,
        default=1,
        help="contiguous positional prototypes per residual tree node",
    )
    parser.add_argument(
        "--summary-mode",
        choices=SUMMARY_MODES,
        default="mean",
        help="summary compressor; learned mode requires a trained checkpoint",
    )
    parser.add_argument(
        "--summary-checkpoint",
        help="learned compressor checkpoint (unused by deterministic mean mode)",
    )


def residual_summary_config_from_args(args) -> ResidualSummaryConfig:
    return ResidualSummaryConfig(
        enabled=bool(args.residual_summaries),
        prototypes=int(args.summary_prototypes),
        mode=str(args.summary_mode),
        checkpoint=args.summary_checkpoint,
    )

