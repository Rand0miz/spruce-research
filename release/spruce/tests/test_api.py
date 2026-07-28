import json

import pytest

from spruce_attn import CompilerConfig, SpruceCompiler
from spruce_attn.cli import main

from tests.helpers import WhitespaceTokenizer


def test_frozen_defaults():
    config = CompilerConfig()
    assert config.beam == 16
    assert config.candidate_blocks == 4
    assert config.feature_dim == 512
    assert config.block_radius == 1
    assert config.boundary == "paragraph"


def test_bad_configuration_is_rejected():
    with pytest.raises(ValueError, match="beam"):
        CompilerConfig(beam=2, candidate_blocks=4)


def test_public_api_compiles_exact_source_text():
    tokenizer = WhitespaceTokenizer()
    compiler = SpruceCompiler(
        tokenizer,
        CompilerConfig(
            block_size=8,
            candidate_blocks=2,
            beam=4,
            feature_dim=128,
        ),
    )
    document = "\n\n".join([
        "Routine records describe staffing and ordinary maintenance.",
        "The final committee register approved code CEDAR-441.",
        "A draft mentioned code MAPLE-103 but it was rejected.",
        "Closing notes cover the archive schedule.",
    ])
    result = compiler.compile(
        document, "Which code did the final committee register approve?")
    assert "CEDAR-441" in result.content
    assert result.packet.compiled_prompt_tokens > 0
    assert result.visited_nodes > 0
    assert result.metadata()["selected_blocks"]


def test_cli_info_is_machine_readable(capsys):
    assert main(["info"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["package"] == "spruce-attn"
    assert payload["version"] == "0.1.0"
