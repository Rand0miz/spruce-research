import pytest

from configs.long_context import (
    QWEN_128K_CONTEXT,
    apply_yarn,
    configure_tokenizer,
    context_limit,
    yarn_metadata,
)


class ModernConfig:
    max_position_embeddings = 32768
    rope_parameters = {"rope_type": "default", "rope_theta": 1_000_000.0}


class LegacyConfig:
    max_position_embeddings = 32768
    rope_scaling = None
    rope_theta = 1_000_000.0


class Tokenizer:
    model_max_length = 32768


def test_four_x_yarn_builds_131072_modern_config():
    config = apply_yarn(ModernConfig(), yarn_factor=4.0)
    assert config.max_position_embeddings == QWEN_128K_CONTEXT
    assert config.rope_parameters == {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
        "rope_theta": 1_000_000.0,
    }


def test_four_x_yarn_supports_legacy_rope_scaling_name():
    config = apply_yarn(LegacyConfig(), yarn_factor=4.0)
    assert config.rope_scaling["type"] == "yarn"
    assert config.rope_scaling["factor"] == 4.0
    assert config.max_position_embeddings == QWEN_128K_CONTEXT


def test_tokenizer_and_metadata_share_context_limit():
    tokenizer = configure_tokenizer(Tokenizer(), yarn_factor=4.0)
    metadata = yarn_metadata(yarn_factor=4.0)
    assert tokenizer.model_max_length == context_limit(yarn_factor=4.0)
    assert metadata["max_position_embeddings"] == QWEN_128K_CONTEXT
    assert metadata["enabled"] is True


def test_invalid_yarn_factor_is_rejected():
    with pytest.raises(ValueError, match="yarn_factor"):
        context_limit(yarn_factor=0.5)

