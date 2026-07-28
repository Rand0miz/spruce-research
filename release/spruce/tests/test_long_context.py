from types import SimpleNamespace

from spruce_attn.long_context import apply_yarn, context_limit


def test_static_yarn_sets_128k_limit():
    config = SimpleNamespace(
        max_position_embeddings=32768,
        rope_scaling=None,
        rope_theta=1_000_000.0,
    )
    apply_yarn(
        config, yarn_factor=4.0,
        original_max_position_embeddings=32768)
    assert context_limit(
        yarn_factor=4.0,
        original_max_position_embeddings=32768) == 131072
    assert config.max_position_embeddings == 131072
    assert config.rope_scaling["type"] == "yarn"
