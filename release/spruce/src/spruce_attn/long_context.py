"""Explicit Qwen YaRN configuration for long-context experiments.

Qwen2.5-Coder checkpoints ship with a 32,768-token config even though the
model card documents 4x YaRN for inputs up to 131,072 tokens.  Keep the
override in one place so teacher extraction, sparse replay, and dense
baselines cannot silently use different rotary embeddings.
"""
from __future__ import annotations


QWEN_NATIVE_CONTEXT = 32_768
QWEN_128K_CONTEXT = 131_072
QWEN_128K_YARN_FACTOR = 4.0


def context_limit(*, yarn_factor=1.0, original_max_position_embeddings=QWEN_NATIVE_CONTEXT):
    """Return the configured context limit for a static YaRN factor."""
    factor = float(yarn_factor)
    original = int(original_max_position_embeddings)
    if factor < 1.0:
        raise ValueError(f"yarn_factor must be >= 1, got {factor}")
    if original < 1:
        raise ValueError(
            "original_max_position_embeddings must be >= 1")
    return int(original * factor)


def apply_yarn(
        config, *, yarn_factor=1.0,
        original_max_position_embeddings=QWEN_NATIVE_CONTEXT):
    """Apply a Transformers-compatible static YaRN override in place.

    Transformers 5 uses ``rope_parameters``/``rope_type``. Older supported
    releases expose ``rope_scaling``/``type``. The model config object tells
    us which spelling its Qwen implementation consumes.
    """
    factor = float(yarn_factor)
    original = int(original_max_position_embeddings)
    maximum = context_limit(
        yarn_factor=factor,
        original_max_position_embeddings=original)
    if factor == 1.0:
        return config

    current = getattr(config, "rope_parameters", None)
    if not current:
        current = getattr(config, "rope_scaling", None)
    current = current or {}
    rope_theta = current.get(
        "rope_theta", getattr(config, "rope_theta", None))

    config.max_position_embeddings = maximum
    if hasattr(config, "rope_parameters"):
        parameters = {
            "rope_type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": original,
        }
        if rope_theta is not None:
            parameters["rope_theta"] = float(rope_theta)
        config.rope_parameters = parameters
    else:
        parameters = {
            "type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": original,
        }
        if rope_theta is not None:
            parameters["rope_theta"] = float(rope_theta)
        config.rope_scaling = parameters
    return config


def load_model_config(
        model_name, *, yarn_factor=1.0,
        original_max_position_embeddings=QWEN_NATIVE_CONTEXT, **kwargs):
    """Load a Hugging Face config and apply the requested YaRN override."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, **kwargs)
    return apply_yarn(
        config, yarn_factor=yarn_factor,
        original_max_position_embeddings=original_max_position_embeddings)


def configure_tokenizer(
        tokenizer, *, yarn_factor=1.0,
        original_max_position_embeddings=QWEN_NATIVE_CONTEXT):
    """Raise tokenizer warning/truncation metadata to the configured limit."""
    tokenizer.model_max_length = context_limit(
        yarn_factor=yarn_factor,
        original_max_position_embeddings=original_max_position_embeddings)
    return tokenizer


def yarn_metadata(
        *, yarn_factor=1.0,
        original_max_position_embeddings=QWEN_NATIVE_CONTEXT):
    """Serializable description of the rotary configuration."""
    factor = float(yarn_factor)
    original = int(original_max_position_embeddings)
    return {
        "enabled": factor > 1.0,
        "rope_type": "yarn" if factor > 1.0 else "default",
        "factor": factor,
        "original_max_position_embeddings": original,
        "max_position_embeddings": context_limit(
            yarn_factor=factor,
            original_max_position_embeddings=original),
        "static": factor > 1.0,
    }

