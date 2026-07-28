"""Stable public API for training-free SPRUCE context compilation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from .compiler import (
    EvidencePacket,
    compile_evidence_packet_from_layout,
    locate_prompt_layout,
)
from .selector import (
    PreQwenSelection,
    build_pre_qwen_index,
    question_feature_weights,
    select_pre_qwen_blocks,
)


SOURCE_SYSTEM_PROMPT = (
    "Answer the user's final question using only the supplied document. "
    "Follow the requested answer format exactly. Do not continue, summarize, "
    "or rewrite the document."
)


@dataclass(frozen=True)
class CompilerConfig:
    """Frozen beam-16 configuration validated in the paper sweep."""

    block_size: int = 64
    candidate_blocks: int = 4
    block_radius: int = 1
    boundary: str = "paragraph"
    beam: int = 16
    feature_dim: int = 512
    unigram_fraction: float = 0.5
    idf_power: float = 2.0
    radix: int = 2

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.candidate_blocks < 1:
            raise ValueError("candidate_blocks must be >= 1")
        if self.block_radius < 0:
            raise ValueError("block_radius must be >= 0")
        if self.boundary not in {"block", "paragraph"}:
            raise ValueError("boundary must be 'block' or 'paragraph'")
        if self.beam < self.candidate_blocks:
            raise ValueError("beam must be >= candidate_blocks")
        if self.feature_dim < 16:
            raise ValueError("feature_dim must be >= 16")
        if not 0.1 <= self.unigram_fraction <= 0.9:
            raise ValueError("unigram_fraction must be in [0.1, 0.9]")
        if self.idf_power <= 0:
            raise ValueError("idf_power must be > 0")
        if self.radix < 2:
            raise ValueError("radix must be >= 2")


@dataclass(frozen=True)
class CompilationResult:
    """Compiled packet plus inexpensive selection diagnostics."""

    packet: EvidencePacket
    selected_scores: tuple[float, ...]
    visited_nodes: int
    levels_descended: int
    indexed_tokens: int

    @property
    def prompt(self) -> str:
        return self.packet.prompt

    @property
    def content(self) -> str:
        return self.packet.content

    def metadata(self) -> dict[str, Any]:
        return {
            "compiler": {
                "selected_scores": list(self.selected_scores),
                "visited_nodes": self.visited_nodes,
                "levels_descended": self.levels_descended,
                "indexed_tokens": self.indexed_tokens,
            },
            **self.packet.metadata(),
        }


class SpruceCompiler:
    """Compile a long document into a short exact-text evidence packet.

    The selector uses tokenizer IDs only. It never reads model hidden states,
    Q/K features, attention matrices, or a trained selector checkpoint.
    """

    def __init__(
            self, tokenizer, config: CompilerConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or CompilerConfig()

    @classmethod
    def from_pretrained(
            cls, model_or_tokenizer: str, *,
            config: CompilerConfig | None = None,
            **tokenizer_kwargs) -> "SpruceCompiler":
        """Load the checkpoint tokenizer without loading model weights."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_or_tokenizer, **tokenizer_kwargs)
        return cls(tokenizer, config=config)

    def format_source_prompt(
            self, document: str, question: str) -> tuple[str, str, str]:
        """Return full chat prompt, user content, and exact question suffix."""
        if not isinstance(document, str) or not document.strip():
            raise ValueError("document must be a non-empty string")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        clean_document = document.rstrip()
        clean_question = question.strip()
        user_prompt = f"{clean_document}\n\n{clean_question}"
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise TypeError("tokenizer does not provide apply_chat_template")
        full_prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SOURCE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return full_prompt, user_prompt, clean_question

    def compile(
            self, document: str, question: str) -> CompilationResult:
        """Select and compile exact evidence from a plain document."""
        full_prompt, user_prompt, clean_question = (
            self.format_source_prompt(document, question))
        return self.compile_prompt(
            full_prompt, user_prompt, clean_question)

    def compile_prompt(
            self, full_prompt: str, user_prompt_text: str,
            question: str) -> CompilationResult:
        """Compile an already formatted chat prompt.

        ``user_prompt_text`` must occur verbatim in ``full_prompt`` and end
        with the exact ``question`` string.
        """
        cfg = self.config
        layout = locate_prompt_layout(
            self.tokenizer, full_prompt, user_prompt_text, question)
        index = build_pre_qwen_index(
            layout, cfg.block_size,
            feature_dim=cfg.feature_dim,
            unigram_fraction=cfg.unigram_fraction,
            radix=cfg.radix,
        )
        weights = question_feature_weights(
            self.tokenizer, question, index,
            idf_power=cfg.idf_power)
        selection: PreQwenSelection = select_pre_qwen_blocks(
            index, weights,
            top_m=cfg.candidate_blocks,
            beam=cfg.beam,
            radix=cfg.radix,
        )
        packet = compile_evidence_packet_from_layout(
            self.tokenizer, full_prompt, layout, question,
            selection.blocks, cfg.block_size,
            block_radius=cfg.block_radius,
            boundary=cfg.boundary,
        )
        return CompilationResult(
            packet=packet,
            selected_scores=selection.scores,
            visited_nodes=selection.visited_nodes,
            levels_descended=selection.levels_descended,
            indexed_tokens=index.indexed_tokens,
        )

    def generate(
            self, model, compilation: CompilationResult, *,
            max_new_tokens: int = 64) -> str:
        """Run greedy dense generation over a compiled packet."""
        if int(max_new_tokens) < 1:
            raise ValueError("max_new_tokens must be >= 1")
        encoded = self.tokenizer(
            compilation.prompt, return_tensors="pt")
        try:
            device = next(model.parameters()).device
        except StopIteration as error:
            raise ValueError("model has no parameters") from error
        encoded = {
            name: tensor.to(device)
            for name, tensor in encoded.items()
        }
        prompt_tokens = int(encoded["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                use_cache=True,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id),
            )
        new_tokens = generated[0, prompt_tokens:]
        return self.tokenizer.decode(
            new_tokens, skip_special_tokens=True).strip()

    def answer(
            self, model, document: str, question: str, *,
            max_new_tokens: int = 64) -> tuple[str, CompilationResult]:
        """Compile a document and return the answer plus provenance."""
        result = self.compile(document, question)
        return (
            self.generate(
                model, result, max_new_tokens=max_new_tokens),
            result,
        )

    def configuration(self) -> dict[str, Any]:
        return asdict(self.config)
