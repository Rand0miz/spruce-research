"""Command-line interface for SPRUCE context compilation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .api import CompilerConfig, SpruceCompiler


def _configuration(args) -> CompilerConfig:
    return CompilerConfig(
        block_size=args.block_size,
        candidate_blocks=args.candidate_blocks,
        block_radius=args.block_radius,
        boundary=args.boundary,
        beam=args.beam,
        feature_dim=args.feature_dim,
        unigram_fraction=args.unigram_fraction,
        idf_power=args.idf_power,
        radix=args.radix,
    )


def _add_compiler_arguments(parser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--document", required=True, type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--candidate-blocks", type=int, default=4)
    parser.add_argument("--block-radius", type=int, default=1)
    parser.add_argument(
        "--boundary", choices=("block", "paragraph"),
        default="paragraph")
    parser.add_argument("--beam", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--unigram-fraction", type=float, default=0.5)
    parser.add_argument("--idf-power", type=float, default=2.0)
    parser.add_argument("--radix", type=int, default=2)
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="pass trust_remote_code=True when loading the tokenizer/model")


def _load_compiler(args) -> SpruceCompiler:
    if not args.document.is_file():
        raise SystemExit(f"document not found: {args.document}")
    return SpruceCompiler.from_pretrained(
        args.model,
        config=_configuration(args),
        trust_remote_code=args.trust_remote_code,
    )


def _write_text(path: Path | None, text: str) -> None:
    if path is None:
        print(text)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(path)


def _compile(args) -> int:
    compiler = _load_compiler(args)
    document = args.document.read_text(encoding="utf-8")
    result = compiler.compile(document, args.question)
    output = result.prompt if args.output_format == "prompt" else result.content
    _write_text(args.output, output)
    if args.metadata is not None:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(
            json.dumps(result.metadata(), indent=2),
            encoding="utf-8")
        print(args.metadata)
    return 0


def _answer(args) -> int:
    import torch
    from transformers import AutoModelForCausalLM

    compiler = _load_compiler(args)
    document = args.document.read_text(encoding="utf-8")
    load_kwargs = {
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
        "attn_implementation": "sdpa",
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model, **load_kwargs).eval()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    answer, result = compiler.answer(
        model, document, args.question,
        max_new_tokens=args.max_new_tokens)
    print(answer)
    if args.metadata is not None:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(
            json.dumps(result.metadata(), indent=2),
            encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spruce",
        description=(
            "Training-free hierarchical exact-text context compilation."))
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    compile_parser = commands.add_parser(
        "compile", help="compile a document into an evidence packet")
    _add_compiler_arguments(compile_parser)
    compile_parser.add_argument("--output", type=Path)
    compile_parser.add_argument("--metadata", type=Path)
    compile_parser.add_argument(
        "--output-format", choices=("content", "prompt"),
        default="content")
    compile_parser.set_defaults(handler=_compile)

    answer_parser = commands.add_parser(
        "answer", help="compile a document and run the reader model")
    _add_compiler_arguments(answer_parser)
    answer_parser.add_argument("--max-new-tokens", type=int, default=64)
    answer_parser.add_argument(
        "--device", default="auto",
        help="auto, cpu, cuda, or an explicit torch device")
    answer_parser.add_argument("--metadata", type=Path)
    answer_parser.set_defaults(handler=_answer)

    info_parser = commands.add_parser(
        "info", help="show package and frozen default configuration")
    info_parser.set_defaults(handler=lambda _args: _info())
    return parser


def _info() -> int:
    print(json.dumps({
        "package": "spruce-attn",
        "version": __version__,
        "selector": "training-free tokenizer-level lexical hierarchy",
        "defaults": CompilerConfig().__dict__,
        "verified_reader": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    }, indent=2))
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
