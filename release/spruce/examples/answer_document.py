from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from spruce_attn import SpruceCompiler


MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compiler = SpruceCompiler.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype="auto",
        attn_implementation="sdpa",
    ).to(device).eval()
    document = Path("book.txt").read_text(encoding="utf-8")
    answer, result = compiler.answer(
        model,
        document,
        "What is the main conclusion supported by the document?",
    )
    print(answer)
    print(result.metadata())


if __name__ == "__main__":
    main()
