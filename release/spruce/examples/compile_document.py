from pathlib import Path

from spruce_attn import SpruceCompiler


MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def main():
    compiler = SpruceCompiler.from_pretrained(MODEL)
    document = Path("book.txt").read_text(encoding="utf-8")
    result = compiler.compile(
        document,
        "What is the main conclusion supported by the document?",
    )
    Path("evidence.txt").write_text(result.content, encoding="utf-8")
    print(result.metadata())


if __name__ == "__main__":
    main()
