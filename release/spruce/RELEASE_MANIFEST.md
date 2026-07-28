# SPRUCE v0.1.0 release manifest

This archive is the clean source candidate for the first public
`spruce-attn` release.

Included:

- the training-free lexical hierarchy and evidence compiler;
- the frozen beam-16 public API and command-line interface;
- unit tests and two minimal examples;
- a browser-runnable Colab quickstart;
- package, contribution, security, and release documentation;
- the sealed natural prompt-bank specification;
- the immutable 16K–128K paper result report, tables, and figures.

Deliberately excluded:

- model weights, selector checkpoints, and optimizer state;
- local `.env` files, credentials, caches, and editor state;
- exploratory notebooks and obsolete experiment outputs;
- the experimental learned selector and sparse-attention kernel, which are not
  part of the verified v0.1.0 inference path.

Evidence provenance:

- report: `benchmarks/paper_results/natural_yarn_beam16/combined_report.json`
- report SHA-256:
  `AE1DEB4939FA1D06111D663D80355BD4423D3627AC611F854D9EB827FC5FB35A`
- sealed prompt bank:
  `benchmarks/prompt_banks/natural_paper_untouched.json`
- verified compiler configuration: beam 16, four candidate blocks, one-block
  expansion radius, paragraph-boundary expansion, 512 lexical features,
  radix 2.

Before public publication, the repository owner must confirm the project name,
copyright holder, package namespace, and Apache-2.0 licensing decision.
