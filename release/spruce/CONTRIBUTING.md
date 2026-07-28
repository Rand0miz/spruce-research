# Contributing

Thank you for helping improve SPRUCE.

1. Open an issue before making a large API or algorithm change.
2. Create a focused branch and keep unrelated research artifacts out of it.
3. Install development dependencies with `pip install -e ".[dev]"`.
4. Run `pytest -q` before opening a pull request.
5. Add tests for every behavior change.
6. Record new measured results with model, hardware, length, precision,
   prompt distribution, and timing boundary.

Changes to the frozen beam-16 defaults must be evaluated on new development
data. Do not tune against the opened paper-result bank.

By contributing, you agree that your contribution is licensed under the
Apache License 2.0.
