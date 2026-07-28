# Release checklist

## Before the first public push

- Confirm the repository name and package URLs in `pyproject.toml`.
- Replace “SPRUCE contributors” with the preferred copyright attribution.
- Review Apache-2.0 with the project owner.
- Confirm no `.env`, tokens, private prompts, model weights, or checkpoints.
- Run `python -m pytest -q`.
- Run `python -m build`.
- Run `python -m twine check dist/*`.
- Install the wheel in a clean environment.
- Run `spruce info`.
- Run the Colab quickstart on a fresh GPU runtime.

## Before PyPI

- Create the `spruce-attn` project or pending trusted publisher on PyPI.
- Configure the GitHub `pypi` environment.
- Publish `0.1.0rc1` to TestPyPI.
- Install and smoke-test the TestPyPI wheel.
- Tag the final release `v0.1.0`.

## Claim review

- Keep Qwen2.5-Coder-1.5B as the only verified reader.
- Label other Qwen sizes experimental until frozen cross-size validation.
- Quote hardware and timing boundaries with every latency number.
- Do not treat the 288 rows as 288 independent semantic tasks.
- Do not use reserved-memory results as compiler-only memory.
