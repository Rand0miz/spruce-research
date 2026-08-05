import hashlib
import json

import pytest
import torch

from benchmarks.ruler_cached_factorial import DEFAULT_TASKS
from benchmarks.ruler_paper_accuracy import (
    TASK_MAX_NEW_TOKENS,
    append_official_answer_prefix,
    file_sha256,
    generate_one,
    macro_block,
    official_prompt_parts,
    verify_dataset_manifest,
)


def test_generate_one_records_ttft_decode_and_throughput():
    class Tokenizer:
        pad_token_id = 0

        def __call__(self, prompt, return_tensors=None):
            assert return_tensors == "pt"
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, token_ids, skip_special_tokens=True):
            return "answer"

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def generate(self, input_ids, logits_processor, **kwargs):
            scores = torch.zeros((1, 4))
            for processor in logits_processor:
                scores = processor(input_ids, scores)
            return torch.tensor([[1, 2, 3]])

    result = generate_one(Model(), Tokenizer(), "prompt", 1)
    assert result["generated_tokens"] == 1
    assert result["ttft_seconds"] >= 0
    assert result["decode_seconds"] >= 0
    assert result["decode_tokens_per_second"] == 0.0


def test_official_prompt_appends_answer_prefix_and_preserves_niah_suffix():
    sample = {
        "input": (
            "Some special magic numbers are hidden.\nEvidence line."
            "\nWhat are all the special magic numbers for Alice?"),
        "answer_prefix": " The special magic numbers for Alice are",
    }
    full, query, suffix = official_prompt_parts(sample, "niah_single_1")
    assert full == sample["input"] + sample["answer_prefix"]
    assert query == "\nWhat are all the special magic numbers for Alice?"
    assert suffix == query + sample["answer_prefix"]
    assert full.endswith(suffix)
    assert append_official_answer_prefix("<assistant>\n", sample) == (
        "<assistant>\n The special magic numbers for Alice are")


@pytest.mark.parametrize("task", ["vt", "cwe", "fwe"])
def test_single_line_aggregation_context_is_not_part_of_query(task):
    context = "1. alpha 2. beta 3. alpha " * 200
    sample = {
        "input": "Task instruction.\n" + context + "\nQuestion: Find the answer.",
        "answer_prefix": " Answer:",
    }
    full, query, suffix = official_prompt_parts(sample, task)
    assert query == "\nQuestion: Find the answer."
    assert context not in query
    assert suffix == query + " Answer:"
    assert full.endswith(suffix)


def test_qa_layout_keeps_official_reader_instruction_but_selector_uses_question():
    repeated = (
        "Answer the question based on the given documents. Only give me the "
        "answer and do not output any other words.")
    sample = {
        "input": (
            repeated + "\n\nThe following are given documents.\n\nDocument A."
            "\n\n" + repeated + "\n\nQuestion: Who wrote it?"),
        "answer_prefix": " Answer:",
    }
    full, query, suffix = official_prompt_parts(sample, "qa_1")
    assert query == "\nQuestion: Who wrote it?"
    assert suffix == "\n\n" + repeated + "\n" + query + " Answer:"
    assert full.endswith(suffix)


def test_changed_or_unknown_templates_fail_closed():
    sample = {"input": "Document only.", "answer_prefix": " Answer:"}
    with pytest.raises(ValueError, match="missing its query marker"):
        official_prompt_parts(sample, "cwe")
    with pytest.raises(ValueError, match="unsupported RULER task"):
        official_prompt_parts(sample, "new_task")


def test_task_specific_budgets_cover_the_official_13_task_grid():
    assert set(TASK_MAX_NEW_TOKENS) == set(DEFAULT_TASKS)
    assert TASK_MAX_NEW_TOKENS["niah_single_1"] == 128
    assert TASK_MAX_NEW_TOKENS["vt"] == 30
    assert TASK_MAX_NEW_TOKENS["cwe"] == 120
    assert TASK_MAX_NEW_TOKENS["fwe"] == 50
    assert TASK_MAX_NEW_TOKENS["qa_1"] == 32


def test_manifest_verification_freezes_grid_and_file_hashes(tmp_path):
    tasks = ["qa_1"]
    lengths = [4096]
    data_file = tmp_path / "4096" / "qa_1" / "test.jsonl"
    data_file.parent.mkdir(parents=True)
    data_file.write_text('{"input":"x"}\n', encoding="utf-8")
    relative = "4096/qa_1/test.jsonl"
    manifest = {
        "kind": "nvidia_ruler_dataset_manifest_v1",
        "random_seed": 314159,
        "subset": "test",
        "num_samples": 1,
        "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "tasks": tasks,
        "lengths": lengths,
        "files": {
            relative: {
                "rows": 1,
                "bytes": data_file.stat().st_size,
                "sha256": file_sha256(data_file),
            }
        },
    }
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    observed, digest = verify_dataset_manifest(
        tmp_path, manifest_path, tasks, lengths, "test", 1)
    canonical = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")).encode()
    assert observed == manifest
    assert digest == hashlib.sha256(canonical).hexdigest().upper()

    data_file.write_text('{"input":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_dataset_manifest(
            tmp_path, manifest_path, tasks, lengths, "test", 1)


def test_macro_block_weights_tasks_equally(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.ruler_paper_accuracy.stratified_bootstrap",
        lambda rows, value_fn, repeats=10000: [0.0, 1.0],
    )
    dense_metrics = {
        "input_tokens": 10, "fully_charged_request_seconds": 2.0,
        "fully_charged_ttft_seconds": 1.0, "generated_tokens": 3,
        "decode_seconds": 1.0, "peak_memory_allocated_gb": 2.0,
    }
    spruce_metrics = {
        "input_tokens": 5, "compression_fraction": 0.5,
        "evidence_score": 1.0, "fully_charged_request_seconds": 1.0,
        "request_seconds": 0.8, "fully_charged_ttft_seconds": 0.5,
        "compiler_seconds": 0.2, "layout_seconds": 0.05,
        "index_seconds": 0.05, "selection_seconds": 0.05,
        "compile_seconds": 0.05, "generated_tokens": 3,
        "decode_seconds": 0.5, "peak_memory_allocated_gb": 1.0,
    }
    rows = [
        {
            "task": "a", "dense": dict(dense_metrics, score=1.0),
            "spruce": dict(spruce_metrics, score=1.0),
            "score_delta_spruce_minus_dense": 0.0,
        },
        {
            "task": "a", "dense": dict(dense_metrics, score=1.0),
            "spruce": dict(spruce_metrics, score=1.0),
            "score_delta_spruce_minus_dense": 0.0,
        },
        {
            "task": "b", "dense": dict(dense_metrics, score=0.0),
            "spruce": dict(spruce_metrics, score=1.0),
            "score_delta_spruce_minus_dense": 1.0,
        },
    ]
    block = macro_block(rows)
    assert block["score_dense_macro"] == pytest.approx(0.5)
    assert block["score_spruce_macro"] == pytest.approx(1.0)
    assert block["delta_macro"] == pytest.approx(0.5)
    assert block["fully_charged_speedup_ratio_of_sums"] == pytest.approx(2.0)
    assert block["fully_charged_ttft_speedup_ratio_of_sums"] == pytest.approx(2.0)
