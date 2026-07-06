from eval.haystack import build_haystack
from eval.score import score_retrival

def run_needle_test(generate, tokenizer, target_tokens, depth,
                    needle=None, filler_unit=None, question=None):
    """
    Run one needle-in-haystack test and return the score.

    generate : a callable  generate(prompt_str) -> answer_str.
               The harness knows NOTHING else about the model. Dense or
               sparse, both are passed in the same way.
    tokenizer: used only to measure/insert by token length.

    Returns: dict with target_tokens, depth, and the score fields.
    """
    if needle is None:
        needle = "The secret access code is heron-4417."
    if filler_unit is None:
        filler_unit = "The quarterly logistics report noted no unusual activity."
    if question is None:
        question = "\n\nWhat is the secret access code? Answer with the exact sentence."

    prompt, true_needle = build_haystack(
        tokenizer, target_tokens, needle, depth, filler_unit
    )

    answer = generate(prompt + question)
    result = score_retrival(answer, true_needle)

    return {
        "target_tokens": target_tokens,
        "depth": depth,
        **result
    }
