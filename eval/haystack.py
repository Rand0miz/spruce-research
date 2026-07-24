def build_haystack(tokenizer, target_tokens, needle, depth, filler_unit):
    """
    Build a prompt of ~target_tokens length with `needle` inserted at `depth`.

    tokenizer     : used ONLY to measure length in tokens, not to run a model.
    target_tokens : desired prompt length (2000 now, 128000 later — same code).
    needle        : the exact sentence to hide and later retrieve.
    depth         : float 0.0-1.0. 0.0 = needle at start, 0.5 = middle, 1.0 = end.
    filler_unit   : one sentence of neutral filler, repeated to reach length.

    Returns: (prompt_string, needle_string)
    """
    assert 0.0 <= depth <= 1.0, f"depth must be in [0,1], got {depth}"

    # How many filler units fit before we hit the token budget.
    unit_len = len(tokenizer(filler_unit)["input_ids"])
    needle_len = len(tokenizer(needle)["input_ids"])
    room = target_tokens - needle_len
    n_units = max(0, room // unit_len)

    # Split filler into a before-chunk and an after-chunk, based on depth.
    n_before = int(n_units * depth)
    n_after = n_units - n_before

    before = filler_unit * n_before
    after = filler_unit * n_after
    prompt = before + needle + " " + after

    return prompt, needle


def build_haystack_calibrated(
        tokenizer, target_tokens, needle, depth, filler_unit, suffix=""):
    """Build the closest prompt not exceeding an exact total token budget.

    Repeated BPE text is usually shorter than ``n * tokenize(unit)`` because
    merges cross repetition boundaries. Binary-search the actual tokenized
    prompt instead of relying on that estimate. ``suffix`` is included while
    measuring (normally the retrieval question) but is not returned as part
    of the haystack.

    Returns ``(prompt, needle, filler_units)``. Saving ``filler_units`` makes
    replay exact without rerunning the calibration search.
    """
    if not 0.0 <= depth <= 1.0:
        raise ValueError(f"depth must be in [0,1], got {depth}")
    if target_tokens < 1:
        raise ValueError("target_tokens must be >= 1")

    def prompt_for(n_units):
        n_before = int(n_units * depth)
        return (
            filler_unit * n_before
            + needle
            + " "
            + filler_unit * (n_units - n_before)
        )

    def measured(n_units):
        prompt = prompt_for(n_units)
        original_max = getattr(tokenizer, "model_max_length", None)
        if original_max is not None:
            # Calibration intentionally probes above the final target while
            # bracketing the binary search. Suppress misleading model-length
            # warnings for this tokenizer-only probe, then restore the limit.
            tokenizer.model_max_length = max(
                int(original_max), int(target_tokens) * 4)
        try:
            length = len(tokenizer(prompt + suffix)["input_ids"])
        finally:
            if original_max is not None:
                tokenizer.model_max_length = original_max
        return length, prompt

    unit_len = max(1, len(tokenizer(filler_unit)["input_ids"]))
    high = max(1, target_tokens // unit_len)
    high_len, _ = measured(high)
    while high_len <= target_tokens:
        high *= 2
        high_len, _ = measured(high)

    low = 0
    while low + 1 < high:
        middle = (low + high) // 2
        middle_len, _ = measured(middle)
        if middle_len <= target_tokens:
            low = middle
        else:
            high = middle

    # Depth rounding can introduce tiny boundary irregularities. Inspect a
    # local window and choose the longest valid prompt deterministically.
    best = None
    for n_units in range(max(0, low - 4), high + 5):
        length, prompt = measured(n_units)
        if length <= target_tokens and (
                best is None or length > best[0]
                or (length == best[0] and n_units < best[1])):
            best = (length, n_units, prompt)
    if best is None:
        raise ValueError(
            "needle and suffix alone exceed the requested token budget")
    return best[2], needle, best[1]
