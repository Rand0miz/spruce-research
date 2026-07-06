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