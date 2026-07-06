def score_retrival(model_awnser, needle):
    """
    Compare the model's answer to the true needle.
    Returns a dict: exact (bool) and fuzzy (float 0-1, word overlap).

    No model logic here — pure string comparison. This is why the harness
    stays model-agnostic.
    """
    ans = model_awnser.strip().lower()
    needle = needle.strip().lower()

    exact = needle in ans

    # Fuzzy: word overlap
    ndl_words = set(needle.split())
    ans_words = set(ans.split())
    overlap = len(ndl_words & ans_words) / max(1, len(ndl_words))

    return {"exact": exact, "fuzzy": overlap}