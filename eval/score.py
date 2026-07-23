import re


def _normalize(text):
    text = text.strip().lower().replace("’", "'")
    # Prompt banks historically stored some possessives without apostrophes
    # ("bobs") while model answers use ordinary English ("Bob's").
    text = re.sub(r"(?<=\w)'s\b", "s", text)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def score_retrival(model_awnser, needle):
    """
    Compare the model's answer to the true needle.
    Returns a dict: exact (bool) and fuzzy (float 0-1, word overlap).

    No model logic here — pure string comparison. This is why the harness
    stays model-agnostic.
    """
    displayed_needle = needle.strip().lower()
    ans = _normalize(model_awnser)
    needle = _normalize(needle)

    exact = needle in ans

    # Fuzzy: word overlap
    ndl_words = set(needle.split())
    ans_words = set(ans.split())
    overlap = len(ndl_words & ans_words) / max(1, len(ndl_words))

    return {"exact": exact, "fuzzy": overlap,
            "answer": model_awnser, "needle": displayed_needle}
