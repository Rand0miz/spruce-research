import re


def _normalize(text):
    text = text.strip().lower().replace("’", "'")
    # Prompt banks historically stored some possessives without apostrophes
    # ("bobs") while model answers use ordinary English ("Bob's").
    text = re.sub(r"(?<=\w)'s\b", "s", text)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def score_retrieval(model_answer, needle, reference_answers=None):
    """
    Compare the model's answer to the true needle or short answer aliases.

    ``reference_answers`` is used by natural-document QA so a correct concise
    answer or harmless formatting variant is not penalized for failing to copy
    the complete evidence sentence.

    Returns a dict: exact (bool) and fuzzy (float 0-1, word overlap).
    """
    displayed_needle = needle.strip().lower()
    references = (
        list(reference_answers)
        if reference_answers
        else [needle]
    )
    if any(not isinstance(reference, str) or not reference.strip()
           for reference in references):
        raise ValueError("reference answers must be non-empty strings")
    ans = _normalize(model_answer)
    normalized_references = [_normalize(reference) for reference in references]
    exact = any(reference in ans for reference in normalized_references)

    # Fuzzy: maximum reference-word recall over accepted aliases.
    ans_words = set(ans.split())
    overlap = max(
        len(set(reference.split()) & ans_words)
        / max(1, len(set(reference.split())))
        for reference in normalized_references
    )

    return {"exact": exact, "fuzzy": overlap,
            "answer": model_answer, "needle": displayed_needle,
            "reference_answers": references}


def score_concise_retrieval(model_answer, reference_answers):
    """Require the whole generated answer to equal one accepted short alias."""
    if not reference_answers:
        raise ValueError("reference_answers must be non-empty")
    answer = _normalize(model_answer)
    references = [_normalize(reference) for reference in reference_answers]
    exact = any(answer == reference for reference in references)
    return {
        "exact": exact,
        "answer": model_answer,
        "reference_answers": list(reference_answers),
    }


def score_retrival(model_awnser, needle, reference_answers=None):
    """Backward-compatible alias for the original misspelled public helper."""
    return score_retrieval(
        model_awnser, needle, reference_answers=reference_answers)
