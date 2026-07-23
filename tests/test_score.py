from eval.score import score_retrival


def test_score_normalizes_possessives_and_punctuation():
    result = score_retrival(
        " Bob's favorite color is green.",
        "bobs favorite color is green.",
    )
    assert result["exact"] is True
    assert result["fuzzy"] == 1.0


def test_score_still_rejects_wrong_evidence():
    result = score_retrival(
        "Bob's favorite color is blue.",
        "bobs favorite color is green.",
    )
    assert result["exact"] is False
    assert result["fuzzy"] < 1.0
