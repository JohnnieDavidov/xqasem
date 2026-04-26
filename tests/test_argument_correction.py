from xqasem.argument_correction import validate_and_fix_answer


def test_exact_answer_is_kept() -> None:
    result = validate_and_fix_answer(
        "The committee postponed the report.",
        "the report",
    )

    assert result["answer_found"] is True
    assert result["answer_status"] == "exact"
    assert result["fixed_answer"] == "the report"


def test_fuzzy_answer_is_repaired_to_sentence_span() -> None:
    result = validate_and_fix_answer(
        "The developers fixed unexpected service failures.",
        "unexpected services failure",
    )

    assert result["answer_found"] is True
    assert result["fixed_answer"] == "unexpected service failures"
