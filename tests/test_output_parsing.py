from xqasem.argument_detection import XQasemArgumentParser
from xqasem.presets import DEFAULT_MODELS, DEFAULT_SPACY_MODELS


def make_parser(language: str = "fr") -> XQasemArgumentParser:
    parser = XQasemArgumentParser.__new__(XQasemArgumentParser)
    parser.language = language
    return parser


def test_parse_arrow_separated_output() -> None:
    parser = make_parser("fr")

    pairs = parser._parse_model_output_to_qa_pairs(
        "Qu'est-ce que les développeurs ont expliqué ? -> 'pourquoi la mise à jour avait provoqué des pannes'"
    )

    assert pairs == [
        {
            "question": "Qu'est-ce que les développeurs ont expliqué ?",
            "answer": "pourquoi la mise à jour avait provoqué des pannes",
        }
    ]


def test_parse_hebrew_pipe_separated_output() -> None:
    parser = make_parser("he")

    pairs = parser._parse_model_output_to_qa_pairs(
        "מי הסביר? -> המפתחים | מה גרם? -> העדכון"
    )

    assert pairs == [
        {"question": "מי הסביר?", "answer": "המפתחים"},
        {"question": "מה גרם?", "answer": "העדכון"},
    ]


def test_language_presets_are_available() -> None:
    assert set(DEFAULT_MODELS) == {"fr", "ru", "he"}
    assert set(DEFAULT_SPACY_MODELS) == {"fr", "ru", "he"}
