import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional, Dict, Any, List


def normalize_text(text: str) -> str:
    """
    Light normalization for matching answers against the sentence.
    Keeps content but smooths formatting differences.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    # Normalize apostrophes
    text = text.replace("’", "'").replace("`", "'")

    # Normalize spaces around punctuation
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # Lowercase for robust comparison
    text = text.lower().strip()

    return text


def answer_in_sentence_exact(sentence: str, answer: str) -> bool:
    """Return True when the answer appears as an exact substring."""
    if not sentence or not answer:
        return False
    return answer in sentence


def answer_in_sentence_normalized(sentence: str, answer: str) -> bool:
    """Return True when the normalized answer appears in the normalized sentence."""
    if not sentence or not answer:
        return False
    return normalize_text(answer) in normalize_text(sentence)


def extract_candidate_spans(sentence: str, min_words: int = 1, max_words: int = 12) -> List[str]:
    """
    Generate contiguous candidate spans from the sentence.
    This is used only for fuzzy repair.
    """
    tokens = sentence.split()
    spans = []

    for i in range(len(tokens)):
        for j in range(i + min_words, min(len(tokens), i + max_words) + 1):
            span = " ".join(tokens[i:j]).strip()
            if span:
                spans.append(span)

    return spans


def similarity(a: str, b: str) -> float:
    """Compute normalized string similarity between two text spans."""
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def find_best_sentence_span(
    sentence: str,
    answer: str,
    min_score: float = 0.82,
    min_words: int = 1,
    max_words: int = 12
) -> Optional[str]:
    """
    Try to repair an answer by finding the most similar contiguous span in the sentence.
    Only returns a span if similarity passes min_score.
    """
    if not sentence or not answer:
        return None

    candidates = extract_candidate_spans(sentence, min_words=min_words, max_words=max_words)

    best_span = None
    best_score = -1.0

    for cand in candidates:
        score = similarity(cand, answer)
        if score > best_score:
            best_score = score
            best_span = cand

    if best_score >= min_score:
        return best_span

    return None


def validate_and_fix_answer(sentence: str, answer: str) -> Dict[str, Any]:
    """
    Validate whether answer is a real sentence span.
    If not, try normalized match and then fuzzy repair.
    """
    result = {
        "original_answer": answer,
        "fixed_answer": answer,
        "answer_in_sentence_exact": False,
        "answer_in_sentence_normalized": False,
        "answer_status": "not_found",
        "answer_spans": [],
        "answer_found": False,
    }

    if not sentence or not answer:
        return result

    # Exact match
    if answer_in_sentence_exact(sentence, answer):
        result["answer_in_sentence_exact"] = True
        result["answer_in_sentence_normalized"] = True
        result["answer_status"] = "exact"
        result["answer_found"] = True
        return result

    # Normalized match
    if answer_in_sentence_normalized(sentence, answer):
        result["answer_in_sentence_normalized"] = True
        result["answer_status"] = "normalized_match"
        result["answer_found"] = True

        # Try to recover the actual surface span from the sentence
        repaired = find_best_sentence_span(sentence, answer, min_score=0.88)
        if repaired:
            repaired = repaired.rstrip('.,!?;:')
            result["fixed_answer"] = repaired

        return result

    # Fuzzy repair
    repaired = find_best_sentence_span(sentence, answer, min_score=0.82)
    if repaired:
        repaired = repaired.rstrip('.,!?;:')
        result["fixed_answer"] = repaired
        result["answer_status"] = "fuzzy_fixed"
        result["answer_in_sentence_normalized"] = True
        result["answer_found"] = True
        return result

    return result
