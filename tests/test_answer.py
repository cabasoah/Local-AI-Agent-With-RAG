"""Checks for answer.py. Run: python -m pytest tests -q (no Ollama needed)."""

from rag.answer import build_prompt, parse_answer

CHUNKS = [{"id": f"a.txt::{i}", "text": f"passage {i} text"} for i in range(3)]


def test_prompt_labels_passages_and_holds_question():
    p = build_prompt("Is there a non-compete?", CHUNKS)
    assert "[C1]\npassage 0 text" in p and "[C3]\npassage 2 text" in p
    assert p.rstrip().endswith("Question: Is there a non-compete?\nAnswer:")


def test_citations_map_back_to_chunk_ids_in_order_without_repeats():
    r = parse_answer("Yes [C2]. Also this [C1][C2].", CHUNKS)
    assert r["cited_labels"] == [2, 1]
    assert r["cited_ids"] == ["a.txt::1", "a.txt::0"]
    assert not r["not_found"]


def test_labels_that_were_never_given_are_flagged():
    r = parse_answer("Yes [C1] and [C9].", CHUNKS)
    assert r["cited_labels"] == [1] and r["invalid_labels"] == [9]


def test_not_found_detection():
    assert parse_answer("NOT FOUND", CHUNKS)["not_found"]
    assert parse_answer("Not found.", CHUNKS)["not_found"]
    # Cites passages, so it's an answer even though it says "not found".
    assert not parse_answer("The term is not found in [C1], but [C2] says X.", CHUNKS)["not_found"]
