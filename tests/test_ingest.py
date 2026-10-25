"""Checks for file reading and chunking. Run: python -m pytest tests -q"""
import random

import pytest

from rag.chunking import split_with_offsets
from rag.ingest import chunk_document, read_document


def test_read_document_matches_benchmark_newlines(tmp_path):
    # The benchmark computes offsets on text read with plain open(), which
    # turns "\r\n" into "\n". We must read it the same way.
    p = tmp_path / "doc.txt"
    p.write_bytes(b"line one\r\nline two\r\n")
    assert read_document(p) == "line one\nline two\n"


def test_repeated_text_gets_its_own_position():
    # The case that broke str.find(): the same "3" appears at 6, 19 and 22.
    text = "____\n\n3\n\n____\n\n12\n\n3\n\n3"
    chunks = chunk_document("x.txt", text, size=8, overlap=1)
    for c in chunks:
        assert text[c["start"]:c["end"]] == c["text"]
    covered = {i for c in chunks for i in range(c["start"], c["end"])}
    assert {6, 19, 22} <= covered


def test_paragraphs_are_kept_whole_when_they_fit():
    text = "First paragraph here.\n\nSecond paragraph here."
    assert [text[s:e] for s, e in split_with_offsets(text, 25, 0)] == [
        "First paragraph here.", "Second paragraph here."]


def test_overlap_repeats_the_end_of_the_previous_chunk():
    text = " ".join(f"w{i}" for i in range(40))
    spans = split_with_offsets(text, 30, 10)
    assert any(s < prev_e for (_, prev_e), (s, _) in zip(spans, spans[1:]))


def _check_guarantees(text, size, overlap):
    spans = split_with_offsets(text, size, overlap)
    for s, e in spans:
        assert 0 < e - s <= size
        assert not text[s].isspace() and not text[e - 1].isspace()
    assert all(b[0] > a[0] and b[1] > a[1] for a, b in zip(spans, spans[1:]))
    covered = 0
    for s, e in spans:  # every non-whitespace character is in some chunk
        assert not text[covered:s].strip()
        covered = max(covered, e)
    assert not text[covered:].strip()


@pytest.mark.parametrize("seed", range(5))
def test_guarantees_hold_on_random_messy_text(seed):
    rng = random.Random(seed)
    pieces = ["3", "12", "____", "Section 5.1", "the Company shall", "LONGWORD" * 6, "a b c", "  ", "\t"]
    for _ in range(400):
        text = "".join(rng.choice(pieces) + rng.choice(["\n\n", "\n", " ", "\n\n\n", ""])
                       for _ in range(rng.randint(1, 40)))
        size = rng.randint(2, 80)
        _check_guarantees(text, size, rng.randint(0, size - 1))
