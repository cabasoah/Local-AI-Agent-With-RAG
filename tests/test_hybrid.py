"""Checks for hybrid.py. Run: python -m pytest tests -q (no Ollama needed)."""

import pytest

from rag.hybrid import BM25Index, rrf, tokenize


def chunk(fp, start, text):
    return {"id": f"{fp}::{start}", "file_path": fp, "start": start, "end": start + len(text), "text": text}


def test_rrf_top_of_both_beats_top_of_one():
    fused = [item for item, _ in rrf([["a", "b", "c"], ["b", "a", "d"]])]
    assert set(fused[:2]) == {"a", "b"}
    assert fused.index("d") > fused.index("a")


def test_rrf_keeps_items_found_by_only_one_method():
    fused = dict(rrf([["a"], ["z"]]))
    assert "a" in fused and "z" in fused


def test_rrf_score_formula():
    fused = dict(rrf([["a"], ["a"]], k=60))
    assert fused["a"] == pytest.approx(2 / 61)


def test_tokenize_keeps_not():
    assert tokenize("The Company shall NOT solicit") == ["company", "shall", "not", "solicit"]


CORPUS = [
    chunk("a.txt", 0, "Superior Proposal means a bona fide written acquisition proposal"),
    chunk("a.txt", 100, "the company shall pay the merger consideration in cash"),
    chunk("a.txt", 200, "notices shall be delivered in writing to the parties"),
    chunk("b.txt", 0, "Superior Proposal Superior Proposal Superior Proposal"),
    chunk("tiny.txt", 0, "we collect your location data"),
    chunk("tiny.txt", 50, "we share data with advertisers"),
]


def test_bm25_finds_the_rare_exact_term():
    idx = BM25Index(CORPUS)
    assert idx.search('Definition of "Superior Proposal"', ["a.txt"], 3)[0] == "a.txt::0"


def test_bm25_stays_inside_the_routed_file():
    idx = BM25Index(CORPUS)
    assert all(cid.startswith("a.txt") for cid in idx.search("Superior Proposal", ["a.txt"], 10))


def test_bm25_drops_chunks_with_no_matching_words():
    idx = BM25Index(CORPUS)
    assert idx.search("Superior Proposal", ["a.txt"], 10) == ["a.txt::0"]


def test_bm25_works_on_two_chunk_documents():
    # Classic per-document BM25 gives these words zero weight; ours must not.
    idx = BM25Index(CORPUS)
    assert idx.search("Do they collect location?", ["tiny.txt"], 2)[0] == "tiny.txt::0"
    assert all(v > 0 for v in idx.idf.values())
