"""Checks for the scoring logic in eval_retrieval.py. Run: python -m pytest tests -q

These run without Ollama or Chroma. If any fail, every number in the
results table is suspect.
"""

import pytest

from evals.eval_retrieval import char_recall, merged_length, overlaps, score_query


def chunk(fp, start, end):
    return {"id": f"{fp}::{start}", "file_path": fp, "start": start, "end": end}


GOLD = [{"file_path": "a.txt", "span": [100, 200]}]


def test_overlap_edges_are_half_open():
    assert overlaps(0, 100, 99, 200)       # share char 99
    assert not overlaps(0, 100, 100, 200)  # touching, no shared char
    assert not overlaps(300, 400, 100, 200)


def test_merged_length_counts_overlap_once():
    assert merged_length([(0, 100), (50, 150)]) == 150
    assert merged_length([(0, 10), (20, 30)]) == 20
    assert merged_length([]) == 0


def test_same_span_in_different_file_is_not_relevant():
    s = score_query([chunk("b.txt", 0, 1000)], GOLD, n_relevant_in_index=1)
    assert s["hit@1"] == 0.0 and s["mrr@10"] == 0.0


def test_perfect_retrieval_scores_one():
    s = score_query([chunk("a.txt", 0, 1500)], GOLD, n_relevant_in_index=1)
    assert s["hit@1"] == s["mrr@10"] == s["ndcg@10"] == s["char_recall@10"] == 1.0


def test_relevant_at_rank_3():
    ranked = [chunk("a.txt", 5000, 6000), chunk("x.txt", 0, 10), chunk("a.txt", 0, 1500)]
    s = score_query(ranked, GOLD, n_relevant_in_index=1)
    assert s["hit@1"] == 0.0 and s["hit@3"] == 1.0
    assert s["mrr@10"] == pytest.approx(1 / 3)
    assert s["ndcg@10"] == pytest.approx(0.5)  # 1/log2(4) divided by 1/log2(2)


def test_partial_char_recall_and_no_double_counting():
    # Two overlapping chunks both cover chars 100-150 of a 100-char gold span.
    ranked = [chunk("a.txt", 0, 150), chunk("a.txt", 50, 150)]
    assert char_recall(ranked, GOLD) == pytest.approx(0.5)


def test_duplicate_chunks_get_no_extra_credit():
    c = chunk("a.txt", 0, 1500)
    s = score_query([c, c, c], GOLD, n_relevant_in_index=1)
    assert s["ndcg@10"] == pytest.approx(1.0)


def test_span_recall_with_two_gold_spans():
    golds = GOLD + [{"file_path": "a.txt", "span": [9000, 9100]}]
    s = score_query([chunk("a.txt", 0, 1500)], golds, n_relevant_in_index=2)
    assert s["span_recall@1"] == 0.5
