"""Checks for rerank.py with a stand-in scorer. Run: python -m pytest tests -q"""

from rag.rerank import RerankRetriever


class FixedScores:
    """Stand-in for the cross-encoder: scores come from a lookup on chunk text."""
    def __init__(self, table):
        self.table, self.seen = table, []

    def score(self, question, texts):
        self.seen.append((question, list(texts)))
        return [self.table.get(t, 0.0) for t in texts]


CHUNKS = [
    {"id": f"a.txt::{i}", "file_path": "a.txt", "start": i, "end": i + 10, "text": t}
    for i, t in enumerate(["alpha", "beta", "gamma", "delta"])
]
QUERY = "Consider the Agreement between X and Y; Where is the No-Shop Clause"
CANDS = {QUERY: ["a.txt::0", "a.txt::1", "a.txt::2", "a.txt::3"]}


def test_reorders_by_score():
    r = RerankRetriever(CANDS, CHUNKS, FixedScores({"gamma": 9, "alpha": 1}), pool_size=4)
    assert [c["id"] for c in r(QUERY, 2)] == ["a.txt::2", "a.txt::0"]


def test_only_the_pool_is_scored():
    scorer = FixedScores({"delta": 99})
    r = RerankRetriever(CANDS, CHUNKS, scorer, pool_size=2)
    ids = [c["id"] for c in r(QUERY, 10)]
    assert "a.txt::3" not in ids and len(scorer.seen[0][1]) == 2


def test_reranker_sees_question_without_title():
    scorer = FixedScores({})
    RerankRetriever(CANDS, CHUNKS, scorer, pool_size=4)(QUERY, 1)
    assert scorer.seen[0][0] == "Where is the No-Shop Clause"


def test_ties_keep_hybrid_order():
    r = RerankRetriever(CANDS, CHUNKS, FixedScores({}), pool_size=4)
    assert [c["id"] for c in r(QUERY, 4)] == CANDS[QUERY]
