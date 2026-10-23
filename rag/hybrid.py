"""
Step 4: hybrid search inside the routed contract.

For each question:
  1. router.py picks the contract from the title                   (step 3)
  2a. dense search: question -> top 50 chunks in that contract     (meaning)
  2b. BM25 search:  question -> top 50 chunks in that contract     (exact words)
  3. Reciprocal Rank Fusion (RRF) merges the two rankings -> top k

Modes let the eval compare each half alone against the combination:
  "dense"  = step 3's system (2a only)
  "bm25"   = keyword search only (2b only)
  "hybrid" = both, fused with RRF
"""

import math
import re
from collections import Counter, defaultdict

from rag.router import DocumentRouter, split_query

MXBAI_PREFIX = "Represent this sentence for searching relevant passages: "

# 60 is the constant from the paper that introduced RRF (Cormack, Clarke &
# Buettcher, SIGIR 2009). Deliberately NOT tuned on this benchmark: picking it
# by looking at these results would be tuning on the test set.
RRF_K = 60
CANDIDATES = 50

# Common words that say nothing about WHICH clause is meant. "not" is kept on
# purpose: "shall not" and "shall" mean opposite things in a contract.
STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "for", "on", "by", "with", "or", "and",
    "is", "are", "be", "been", "was", "were", "as", "at", "from", "that", "this",
    "these", "those", "it", "its", "any", "does", "do", "what", "which", "who",
    "how", "when", "where", "whether", "there", "such", "s",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Merge several ranked lists of IDs into one.

    Each list gives an item 1 / (k + rank) points; points add up across lists.
    Only RANKS are used, never raw scores, because a cosine similarity (0 to 1)
    and a BM25 score (0 to 30+) are on scales that can't be added together.
    An item near the top of both lists beats an item at the top of only one.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] += 1.0 / (k + rank)
    # Tie-break by ID so the same input always gives the same order.
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


# ---------------------------------------------------------------------------
# BM25 over all chunks, searchable inside chosen files
# ---------------------------------------------------------------------------
class BM25Index:
    """BM25 written out by hand (about 30 lines) instead of rank_bm25, for two reasons:

    1. Word rarity (IDF) is computed over ALL 7,059 chunks, then scoring is
       restricted to the routed contract, the way a search engine applies a
       filter. Computing IDF per contract breaks on short documents: a 2-chunk
       privacy policy gives classic BM25 weights of zero or below for nearly
       every word.
    2. It uses the IDF formula Lucene/Elasticsearch use, which never goes negative:
           idf = ln(1 + (N - n + 0.5) / (n + 0.5))
    """

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf: dict[str, Counter] = {}
        self.length: dict[str, int] = {}
        self.ids_by_file: dict[str, list[str]] = defaultdict(list)
        doc_freq: Counter = Counter()

        for c in chunks:
            counts = Counter(tokenize(c["text"]))
            self.tf[c["id"]] = counts
            self.length[c["id"]] = sum(counts.values())
            self.ids_by_file[c["file_path"]].append(c["id"])
            doc_freq.update(counts.keys())

        n_chunks = len(chunks)
        self.avg_length = sum(self.length.values()) / n_chunks
        self.idf = {
            term: math.log(1 + (n_chunks - n + 0.5) / (n + 0.5))
            for term, n in doc_freq.items()
        }

    def score(self, chunk_id: str, query_terms: list[str]) -> float:
        tf = self.tf[chunk_id]
        # Longer chunks get a bigger denominator, so they don't win just by being long.
        norm = self.k1 * (1 - self.b + self.b * self.length[chunk_id] / self.avg_length)
        total = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if f:
                # Repeats help, with diminishing returns: f / (f + norm) never exceeds 1.
                total += self.idf[term] * f * (self.k1 + 1) / (f + norm)
        return total

    def search(self, question: str, files: list[str], n: int) -> list[str]:
        terms = list(dict.fromkeys(tokenize(question)))  # each query word once
        scored = [
            (self.score(cid, terms), cid)
            for fp in files
            for cid in self.ids_by_file.get(fp, [])
        ]
        # A chunk sharing no words with the question gets 0. Leaving it out keeps
        # it from collecting RRF points just for being in the list.
        scored = [pair for pair in scored if pair[0] > 0]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [cid for _, cid in scored[:n]]


# ---------------------------------------------------------------------------
# The retriever: (query, k) -> ranked chunks, same signature as every other
# ---------------------------------------------------------------------------
class HybridRetriever:
    def __init__(self, chunks: list[dict], store, router: DocumentRouter,
                 mode: str = "hybrid", candidates: int = CANDIDATES, rrf_k: int = RRF_K):
        if mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"unknown mode {mode!r}")
        self.store, self.router, self.mode = store, router, mode
        self.candidates, self.rrf_k = candidates, rrf_k
        self.bm25 = BM25Index(chunks) if mode != "dense" else None
        self.chunk_by_id = {
            c["id"]: {"id": c["id"], "file_path": c["file_path"], "start": c["start"], "end": c["end"]}
            for c in chunks
        }

    def dense_search(self, question: str, file_path: str) -> list[str]:
        docs = self.store.similarity_search(
            MXBAI_PREFIX + question, k=self.candidates, filter={"file_path": file_path}
        )
        return [f"{d.metadata['file_path']}::{d.metadata['start']}" for d in docs]

    def __call__(self, query: str, k: int) -> list[dict]:
        """Benchmark entry point: route by the title, then search with the question."""
        title, question = split_query(query)
        file_path = self.router.route(title or query, top_n=1)[0][0]
        return self.search(question, file_path, k)

    def search(self, question: str, file_path: str, k: int) -> list[dict]:
        """App entry point: the user already chose the contract."""
        dense_ids = self.dense_search(question, file_path) if self.mode != "bm25" else []
        bm25_ids = self.bm25.search(question, [file_path], self.candidates) if self.mode != "dense" else []

        if self.mode == "dense":
            ranked = dense_ids
        elif self.mode == "bm25":
            ranked = bm25_ids
        else:
            ranked = [cid for cid, _ in rrf([dense_ids, bm25_ids], self.rrf_k)]

        return [self.chunk_by_id[cid] for cid in ranked[:k]]
