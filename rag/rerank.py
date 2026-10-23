"""
Step 5: rerank hybrid search candidates with a cross-encoder.

Embeddings (dense search) turn the question and each chunk into vectors
SEPARATELY, then compare the vectors. Fast, because chunk vectors are computed
once ahead of time, but the model never sees question and chunk together.

A cross-encoder reads "question + chunk" as ONE input and outputs a single
relevance score. It can notice that a chunk saying "shall not solicit,
initiate or knowingly encourage" answers a question about a "No-Shop Clause".
It is far more accurate per pair, and far slower: nothing can be precomputed,
so it runs once per (question, chunk) pair at question time.

Hence the standard two-stage pattern:
    fast search (hybrid) -> top N candidates -> slow, accurate rerank -> top k
"""

from rag.router import split_query

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 22M parameters, fast on CPU


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_MODEL, max_length: int = 512, batch_size: int = 32):
        from sentence_transformers import CrossEncoder  # imported here so tests don't need torch

        # max_length=512 tokens fits the question plus one 1,500-character chunk.
        self.model = CrossEncoder(model_name, max_length=max_length)
        self.batch_size = batch_size

    def score(self, question: str, texts: list[str]) -> list[float]:
        pairs = [(question, t) for t in texts]
        return [float(s) for s in self.model.predict(pairs, batch_size=self.batch_size)]


class RerankRetriever:
    """(query, k) -> ranked chunks, like every other retriever.

    candidates_by_query: the hybrid ranking for each query, computed once and
    cached by eval_rerank.py, so trying another pool size or model doesn't
    re-run Ollama.
    """

    def __init__(self, candidates_by_query: dict[str, list[str]], chunks: list[dict],
                 reranker, pool_size: int):
        self.candidates = candidates_by_query
        self.reranker = reranker
        self.pool_size = pool_size
        self.text = {c["id"]: c["text"] for c in chunks}
        self.meta = {
            c["id"]: {"id": c["id"], "file_path": c["file_path"], "start": c["start"], "end": c["end"]}
            for c in chunks
        }

    def __call__(self, query: str, k: int) -> list[dict]:
        question = split_query(query)[1]
        pool = self.candidates[query][: self.pool_size]
        if not pool:
            return []
        scores = self.reranker.score(question, [self.text[cid] for cid in pool])
        # Highest score first; on a tie, keep the hybrid order (lower index first).
        order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
        return [self.meta[pool[i]] for i in order[:k]]
