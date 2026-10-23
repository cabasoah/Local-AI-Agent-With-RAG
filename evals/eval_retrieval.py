"""
Step 2: span-overlap retrieval evaluation on LegalBench-RAG mini.

The ground truth is character spans in documents, not chunk IDs. So a retrieved
chunk counts as relevant when it comes from the same file as a gold span and
its [start, end) range overlaps that span.

Usage:
    python -m evals.eval_retrieval                       # dense baseline, all queries
    python -m evals.eval_retrieval --limit 10            # quick smoke test (10 per dataset)
    python -m evals.eval_retrieval --name dense_prefix --query-prefix mxbai

Outputs:
    evals/results/<name>.jsonl         per-query scores + retrieved chunk IDs
    evals/results/<name>.summary.json  averaged metrics, overall and per dataset
"""

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

QUERIES_PATH = Path("data/processed/legalbench_mini.jsonl")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
RESULTS_DIR = Path("evals/results")

K_VALUES = [1, 3, 5, 10]
MAX_K = max(K_VALUES)

# mxbai-embed-large was trained to see this prefix on queries (not documents).
# The model card recommends it for retrieval. Your restaurant version never used it.
QUERY_PREFIXES = {
    "none": "",
    "mxbai": "Represent this sentence for searching relevant passages: ",
}


# ---------------------------------------------------------------------------
# 1. Overlap helpers
# ---------------------------------------------------------------------------
def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Half-open intervals [start, end) share at least one character."""
    return a_start < b_end and b_start < a_end


def chunk_hits_span(chunk: dict, gold: dict) -> bool:
    gs, ge = gold["span"]
    return chunk["file_path"] == gold["file_path"] and overlaps(chunk["start"], chunk["end"], gs, ge)


def merged_length(intervals: list[tuple[int, int]]) -> int:
    """Total length covered by intervals, counting overlapping parts once.

    Needed because neighbouring chunks overlap by 200 chars: two chunks covering
    the same gold characters must not count those characters twice.
    """
    total, cur_start, cur_end = 0, None, None
    for s, e in sorted(intervals):
        if cur_end is None or s > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = s, e
        else:
            cur_end = max(cur_end, e)
    if cur_end is not None:
        total += cur_end - cur_start
    return total


# ---------------------------------------------------------------------------
# 2. Metrics for one query
# ---------------------------------------------------------------------------
def char_recall(top: list[dict], golds: list[dict]) -> float:
    """Share of gold characters covered by the retrieved chunks.
    This is the recall definition the LegalBench-RAG paper uses."""
    total = covered = 0
    for g in golds:
        gs, ge = g["span"]
        total += ge - gs
        pieces = [
            (max(gs, c["start"]), min(ge, c["end"]))
            for c in top
            if chunk_hits_span(c, g)
        ]
        covered += merged_length(pieces)
    return covered / total if total else 0.0


def score_query(retrieved: list[dict], golds: list[dict], n_relevant_in_index: int) -> dict:
    """retrieved: ranked chunks (best first), each with file_path/start/end.
    golds: ground-truth snippets, each with file_path and span=[start, end].
    n_relevant_in_index: how many chunks in the WHOLE index overlap a gold span
    (needed for nDCG's "perfect ranking" baseline)."""
    # Drop repeated chunks (keep first position). Without this, a retriever
    # that returns the same chunk twice gets double credit and nDCG can go
    # above 1. Hybrid search in step 4 merges two lists, so this can happen.
    seen = set()
    retrieved = [c for c in retrieved if not (c["id"] in seen or seen.add(c["id"]))]

    relevant =[any(chunk_hits_span(c, g) for g in golds) for c in retrieved]
    scores = {}

    for k in K_VALUES:
        top = retrieved[:k]
        # hit@k: did ANY relevant chunk show up in the top k?
        scores[f"hit@{k}"] = 1.0 if any(relevant[:k]) else 0.0
        # span_recall@k: what share of the gold snippets did we touch at all?
        touched = sum(1 for g in golds if any(chunk_hits_span(c, g) for c in top))
        scores[f"span_recall@{k}"] = touched / len(golds)
        # char_recall@k: what share of the gold characters did we return?
        scores[f"char_recall@{k}"] = char_recall(top, golds)

    # MRR@10: 1 / rank of the first relevant chunk (0 if none in top 10)
    first = next((i for i, r in enumerate(relevant[:MAX_K], 1) if r), None)
    scores[f"mrr@{MAX_K}"] = 1.0 / first if first else 0.0

    # nDCG@10: rewards relevant chunks near the top, compared to the best
    # possible ranking given how many relevant chunks exist in the index.
    dcg = sum(1 / math.log2(i + 1) for i, r in enumerate(relevant[:MAX_K], 1) if r)
    ideal_n = min(n_relevant_in_index, MAX_K)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    scores[f"ndcg@{MAX_K}"] = dcg / idcg if idcg else 0.0

    return scores


# ---------------------------------------------------------------------------
# 3. Facts about the index that don't depend on the retriever
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def index_chunks_by_file(chunks: list[dict]) -> dict[str, list[dict]]:
    by_file = defaultdict(list)
    for c in chunks:
        by_file[c["file_path"]].append(c)
    return by_file


def count_relevant_chunks(query: dict, by_file: dict) -> int:
    return sum(
        1
        for g in query["snippets"]
        for c in by_file.get(g["file_path"], [])
        if chunk_hits_span(c, g)
    )


def coverage_report(queries: list[dict], by_file: dict) -> None:
    """Per dataset: how long are gold spans, and how many fit in ONE chunk?"""
    stats = defaultdict(lambda: {"n": 0, "inside": 0, "lens": []})
    for q in queries:
        for g in q["snippets"]:
            gs, ge = g["span"]
            st = stats[q["dataset"]]
            st["n"] += 1
            st["lens"].append(ge - gs)
            if any(c["start"] <= gs and ge <= c["end"] for c in by_file.get(g["file_path"], [])):
                st["inside"] += 1

    print("Gold spans vs. chunks (independent of the retriever)")
    print("| dataset | gold spans | median span chars | max span chars | fully inside one chunk |")
    print("|---|---|---|---|---|")
    for ds in sorted(stats):
        st = stats[ds]
        print(f"| {ds} | {st['n']} | {statistics.median(st['lens']):.0f} | "
              f"{max(st['lens'])} | {st['inside'] / st['n']:.1%} |")
    print()


# ---------------------------------------------------------------------------
# 4. Retrievers. Every retriever is just: (query, k) -> ranked list of chunks.
#    Steps 3-5 add BM25, hybrid and reranked versions with the same signature.
# ---------------------------------------------------------------------------
def make_dense_retriever(query_prefix: str = ""):
    from rag.build_index import get_vector_store  # imported here so tests don't need Ollama

    store = get_vector_store()

    def retrieve(query: str, k: int) -> list[dict]:
        docs = store.similarity_search(query_prefix + query, k=k)
        return [
            {
                "id": f"{d.metadata['file_path']}::{d.metadata['start']}",
                "file_path": d.metadata["file_path"],
                "start": int(d.metadata["start"]),
                "end": int(d.metadata["end"]),
            }
            for d in docs
        ]

    return retrieve


# ---------------------------------------------------------------------------
# 5. Run + report
# ---------------------------------------------------------------------------
TABLE_COLS = ["hit@1", "hit@5", "span_recall@5", "span_recall@10",
              "char_recall@10", "mrr@10", "ndcg@10"]


def average(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if isinstance(rows[0][k], float) and k != "latency_ms"]
    out = {k: statistics.mean(r[k] for r in rows) for k in keys}
    lat = sorted(r["latency_ms"] for r in rows)
    out["latency_p50_ms"] = statistics.median(lat)
    out["latency_p95_ms"] = lat[min(len(lat) - 1, math.ceil(0.95 * len(lat)) - 1)]
    out["n"] = len(rows)
    return out


def summarize(rows: list[dict]) -> dict:
    per_ds = defaultdict(list)
    for r in rows:
        per_ds[r["dataset"]].append(r)
    summary = {ds: average(rs) for ds, rs in sorted(per_ds.items())}

    # "overall" = equal weight per dataset (macro average), like the paper.
    # A plain average over all queries would let whichever dataset has the
    # most queries dominate.
    metric_keys = [k for k in summary[next(iter(summary))] if k != "n"]
    summary["overall"] = {k: statistics.mean(summary[ds][k] for ds in per_ds) for k in metric_keys}
    summary["overall"]["n"] = len(rows)
    return summary


def print_table(summary: dict) -> None:
    print("| dataset | n | " + " | ".join(TABLE_COLS) + " | p50 ms | p95 ms |")
    print("|---" * (len(TABLE_COLS) + 4) + "|")
    for ds, m in summary.items():
        cells = " | ".join(f"{m[c]:.3f}" for c in TABLE_COLS)
        label = f"**{ds}**" if ds == "overall" else ds
        print(f"| {label} | {m['n']} | {cells} | {m['latency_p50_ms']:.0f} | {m['latency_p95_ms']:.0f} |")


def run(name: str, retrieve, queries: list[dict], by_file: dict) -> dict:
    rows = []
    for i, q in enumerate(queries, 1):
        t0 = time.perf_counter()
        retrieved = retrieve(q["query"], MAX_K)
        latency_ms = (time.perf_counter() - t0) * 1000

        scores = score_query(retrieved, q["snippets"], count_relevant_chunks(q, by_file))
        rows.append({
            "id": q["id"],
            "dataset": q["dataset"],
            "latency_ms": latency_ms,
            **scores,
            "retrieved": [c["id"] for c in retrieved],  # kept for error analysis
        })
        if i % 25 == 0 or i == len(queries):
            print(f"  {i}/{len(queries)}", end="\r", flush=True)
    print()

    summary = summarize(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(RESULTS_DIR / f"{name}.summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="dense", help="label for the results files")
    parser.add_argument("--query-prefix", choices=QUERY_PREFIXES, default="none")
    parser.add_argument("--limit", type=int, default=0, help="first N queries per dataset (0 = all)")
    args = parser.parse_args()

    queries = load_jsonl(QUERIES_PATH)
    if args.limit:
        seen = defaultdict(int)
        kept = []
        for q in queries:
            if seen[q["dataset"]] < args.limit:
                kept.append(q)
                seen[q["dataset"]] += 1
        queries = kept

    by_file = index_chunks_by_file(load_jsonl(CHUNKS_PATH))
    coverage_report(queries, by_file)

    retrieve = make_dense_retriever(QUERY_PREFIXES[args.query_prefix])
    print(f"Running '{args.name}' on {len(queries)} queries...")
    summary = run(args.name, retrieve, queries, by_file)
    print()
    print_table(summary)
    print(f"\nSaved to {RESULTS_DIR}/{args.name}.jsonl and .summary.json")


if __name__ == "__main__":
    main()
