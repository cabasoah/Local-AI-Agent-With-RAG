"""
Step 4: compare dense, BM25 and hybrid search inside the routed contract.

All three use the same router (step 3) and the same 743 questions, in one
session, so the latency numbers are comparable too.

Usage:
    python -m evals.eval_hybrid                     # all three modes
    python -m evals.eval_hybrid --modes bm25        # BM25 only: no Ollama needed, ~seconds
    python -m evals.eval_hybrid --limit 10          # quick smoke test, 10 questions per dataset
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evals.eval_retrieval import (
    CHUNKS_PATH,
    QUERIES_PATH,
    RESULTS_DIR,
    index_chunks_by_file,
    load_jsonl,
    print_table,
    run,
)
from rag.hybrid import CANDIDATES, RRF_K, HybridRetriever
from rag.router import DocumentRouter

COMPARE = ["hit@1", "hit@5", "mrr@10", "char_recall@10"]


def print_comparison(summaries: dict[str, dict]) -> None:
    """One table: rows = dataset, columns = metric per mode."""
    modes = list(summaries)
    datasets = list(next(iter(summaries.values())))
    header = " | ".join(f"{m} {mode}" for m in COMPARE for mode in modes)
    print(f"| dataset | {header} |")
    print("|---" * (1 + len(COMPARE) * len(modes)) + "|")
    for ds in datasets:
        cells = " | ".join(f"{summaries[mode][ds][m]:.3f}" for m in COMPARE for mode in modes)
        print(f"| {ds} | {cells} |")


def wins_and_losses(name_a: str, name_b: str, metric: str = "hit@5") -> None:
    """Per question: where did B find something A missed, and the reverse?
    An average can hide a method that fixes 40 questions and breaks 35."""
    def load(name):
        with open(RESULTS_DIR / f"{name}.jsonl", encoding="utf-8") as f:
            return {r["id"]: r for r in map(json.loads, f)}

    a, b = load(name_a), load(name_b)
    counts = defaultdict(lambda: [0, 0])
    for qid, ra in a.items():
        rb = b[qid]
        if rb[metric] > ra[metric]:
            counts[ra["dataset"]][0] += 1
        elif rb[metric] < ra[metric]:
            counts[ra["dataset"]][1] += 1

    print(f"\n{name_b} vs {name_a} on {metric}: questions gained / lost")
    print("| dataset | gained | lost |")
    print("|---|---|---|")
    for ds in sorted(counts):
        print(f"| {ds} | {counts[ds][0]} | {counts[ds][1]} |")


def main(argv=None, store=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--modes", nargs="+", default=["dense", "bm25", "hybrid"],
                   choices=["dense", "bm25", "hybrid"])
    p.add_argument("--limit", type=int, default=0, help="first N questions per dataset (0 = all)")
    p.add_argument("--candidates", type=int, default=CANDIDATES)
    p.add_argument("--rrf-k", type=int, default=RRF_K,
                   help="leave at 60; changing it after seeing results is tuning on the test set")
    args = p.parse_args(argv)

    queries = load_jsonl(QUERIES_PATH)
    if args.limit:
        seen, kept = defaultdict(int), []
        for q in queries:
            if seen[q["dataset"]] < args.limit:
                kept.append(q)
                seen[q["dataset"]] += 1
        queries = kept

    chunks = load_jsonl(CHUNKS_PATH)
    by_file = index_chunks_by_file(chunks)
    router = DocumentRouter(chunks)
    if store is None and any(m != "bm25" for m in args.modes):
        from rag.build_index import get_vector_store
        store = get_vector_store()

    summaries = {}
    for mode in args.modes:
        name = f"routed_{mode}" + (f"_limit{args.limit}" if args.limit else "")
        retriever = HybridRetriever(chunks, store, router, mode=mode,
                                    candidates=args.candidates, rrf_k=args.rrf_k)
        print(f"\nRunning '{name}' on {len(queries)} queries...")
        summaries[mode] = run(name, retriever, queries, by_file)
        print_table(summaries[mode])

    if len(summaries) > 1:
        print("\nSide by side:")
        print_comparison(summaries)
    if {"dense", "hybrid"} <= set(summaries):
        suffix = f"_limit{args.limit}" if args.limit else ""
        wins_and_losses(f"routed_dense{suffix}", f"routed_hybrid{suffix}")


if __name__ == "__main__":
    main()
