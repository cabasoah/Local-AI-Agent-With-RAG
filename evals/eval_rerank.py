"""
Step 5: does a cross-encoder reranker beat hybrid search?

Three parts:
  A. Candidates: run hybrid search once per question (top 100 of the fused list)
     and cache them. Only this part needs Ollama; later runs reuse the cache.
  B. Pool check: how often is a relevant chunk ANYWHERE in the top 10/20/50/100
     candidates? A reranker can only reorder what it's given, so this is its ceiling.
  C. Rerank the top N candidates with the cross-encoder and score the top 10.

"""

import argparse
import json
import statistics
import time
from collections import defaultdict

from evals.eval_hybrid import print_comparison, wins_and_losses
from evals.eval_retrieval import (
    CHUNKS_PATH,
    QUERIES_PATH,
    RESULTS_DIR,
    chunk_hits_span,
    index_chunks_by_file,
    load_jsonl,
    print_table,
    run,
)
from rag.hybrid import HybridRetriever
from rag.rerank import DEFAULT_MODEL, RerankRetriever
from rag.router import DocumentRouter

POOL_CHECKS = [10, 20, 50, 100]


def candidates_path(limit: int):
    return RESULTS_DIR / ("candidates_hybrid.jsonl" if not limit else f"candidates_hybrid_limit{limit}.jsonl")


def build_candidates(queries, chunks, store, path) -> None:
    hybrid = HybridRetriever(chunks, store, DocumentRouter(chunks), mode="hybrid", candidates=50)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, q in enumerate(queries, 1):
            t0 = time.perf_counter()
            cands = hybrid(q["query"], 100)  # 50 dense + 50 BM25, fused: at most 100
            ms = (time.perf_counter() - t0) * 1000
            f.write(json.dumps({"id": q["id"], "query": q["query"], "retrieval_ms": ms,
                                "candidates": [c["id"] for c in cands]}) + "\n")
            if i % 25 == 0 or i == len(queries):
                print(f"  candidates {i}/{len(queries)}", end="\r", flush=True)
    print()


def pool_report(queries, cand_rows, chunk_meta) -> None:
    """Share of questions with at least one relevant chunk in the first N candidates."""
    by_id = {r["id"]: r for r in cand_rows}
    stats = defaultdict(lambda: defaultdict(float))
    for q in queries:
        cands = by_id[q["id"]]["candidates"]
        relevant = [any(chunk_hits_span(chunk_meta[cid], g) for g in q["snippets"]) for cid in cands]
        st = stats[q["dataset"]]
        st["n"] += 1
        st["pool_size"] += len(cands)
        for n in POOL_CHECKS:
            st[n] += any(relevant[:n])

    print("Reranker ceiling: share of questions with a relevant chunk in the first N candidates")
    print("| dataset | n | avg pool | " + " | ".join(f"in top {n}" for n in POOL_CHECKS) + " |")
    print("|---" * (len(POOL_CHECKS) + 3) + "|")
    for ds in sorted(stats):
        st = stats[ds]
        cells = " | ".join(f"{st[n] / st['n']:.3f}" for n in POOL_CHECKS)
        print(f"| {ds} | {int(st['n'])} | {st['pool_size'] / st['n']:.0f} | {cells} |")


def end_to_end_latency(name: str, cand_rows) -> None:
    """run() timed only the rerank step (candidates came from the cache).
    Add each question's own hybrid time back to get the real total."""
    hybrid_ms = {r["id"]: r["retrieval_ms"] for r in cand_rows}
    with open(RESULTS_DIR / f"{name}.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    rerank_ms = sorted(r["latency_ms"] for r in rows)
    total = sorted(r["latency_ms"] + hybrid_ms[r["id"]] for r in rows)
    p95 = lambda xs: xs[min(len(xs) - 1, int(0.95 * len(xs)))]
    print(f"Latency for '{name}': rerank only p50 {statistics.median(rerank_ms):.0f} ms / "
          f"p95 {p95(rerank_ms):.0f} ms; end to end (hybrid + rerank) p50 "
          f"{statistics.median(total):.0f} ms / p95 {p95(total):.0f} ms")


def main(argv=None, store=None, reranker=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--pools", type=int, nargs="+", default=[20], help="candidates passed to the reranker")
    p.add_argument("--limit", type=int, default=0, help="first N questions per dataset (0 = all)")
    p.add_argument("--rebuild-candidates", action="store_true")
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
    chunk_meta = {c["id"]: c for c in chunks}
    by_file = index_chunks_by_file(chunks)

    # --- A. candidates (cached) ---
    path = candidates_path(args.limit)
    if args.rebuild_candidates or not path.exists():
        if store is None:
            from rag.build_index import get_vector_store
            store = get_vector_store()
        print(f"Building hybrid candidates for {len(queries)} questions (cached to {path})...")
        build_candidates(queries, chunks, store, path)
    else:
        print(f"Using cached candidates from {path} (--rebuild-candidates to refresh)")
    cand_rows = load_jsonl(path)

    # Retrievers receive only the query TEXT, so candidates are looked up by text.
    # Some benchmark questions share identical wording (each with its own answer
    # key). That's fine as long as identical wording got identical candidates,
    # which it should: retrieval depends only on the text. Verify, don't assume.
    lists_by_text = defaultdict(list)
    for r in cand_rows:
        lists_by_text[r["query"]].append(r["candidates"])
    shared = {t: ls for t, ls in lists_by_text.items() if len(ls) > 1}
    if any(any(l != ls[0] for l in ls) for ls in shared.values()):
        raise SystemExit("Identical query texts got different candidates; text lookup would be wrong.")
    if shared:
        n_questions = sum(len(ls) for ls in shared.values())
        print(f"Note: {n_questions} questions share their wording with another question "
              f"({len(shared)} distinct texts). They get identical candidates and are each "
              f"scored against their own answer key.")
    candidates_by_query = {r["query"]: r["candidates"] for r in cand_rows}

    # --- B. ceiling ---
    print()
    pool_report(queries, cand_rows, chunk_meta)

    # --- C. rerank ---
    if reranker is None:
        from rag.rerank import CrossEncoderReranker
        print(f"\nLoading {args.model} (downloads once, then cached)...")
        reranker = CrossEncoderReranker(args.model)

    short = args.model.split("/")[-1].lower()
    suffix = f"_limit{args.limit}" if args.limit else ""
    summaries = {}
    hybrid_summary = RESULTS_DIR / f"routed_hybrid{suffix}.summary.json"
    if hybrid_summary.exists():
        summaries["hybrid"] = json.loads(hybrid_summary.read_text(encoding="utf-8"))

    for pool in args.pools:
        name = f"rerank_{short}_pool{pool}{suffix}"
        print(f"\nRunning '{name}' on {len(queries)} queries...")
        retriever = RerankRetriever(candidates_by_query, chunks, reranker, pool)
        summaries[f"rerank{pool}"] = run(name, retriever, queries, by_file)
        print_table(summaries[f"rerank{pool}"])
        end_to_end_latency(name, cand_rows)
        if "hybrid" in summaries:
            wins_and_losses(f"routed_hybrid{suffix}", name)

    if len(summaries) > 1:
        print("\nSide by side:")
        print_comparison(summaries)
    elif "hybrid" not in summaries:
        print(f"\n(No {hybrid_summary.name} found: run python -m evals.eval_hybrid"
              f"{' --limit ' + str(args.limit) if args.limit else ''} to get the comparison.)")


if __name__ == "__main__":
    main()
