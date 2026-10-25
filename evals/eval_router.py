"""
Step 3b: evaluate the document router, then the full two-stage pipeline.

Part 1 - router alone: how often does the right document come first / in the top 3?
Part 2 - full pipeline (real router, no answer key):
    routed_bm25_top1   search the question inside the router's #1 document
    routed_bm25_top3   search the question inside the router's top 3 documents
                       (a safety net: costs precision when the router was right,
                       saves the query when it was wrong)

"""

import argparse
from collections import defaultdict

from evals.eval_retrieval import (
    CHUNKS_PATH,
    QUERIES_PATH,
    QUERY_PREFIXES,
    index_chunks_by_file,
    load_jsonl,
    print_table,
    run,
)
from rag.router import CARD_CHARS, DocumentRouter, split_query

PREFIX = QUERY_PREFIXES["mxbai"]


def routing_report(router: DocumentRouter, queries: list[dict], n_examples: int = 3) -> None:
    stats = defaultdict(lambda: defaultdict(float))
    misses = defaultdict(list)
    for q in queries:
        gold = {s["file_path"] for s in q["snippets"]}
        title = split_query(q["query"])[0] or q["query"]
        ranked = [fp for fp, _ in router.route(title, top_n=3)]

        st = stats[q["dataset"]]
        st["n"] += 1
        st["acc@1"] += ranked[0] in gold
        st["acc@3"] += any(fp in gold for fp in ranked)
        if ranked[0] not in gold:
            misses[q["dataset"]].append((title, sorted(gold)[0], ranked[0]))

    print("Router accuracy (is the right document ranked first / in the top 3?)")
    print("| dataset | n | acc@1 | acc@3 |")
    print("|---|---|---|---|")
    for ds in sorted(stats):
        st = stats[ds]
        print(f"| {ds} | {int(st['n'])} | {st['acc@1'] / st['n']:.3f} | {st['acc@3'] / st['n']:.3f} |")
    total = sum(st["n"] for st in stats.values())
    print(f"| **all** | {int(total)} | {sum(st['acc@1'] for st in stats.values()) / total:.3f} | "
          f"{sum(st['acc@3'] for st in stats.values()) / total:.3f} |")

    for ds in sorted(misses):
        print(f"\n{ds}: {len(misses[ds])} misrouted. Examples:")
        for title, wanted, got in misses[ds][:n_examples]:
            print(f"  TITLE:  {title[:150]}")
            print(f"  WANTED: {wanted}")
            print(f"  GOT:    {got}")


def make_routed_retriever(store, router: DocumentRouter, n_docs: int):
    def retrieve(query: str, k: int) -> list[dict]:
        title, question = split_query(query)
        files = [fp for fp, _ in router.route(title or query, top_n=n_docs)]
        where = {"file_path": files[0]} if len(files) == 1 else {"file_path": {"$in": files}}
        docs = store.similarity_search(PREFIX + question, k=k, filter=where)
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


def main(argv=None, store=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-only", action="store_true")
    parser.add_argument("--card-chars", type=int, default=CARD_CHARS)
    parser.add_argument("--unique-terms", action="store_true",
                        help="count each word once per document card")
    parser.add_argument("--explain", metavar="TITLE",
                        help="show which words decided the routing for one title, then exit")
    args = parser.parse_args(argv)

    queries = load_jsonl(QUERIES_PATH)
    chunks = load_jsonl(CHUNKS_PATH)
    router = DocumentRouter(chunks, card_chars=args.card_chars, unique_terms=args.unique_terms)

    if args.explain:
        for fp, total, parts in router.explain(args.explain):
            detail = ", ".join(f"{w}: x{tf} -> {pts:.2f}" for w, (tf, pts) in parts.items())
            print(f"{total:6.2f}  {fp}\n        {detail}")
        return

    routing_report(router, queries)
    if args.router_only:
        return

    if store is None:
        from rag.build_index import get_vector_store
        store = get_vector_store()
    by_file = index_chunks_by_file(chunks)

    for n_docs in (1, 3):
        name = f"routed_bm25_top{n_docs}" + ("_unique" if args.unique_terms else "")
        print(f"\nRunning '{name}' on {len(queries)} queries...")
        print_table(run(name, make_routed_retriever(store, router, n_docs), queries, by_file))


if __name__ == "__main__":
    main()
