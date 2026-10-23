"""
Upper-bound test: how good would retrieval be if we always searched inside
the RIGHT document?

Every LegalBench-RAG query is "Consider the <document title>; <question>".
This script cheats on purpose: it restricts the search to the gold document
(using the answer key) so we can measure the best case for two-stage
retrieval (pick document -> search within it) before building the real router.

Runs two variants, both with the mxbai query prefix:
  routed_full      full query text, searched only inside the gold document
  routed_question  question part only (after the ';'), inside the gold document

Compare against dense_prefix (no routing). These are NOT results a real
system can report, because a real system doesn't know the gold document.
They tell us whether routing is worth building.
"""

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

PREFIX = QUERY_PREFIXES["mxbai"]


def split_query(text: str) -> tuple[str, str]:
    """'Consider the X; What is Y' -> ('Consider the X', 'What is Y')."""
    title, sep, question = text.partition(";")
    return (title.strip(), question.strip()) if sep else ("", text.strip())


def make_routed_retriever(store, gold_files_by_query: dict, question_only: bool):
    def retrieve(query: str, k: int) -> list[dict]:
        files = sorted(gold_files_by_query[query])
        where = {"file_path": files[0]} if len(files) == 1 else {"file_path": {"$in": files}}
        text = split_query(query)[1] if question_only else query
        docs = store.similarity_search(PREFIX + text, k=k, filter=where)
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


def main(store=None) -> None:
    if store is None:
        from rag.build_index import get_vector_store
        store = get_vector_store()

    queries = load_jsonl(QUERIES_PATH)
    by_file = index_chunks_by_file(load_jsonl(CHUNKS_PATH))

    # Retrievers only receive the query TEXT, so we look up gold files by text.
    # That's only safe if no two queries share wording but point at different
    # documents. Check it rather than assume it.
    gold_files_by_query = defaultdict(set)
    for q in queries:
        gold_files_by_query[q["query"]] |= {s["file_path"] for s in q["snippets"]}
    per_query = defaultdict(set)
    for q in queries:
        per_query[q["query"]].add(frozenset(s["file_path"] for s in q["snippets"]))
    clashes = [text for text, sets in per_query.items() if len(sets) > 1]
    if clashes:
        raise SystemExit(f"{len(clashes)} query texts point at different documents, "
                         f"e.g. {clashes[0][:120]!r}. Lookup by text would be wrong.")

    for name, question_only in [("routed_full", False), ("routed_question", True)]:
        print(f"\nRunning '{name}' on {len(queries)} queries...")
        retrieve = make_routed_retriever(store, gold_files_by_query, question_only)
        print_table(run(name, retrieve, queries, by_file))


if __name__ == "__main__":
    main()
