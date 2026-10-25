"""
Do the question's words actually appear in the answer?

Keyword search (BM25) can only find an answer that shares words with the
question. This measures, per dataset, what share of each question's words
appear in the expert answer span. Low overlap = vocabulary mismatch, a
problem keyword search can't fix but meaning-based search can.

No Ollama needed; runs in seconds.
Usage: python -m evals.vocab_overlap [--examples maud]
"""

import argparse
import statistics
from collections import defaultdict

from evals.eval_retrieval import QUERIES_PATH, load_jsonl
from rag.hybrid import tokenize
from rag.ingest import build_disk_lookup, read_document, resolve_on_disk
from rag.router import split_query


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--examples", default="maud", help="dataset to print examples for")
    args = p.parse_args(argv)

    lookup = build_disk_lookup()
    texts: dict[str, str] = {}

    def doc_text(fp: str) -> str:
        if fp not in texts:
            texts[fp] = read_document(resolve_on_disk(fp, lookup))
        return texts[fp]

    overlap = defaultdict(list)
    examples = defaultdict(list)
    for q in load_jsonl(QUERIES_PATH):
        q_words = set(tokenize(split_query(q["query"])[1]))
        if not q_words:
            continue
        answer = " ".join(doc_text(s["file_path"])[s["span"][0]:s["span"][1]] for s in q["snippets"])
        a_words = set(tokenize(answer))
        share = len(q_words & a_words) / len(q_words)
        overlap[q["dataset"]].append(share)
        examples[q["dataset"]].append((share, split_query(q["query"])[1], sorted(q_words - a_words)))

    print("Share of question words that appear in the expert answer")
    print("| dataset | n | mean | median | questions with NO shared word |")
    print("|---|---|---|---|---|")
    for ds in sorted(overlap):
        v = overlap[ds]
        print(f"| {ds} | {len(v)} | {statistics.mean(v):.2f} | {statistics.median(v):.2f} | "
              f"{sum(x == 0 for x in v) / len(v):.1%} |")

    print(f"\nLowest-overlap '{args.examples}' questions (missing words in brackets):")
    seen = set()
    for share, question, missing in sorted(examples[args.examples])[:40]:
        if question in seen:
            continue
        seen.add(question)
        print(f"  {share:.2f}  {question[:90]}  {missing}")
        if len(seen) == 5:
            break


if __name__ == "__main__":
    main()
