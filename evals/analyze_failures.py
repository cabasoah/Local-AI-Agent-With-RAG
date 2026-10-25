"""
Did the retriever fail to find the right DOCUMENT, or the right SECTION of it?

Reads the per-query results that eval_retrieval.py saved, so it needs no
Ollama and runs in about a second.

"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def file_of(chunk_id: str) -> str:
    return chunk_id.rsplit("::", 1)[0]  # IDs look like "<file_path>::<start>"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="dense")
    p.add_argument("--examples", default="maud", help="dataset to print failure examples for")
    args = p.parse_args()

    queries = {q["id"]: q for q in load_jsonl("data/processed/legalbench_mini.jsonl")}
    rows = load_jsonl(Path("evals/results") / f"{args.name}.jsonl")

    stats = defaultdict(lambda: defaultdict(float))
    misses = defaultdict(list)
    for r in rows:
        q = queries[r["id"]]
        gold_files = {s["file_path"] for s in q["snippets"]}
        files = [file_of(cid) for cid in r["retrieved"]]
        top10_right_doc = sum(f in gold_files for f in files[:10])

        st = stats[r["dataset"]]
        st["n"] += 1
        st["chunk_hit@10"] += r["hit@10"]
        st["doc_hit@1"] += files[0] in gold_files if files else 0
        st["doc_hit@10"] += top10_right_doc > 0
        st["right_doc_share@10"] += top10_right_doc / max(len(files), 1)
        if top10_right_doc == 0:
            misses[r["dataset"]].append((q, files))

    print("| dataset | n | chunk hit@10 | right doc @1 | right doc in top 10 | share of top 10 from right doc |")
    print("|---|---|---|---|---|---|")
    for ds in sorted(stats):
        st = stats[ds]
        n = st["n"]
        print(f"| {ds} | {int(n)} | {st['chunk_hit@10'] / n:.3f} | {st['doc_hit@1'] / n:.3f} | "
              f"{st['doc_hit@10'] / n:.3f} | {st['right_doc_share@10'] / n:.3f} |")

    print(f"\nExample '{args.examples}' queries where NO top-10 chunk came from the right document:")
    for q, files in misses[args.examples][:3]:
        print(f"\n  QUERY: {q['query'][:200]}")
        print(f"  WANTED: {q['snippets'][0]['file_path']}")
        for f in files[:3]:
            print(f"  GOT:    {f}")


if __name__ == "__main__":
    main()
