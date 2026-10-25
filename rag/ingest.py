"""
Step 1a: select the LegalBench-RAG mini benchmark and chunk its documents.

Input:
    data/corpus/<dataset>/*.txt        raw contract / policy text files
    data/benchmarks/<dataset>.json     queries + ground-truth character spans

Output:
    data/processed/chunks.jsonl        one line per chunk, with character offsets
    data/processed/legalbench_mini.jsonl  the queries we evaluate against

No embeddings happen here. That's build_index.py, so changing the chunk
settings and re-running this script takes seconds, not an hour.
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from rag.chunking import split_with_offsets

DATA_DIR = Path("data")
CORPUS_DIR = DATA_DIR / "corpus"
BENCHMARK_DIR = DATA_DIR / "benchmarks"
CHUNKS_OUT = DATA_DIR / "processed" / "chunks.jsonl"
QUERIES_OUT = DATA_DIR / "processed" / "legalbench_mini.jsonl"

DATASETS = ["privacy_qa", "contractnli", "maud", "cuad"]
MAX_QUERIES_PER_DATASET = 194  # 4 x 194 = 776, the paper's "mini" set


# ---------------------------------------------------------------------------
# 1. Pick the mini set of queries (same logic as the official repo)
# ---------------------------------------------------------------------------
def select_mini_queries(dataset: str) -> list[dict]:
    with open(BENCHMARK_DIR / f"{dataset}.json", encoding="utf-8") as f:
        tests = json.load(f)["tests"]

    if len(tests) > MAX_QUERIES_PER_DATASET:
        # Each query gets a pseudo-random number seeded by its document's path.
        # Queries about the same document get the same number, so they sort
        # next to each other -> the 194 we keep cover as few documents as
        # possible. This matches benchmark.py in zeroentropy-cc/legalbenchrag.
        def doc_seeded_key(test: dict) -> float:
            random.seed(test["snippets"][0]["file_path"])
            return random.random()

        tests = sorted(tests, key=doc_seeded_key)[:MAX_QUERIES_PER_DATASET]

    return [
        {
            "id": f"{dataset}-{i:03d}",
            "dataset": dataset,
            "query": t["query"],
            # ground truth: list of {"file_path": ..., "span": [start, end]}
            "snippets": t["snippets"],
        }
        for i, t in enumerate(tests)
    ]


# ---------------------------------------------------------------------------
# 1b. Find each document on disk, even if the OS renamed it
# ---------------------------------------------------------------------------
# Some MAUD files are named "A.pdf||A Amendment No. 1.txt". Windows forbids
# "|" in file names, so the download tool renames or drops those files. We
# match on a "key" made of only letters and digits, so "||", "__" or a
# removed "|" all compare equal. The benchmark's ORIGINAL file_path is still
# what we store in chunks, so it lines up with the ground truth in step 2.
def name_key(rel_path: str) -> str:
    return re.sub(r"[^0-9a-z]", "", rel_path.lower())


def build_disk_lookup() -> dict[str, list[Path]]:
    lookup: dict[str, list[Path]] = {}
    for p in CORPUS_DIR.rglob("*.txt"):
        rel = p.relative_to(CORPUS_DIR).as_posix()
        lookup.setdefault(name_key(rel), []).append(p)
    return lookup


def resolve_on_disk(file_path: str, lookup: dict[str, list[Path]]) -> Path | None:
    exact = CORPUS_DIR / file_path
    if "|" not in file_path and exact.exists():
        return exact
    matches = lookup.get(name_key(file_path), [])
    return matches[0] if len(matches) == 1 else None  # 0 or ambiguous -> missing


def read_document(path: Path) -> str:
    # Plain open() turns "\r\n" into "\n", exactly as the benchmark's own code
    # does when it computes the gold offsets (legalbenchrag/benchmark_types.py).
    # Keeping "\r" (newline="") shifted every MAUD span: 15 MAUD files use
    # "\r\n", so offsets counted with "\r" pointed at the wrong text.
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# 2. Chunk one document, keeping each chunk's exact position in the file
# ---------------------------------------------------------------------------
def check_full_coverage(file_path: str, text: str, chunks: list[dict]) -> None:
    """Every non-whitespace character must belong to some chunk.
    The splitter never drops text, so a gap means a chunk was misplaced."""
    covered = 0
    for c in sorted(chunks, key=lambda c: c["start"]):
        if c["start"] > covered and text[covered:c["start"]].strip():
            raise ValueError(f"Text at {covered}-{c['start']} of {file_path} is in no chunk")
        covered = max(covered, c["end"])
    if text[covered:].strip():
        raise ValueError(f"Text after {covered} of {file_path} is in no chunk")


def chunk_document(file_path: str, text: str, size: int, overlap: int) -> list[dict]:
    chunks = [
        {
            "id": f"{file_path}::{start}",   # stable, readable ID
            "file_path": file_path,
            "dataset": file_path.split("/")[0],
            "start": start,
            "end": end,
            "text": text[start:end],
        }
        for start, end in split_with_offsets(text, size, overlap)
    ]
    check_full_coverage(file_path, text, chunks)
    return chunks


def gold_alignment_report(queries: list[dict], texts: dict[str, str]) -> None:
    """Share of gold spans that start or end in the middle of a word, per dataset.

    Expert answers start and end on word boundaries. If many spans cut words
    in half, the offsets don't match the text we read (this is how the
    "\r\n" bug in MAUD showed up: 159 of 258 spans started mid-word).
    """
    cut = Counter()
    total = Counter()
    for q in queries:
        for s in q["snippets"]:
            text = texts[s["file_path"]]
            a, b = s["span"]
            mid_start = 0 < a < len(text) and text[a - 1].isalnum() and text[a].isalnum()
            mid_end = 0 < b < len(text) and text[b - 1].isalnum() and text[b].isalnum()
            total[q["dataset"]] += 1
            cut[q["dataset"]] += mid_start or mid_end
    for ds in sorted(total):
        share = cut[ds] / total[ds]
        flag = "   <-- WARNING: offsets probably don't match the text" if share > 0.05 else ""
        print(f"gold spans cutting a word, {ds}: {cut[ds]}/{total[ds]} ({share:.1%}){flag}")


def gold_span_coverage(queries: list[dict], chunks: list[dict]) -> float:
    """Fraction of ground-truth spans that sit fully inside at least one chunk.

    A low number means chunks are cutting answers in half, which caps how
    well ANY retriever can score. Good to know before blaming the retriever.
    """
    by_file: dict[str, list[tuple[int, int]]] = {}
    for c in chunks:
        by_file.setdefault(c["file_path"], []).append((c["start"], c["end"]))

    total = inside = 0
    for q in queries:
        for s in q["snippets"]:
            g_start, g_end = s["span"]
            total += 1
            if any(cs <= g_start and g_end <= ce for cs, ce in by_file[s["file_path"]]):
                inside += 1
    return inside / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=1500)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()

    # --- queries ---
    queries = [q for ds in DATASETS for q in select_mini_queries(ds)]
    needed_files = sorted({s["file_path"] for q in queries for s in q["snippets"]})

    # --- locate files; drop queries whose documents aren't on disk ---
    lookup = build_disk_lookup()
    disk_path = {fp: resolve_on_disk(fp, lookup) for fp in needed_files}
    missing = {fp for fp, p in disk_path.items() if p is None}
    if missing:
        before = len(queries)
        queries = [q for q in queries if not any(s["file_path"] in missing for s in q["snippets"])]
        needed_files = [fp for fp in needed_files if fp not in missing]
        print(f"WARNING: {len(missing)} documents not found on disk; "
              f"dropped {before - len(queries)} queries that need them:")
        for fp in sorted(missing):
            print(f"   - {fp}")

    # --- chunking ---
    chunks, texts = [], {}
    for fp in needed_files:
        texts[fp] = read_document(disk_path[fp])
        chunks.extend(chunk_document(fp, texts[fp], args.chunk_size, args.chunk_overlap))
    total_chars = sum(len(t) for t in texts.values())

    # --- write outputs ---
    CHUNKS_OUT.parent.mkdir(parents=True, exist_ok=True)
    QUERIES_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_OUT, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    with open(QUERIES_OUT, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    # --- report ---
    print(f"queries:   {len(queries)}  {dict(Counter(q['dataset'] for q in queries))}")
    print(f"documents: {len(needed_files)}  ({total_chars:,} chars)")
    print(f"chunks:    {len(chunks)}  {dict(Counter(c['dataset'] for c in chunks))}")
    print(f"avg chunk: {sum(len(c['text']) for c in chunks) / len(chunks):.0f} chars")
    print(f"gold spans fully inside one chunk: {gold_span_coverage(queries, chunks):.1%}")
    gold_alignment_report(queries, texts)


if __name__ == "__main__":
    main()
