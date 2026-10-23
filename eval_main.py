"""
Step 6: evaluate the generated answers.

Retrieval scores say whether the right text reached the model. These say what
the model DID with it. Two kinds of check:

1. Automatic, from the citations (no judge model needed):
   - did it answer, or say NOT FOUND?
   - did it cite at all, and cite passages that exist?
   - citation precision: share of cited passages that overlap the expert answer
   - citation hit: at least one cited passage overlaps the expert answer
   - false refusals: NOT FOUND although the answer WAS in the passages
2. By hand: a grading sheet (CSV) with question, expert answer and model answer.
   You fill in the `grade` column: y (correct), p (partly), n (wrong).
   Then: python eval_answers.py --summarize evals/grading_sheet.csv

Questions are a random sample, fixed by --seed, so the same set comes back on
every run. The default (10 per dataset) keeps hand-grading to ~40 answers.

"""

import argparse
import csv
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from rag.answer import OllamaGenerator, answer_question
from evals.eval_retrieval import CHUNKS_PATH, QUERIES_PATH, RESULTS_DIR, chunk_hits_span, load_jsonl
from rag.hybrid import HybridRetriever
from rag.ingest import build_disk_lookup, read_exact, resolve_on_disk
from rag.router import DocumentRouter, split_query

SHEET_PATH = Path("evals/grading_sheet.csv")
GRADES = {"y": "correct", "p": "partly", "n": "wrong"}


def sample_questions(queries: list[dict], per_dataset: int, seed: int) -> list[dict]:
    by_ds = defaultdict(list)
    for q in queries:
        by_ds[q["dataset"]].append(q)
    rng = random.Random(seed)
    picked = []
    for ds in sorted(by_ds):
        picked.extend(rng.sample(by_ds[ds], min(per_dataset, len(by_ds[ds]))))
    return picked


def report(rows: list[dict]) -> None:
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    by_ds["ALL"] = rows

    print("| dataset | n | retrieval hit@k | answered | NOT FOUND | false refusals | "
          "cited (of answered) | citation precision | citation hit | invalid labels | p50 gen s |")
    print("|---" * 11 + "|")
    for ds, rs in by_ds.items():
        answered = [r for r in rs if not r["not_found"]]
        cited = [r for r in answered if r["cited_ids"]]
        precision = statistics.mean(r["citation_precision"] for r in cited) if cited else 0.0
        false_ref = sum(r["not_found"] and r["retrieval_hit"] for r in rs)
        label = f"**{ds}**" if ds == "ALL" else ds
        print(f"| {label} | {len(rs)} | {sum(r['retrieval_hit'] for r in rs) / len(rs):.2f} | "
              f"{len(answered)} | {len(rs) - len(answered)} | {false_ref} | "
              f"{len(cited)}/{len(answered)} | {precision:.2f} | "
              f"{sum(r['citation_hit'] for r in answered) / max(len(answered), 1):.2f} | "
              f"{sum(len(r['invalid_labels']) for r in rs)} | "
              f"{statistics.median(r['gen_ms'] for r in rs) / 1000:.1f} |")


def summarize_grades(path: Path) -> None:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    graded = [r for r in rows if r["grade"].strip().lower() in GRADES]
    print(f"{len(graded)} of {len(rows)} answers graded.")
    if not graded:
        return

    by_ds = defaultdict(Counter)
    for r in graded:
        by_ds[r["dataset"]][r["grade"].strip().lower()] += 1
    by_ds = dict(sorted(by_ds.items()))
    by_ds["ALL"] = sum(by_ds.values(), Counter())
    print("\n| dataset | graded | correct | partly | wrong | correct rate |")
    print("|---|---|---|---|---|---|")
    for ds, c in by_ds.items():
        n = sum(c.values())
        print(f"| {ds} | {n} | {c['y']} | {c['p']} | {c['n']} | {c['y'] / n:.2f} |")

    # Was a wrong answer the retriever's fault or the model's?
    cross = Counter((r["retrieval_hit"], r["grade"].strip().lower()) for r in graded)
    print("\nWrong or partly-right answers, by whether retrieval found the answer:")
    print(f"  retrieval HAD the answer:   {cross[('True', 'n')]} wrong, {cross[('True', 'p')]} partly  -> generation problem")
    print(f"  retrieval MISSED the answer: {cross[('False', 'n')]} wrong, {cross[('False', 'p')]} partly  -> retrieval problem")


def main(argv=None, store=None, generate=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-dataset", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=5, help="passages given to the model")
    p.add_argument("--model", default="llama3.2")
    p.add_argument("--sheet", type=Path, default=SHEET_PATH)
    p.add_argument("--overwrite", action="store_true", help="replace an existing grading sheet")
    p.add_argument("--summarize", type=Path, metavar="SHEET", help="report on a graded sheet, then exit")
    args = p.parse_args(argv)

    if args.summarize:
        summarize_grades(args.summarize)
        return
    if args.sheet.exists() and not args.overwrite:
        raise SystemExit(f"{args.sheet} already exists and may hold your grades. "
                         f"Use --sheet another.csv, or --overwrite to replace it.")

    queries = sample_questions(load_jsonl(QUERIES_PATH), args.per_dataset, args.seed)
    chunks = load_jsonl(CHUNKS_PATH)
    text = {c["id"]: c["text"] for c in chunks}
    router = DocumentRouter(chunks)
    if store is None:
        from build_index import get_vector_store
        store = get_vector_store()
    retriever = HybridRetriever(chunks, store, router, mode="hybrid")
    generate = generate or OllamaGenerator(args.model)

    lookup, doc_cache = build_disk_lookup(), {}

    def gold_text(q: dict) -> str:
        parts = []
        for s in q["snippets"]:
            fp = s["file_path"]
            if fp not in doc_cache:
                doc_cache[fp] = read_exact(resolve_on_disk(fp, lookup))
            parts.append(doc_cache[fp][s["span"][0]:s["span"][1]].strip())
        return " ... ".join(parts)

    rows = []
    for i, q in enumerate(queries, 1):
        title, question = split_query(q["query"])
        file_path = router.route(title or q["query"], top_n=1)[0][0]
        found = [dict(c, text=text[c["id"]]) for c in retriever.search(question, file_path, args.k)]

        t0 = time.perf_counter()
        result = answer_question(question, found, generate)
        gen_ms = (time.perf_counter() - t0) * 1000

        relevant = lambda c: any(chunk_hits_span(c, g) for g in q["snippets"])
        cited = [found[n - 1] for n in result["cited_labels"]]
        rows.append({
            "id": q["id"], "dataset": q["dataset"], "question": question,
            "routed_to": file_path,
            "retrieval_hit": any(relevant(c) for c in found),
            **result,
            "citation_precision": sum(map(relevant, cited)) / len(cited) if cited else 0.0,
            "citation_hit": any(relevant(c) for c in cited),
            "gen_ms": gen_ms,
            "gold": gold_text(q),
        })
        print(f"  {i}/{len(queries)}", end="\r", flush=True)
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"answers_seed{args.seed}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # utf-8-sig so Excel on Windows shows quotes and accents correctly.
    args.sheet.parent.mkdir(parents=True, exist_ok=True)
    with open(args.sheet, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "dataset", "question", "expert_answer", "model_answer",
                    "retrieval_hit", "citation_hit", "grade", "notes"])
        for r in rows:
            w.writerow([r["id"], r["dataset"], r["question"], r["gold"][:1500], r["answer"],
                        r["retrieval_hit"], r["citation_hit"], "", ""])

    report(rows)
    print(f"\nSaved {out} and {args.sheet}")
    print("Fill the `grade` column with y / p / n, then: "
          f"python eval_answers.py --summarize {args.sheet}")


if __name__ == "__main__":
    main()