import json
import statistics
import time

from vector import vector_store

QUESTIONS_PATH = "evals/questions.jsonl"
K_VALUES = [1, 3, 5, 10]


def load_questions(path=QUESTIONS_PATH):
    with open(path, encoding="utf-8") as f:
        qs = [json.loads(line) for line in f if line.strip()]
    # return [q for q in qs if q["relevant_row_ids"]]
    return [
        q for q in qs
        if q["type"] in {"lookup", "multi"} and q["relevant_row_ids"]
    ]


def evaluate(k, questions):
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    hits, recalls, rr, latencies = [], [], [], []

    for q in questions:
        relevant = set(q["relevant_row_ids"])

        start = time.perf_counter()
        docs = retriever.invoke(q["question"])
        latencies.append((time.perf_counter() - start) * 1000)

        retrieved = [d.metadata["row_id"] for d in docs]
        found = relevant & set(retrieved)

        hits.append(1.0 if found else 0.0)
        recalls.append(len(found) / len(relevant))
        rank = next((i for i, rid in enumerate(retrieved, 1) if rid in relevant), None)
        rr.append(1 / rank if rank else 0.0)

    return {
        "k": k,
        "hit": statistics.mean(hits),
        "recall": statistics.mean(recalls),
        "mrr": statistics.mean(rr),
        "latency_ms": statistics.median(latencies),
    }


if __name__ == "__main__":
    questions = load_questions()
    print(f"{len(questions)} labeled questions\n")
    print("| k | hit@k | recall@k | MRR | median retrieval ms |")
    print("|---|---|---|---|---|")
    for k in K_VALUES:
        r = evaluate(k, questions)
        print(f"| {r['k']} | {r['hit']:.2f} | {r['recall']:.2f} | {r['mrr']:.2f} | {r['latency_ms']:.0f} |")