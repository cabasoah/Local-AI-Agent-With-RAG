import json
from collections import Counter, defaultdict

N_ROWS = 123
ALLOWED = {"lookup", "multi", "unanswerable", "aggregate"}

seen, counts, by_set = set(), Counter(), defaultdict(list)

with open("evals/questions.jsonl", encoding="utf-8") as f:
    for n, line in enumerate(f, 1):
        if not line.strip():
            continue
        try:
            q = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"line {n}: invalid JSON ({e})")
            continue

        qid, qtype = q.get("id"), q.get("type")
        counts[qtype] += 1
        if qid in seen:
            print(f"line {n}: duplicate id {qid}")
        seen.add(qid)
        if qtype not in ALLOWED:
            print(f"{qid}: unknown type {qtype!r}")

        ids = q.get("relevant_row_ids", q.get("evidence_row_ids", []))
        if len(ids) != len(set(ids)):
            print(f"{qid}: duplicate row ids in list")
        if any(not 0 <= i < N_ROWS for i in ids):
            print(f"{qid}: row id out of range 0-{N_ROWS - 1}")
        if qtype in ("lookup", "multi") and not ids:
            print(f"{qid}: {qtype} question with no relevant rows")
        if qtype == "unanswerable" and ids:
            print(f"{qid}: unanswerable question should have no rows")
        if qtype == "lookup" and len(ids) > 3:
            print(f"{qid}: lookup with {len(ids)} rows, should this be multi?")
        if qtype in ("lookup", "multi") and len(ids) / N_ROWS > 0.10:
            print(f"{qid}: {len(ids)} rows ({len(ids) / N_ROWS:.0%} of corpus), too easy to hit by chance")
        if qtype in ("lookup", "multi") and ids:
            by_set[tuple(sorted(ids))].append(qid)

for ids, qids in by_set.items():
    if len(qids) > 1:
        print(f"identical relevant rows for {qids}: check for copy-paste")

print("\ncounts:", dict(counts))