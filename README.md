# Legal contract RAG

Ask a question about a legal contract and get an answer that cites the exact passages it came from. It runs entirely on a laptop (Ollama + Chroma), and every design choice is measured on [LegalBench-RAG](https://github.com/zeroentropy-cc/legalbenchrag), a benchmark of expert-labelled questions over real contracts and privacy policies.

## At a glance

- **Finds the right passage:** for 77% of 743 benchmark questions, the expert's answer is in the top 5 passages returned (up from 47% for plain vector search).
- **Answers with citations:** every answer points to the passages it used, with their exact position in the contract.
- **Measured, not guessed:** each change was kept or rejected based on its numbers, including two that were rejected (a reranker and a stricter prompt).
- **Honest about bugs:** an independent code review found two evaluation bugs; both were fixed with regression tests, and all numbers below are from after the fixes.

## How it works

1. **Pick the contract.** The user types company names; BM25 keyword search over each document's name and opening lines finds the contract.
2. **Search inside it.** Meaning-based search (mxbai-embed-large embeddings) and keyword search (BM25) each rank the contract's passages; the two rankings are merged with Reciprocal Rank Fusion.
3. **Answer.** Llama 3.2 3B reads the top 5 passages and answers with citations like [C2], or says NOT FOUND.

Code: `rag/` (the system), `evals/` (one script per experiment), `tests/`, `main.py` (command-line app).

## Results

**Retrieval** on 743 questions (hit@5 = share of questions where the expert's answer is in the top 5 passages):

| System | hit@1 | hit@5 | p50 latency |
| --- | --- | --- | --- |
| Vector search over all contracts | 0.178 | 0.468 | 137 ms |
| + pick the contract first, then search inside it | 0.423 | 0.755 | 113 ms |
| **+ keyword and vector search combined (current)** | **0.464** | **0.771** | **111 ms** |
| + reranker model (not adopted: +0.01 hit@5 for 711 ms more per question) | 0.467 | 0.781 | 711 ms* |

\*Reranking step alone. Full metrics (MRR, nDCG, per dataset) are in `evals/results/*.summary.json`.

**Answers**, 40 test questions graded by hand against the expert answers:

| Prompt | Correct | Partly | Wrong |
| --- | --- | --- | --- |
| v1: quote the contract or say NOT FOUND | 14 | 1 | 25 |
| **v2: may also answer Yes / No (current)** | **21** | **9** | **10** |

- v1 refused too often: 21 NOT FOUNDs, 14 of them with the answer in front of it. v2 was tuned on 40 separate development questions, never on the graded ones.
- Of v2's 19 imperfect answers, 14 had the answer in their passages, so the main limit is now the small 3B model, not retrieval.

Caveats: 33 of the benchmark's 776 questions were dropped because 3 file names contain `|`, which Windows can't store; v1's grades were not audited the way v2's were; and splitting each benchmark question at `;` into contract and question is specific to the benchmark (in the app, the user picks the contract).

## Key decisions

- **Pick the contract before searching.** Searching all contracts at once mixes in look-alike clauses from other contracts; picking the contract first was the single biggest gain (hit@5 0.485 → 0.755).
- **Merge rankings, not scores.** Vector and keyword scores are on different scales, so they're merged by rank (RRF), with the standard constant k=60 left untuned to avoid tuning on the test set.
- **Own chunker.** LangChain's splitter can record the wrong position for repeated text, which makes scoring wrong; `rag/chunking.py` records positions while it cuts.
- **Hand grading and a separate development set.** Prompts were changed using different questions from the ones graded, so the v2 result isn't fitted to its own test.

## Run it

```bash
pip install -r requirements.txt
ollama pull mxbai-embed-large && ollama pull llama3.2
# put the LegalBench-RAG corpus/ and benchmarks/ folders under data/
python -m rag.ingest          # select 743 questions, chunk 69 documents into 5,845 chunks
python -m rag.build_index     # embed the chunks into Chroma (slow on CPU)
python main.py                # pick a contract by company name, then ask questions
python -m pytest tests -q     # 37 tests, no Ollama needed
python -m evals.eval_hybrid   # reproduce the retrieval results
```

## Next steps

- Put it behind an API with a live demo.
- Decide NOT FOUND from retrieval scores instead of asking the model.
- Try a larger model (e.g. llama3.1:8b).
- Build an agent on top

## How this was built

Developed with AI assistance. Claude (Anthropic) wrote and reviewed much of the code. I directed the project, ran and checked every experiment, graded the 40 answers by hand, and decided what to adopt or reject based on the results. An independent AI code review found the line-ending and nDCG bugs, which were fixed with regression tests.
