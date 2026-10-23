# Legal contract RAG with measured retrieval

Ask questions about a legal contract and get answers that cite the exact passages they came from. Runs fully locally (Ollama + Chroma). Every retrieval change is measured on [LegalBench-RAG](https://github.com/zeroentropy-cc/legalbenchrag), a benchmark of expert-labelled questions over real contracts and privacy policies.

> Status: work in progress. Retrieval is done; answer evaluation, API and deployment are next.

## Results so far

743 questions from the LegalBench-RAG mini set (776; 33 dropped because 3 MAUD file names contain `|`, which Windows can't store). A retrieved chunk counts as relevant if it overlaps the expert's answer span in the same file.

| System | hit@1 | hit@5 | MRR@10 | p50 latency |
| --- | --- | --- | --- | --- |
| Dense search (mxbai-embed-large, Chroma) | 0.167 | 0.447 | 0.284 | |
| + query prefix the embedding model expects | 0.189 | 0.468 | 0.302 | |
| + BM25 router picks the contract, search inside it | 0.383 | 0.694 | 0.522 | |
| **+ hybrid search (BM25 + dense, RRF), current system** | **0.409** | **0.705** | **0.534** | **104 ms** |
| + cross-encoder rerank (MiniLM, pool 20), not adopted | 0.413 | 0.718 | 0.541 | 915 ms |

What didn't work, and why it's documented: the reranker's gain is within noise (47 questions gained, 38 lost) at about 9x the latency. The router misses one document (23andMe, 13 questions) because its opening never uses the words "privacy" or "policy".

Splitting the benchmark question at `;` into a contract title and a question is benchmark-specific. In the app (`main.py`), the user picks the contract instead.

## Layout

```
rag/      the system: ingest, index, router, hybrid search, rerank, answer
evals/    one script per experiment; summaries in evals/results/
tests/    scoring and retrieval logic, no Ollama needed
main.py   ask questions about a contract from the command line
```

## Run it

```bash
pip install -r requirements.txt
ollama pull mxbai-embed-large && ollama pull llama3.2
# put the LegalBench-RAG corpus/ and benchmarks/ folders under data/
python -m rag.ingest          # select 743 questions, chunk 69 documents
python -m rag.build_index     # embed 7,059 chunks into Chroma (slow on CPU)
python main.py                # ask questions
python -m pytest tests -q     # tests
python -m evals.eval_hybrid   # reproduce the main results row
```
