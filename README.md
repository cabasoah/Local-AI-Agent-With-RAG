# Legal contract RAG with measured retrieval

Ask questions about a legal contract and get answers that cite the exact passages they came from. Runs fully locally (Ollama + Chroma). Every change is measured on [LegalBench-RAG](https://github.com/zeroentropy-cc/legalbenchrag), a benchmark of expert-labelled questions over real contracts and privacy policies.

> Status: work in progress. Retrieval is final; the next steps are fixing answer over-refusal, then an API and deployment.

## Results

743 questions from the LegalBench-RAG mini set (776; 33 dropped because 3 MAUD file names contain `|`, which Windows can't store). A retrieved chunk counts as relevant if it overlaps the expert's answer span in the same file.

| System | hit@1 | hit@5 | MRR@10 | nDCG@10 | p50 latency |
| --- | --- | --- | --- | --- | --- |
| Dense search (mxbai-embed-large, Chroma) | 0.178 | 0.468 | 0.297 | 0.317 | 137 ms |
| + query prefix the embedding model expects | 0.198 | 0.485 | 0.316 | 0.331 | 162 ms |
| + BM25 router picks the contract, search inside it | 0.423 | 0.755 | 0.567 | 0.573 | 113 ms |
| **+ hybrid search (BM25 + dense, RRF): current system** | **0.464** | **0.771** | **0.594** | **0.600** | **111 ms** |
| + cross-encoder rerank (MiniLM, pool 20): not adopted | 0.467 | 0.781 | 0.596 | 0.610 | 902 ms |

**Answers** (Llama 3.2 3B, 40 questions, graded by hand against the expert answers): 14 correct, 1 partly right. In 18 of the 26 failures the answer was among the retrieved passages, so generation is now the bottleneck: the model said NOT FOUND 21 times, 14 of them with the answer in front of it.

**Reading the numbers honestly**

- Hybrid beats dense on hit@5 by 48 questions gained to 37 lost: a lean, not proof. The router's gain is unmistakable.
- The reranker's gain is within noise (48 gained, 40 lost) at about 8x the latency, so it was not adopted. A larger reranker (bge-reranker-base) took about 5 s per question on CPU in a 40-question sample.
- Some hit@5 scores have a high floor: ContractNLI documents are short, so even 5 random chunks from the right contract often contain the answer. MAUD's merger agreements are long, which makes MAUD the meaningful test; it reaches 0.615.
- The router misses one document (23andMe, 13 questions): its opening never uses the words "privacy" or "policy".
- Splitting each benchmark question at `;` into a contract title and a question is benchmark-specific. In the app (`main.py`), the user picks the contract.

**Bugs found and fixed** (by an independent code review, then fixed with regression tests)

- Files were read without converting `\r\n` to `\n`, unlike the benchmark, so every answer span in 15 MAUD files pointed at shifted text. `rag.ingest` now warns when answer spans start or end mid-word.
- LangChain's splitter locates chunks by searching for their text, which can land on an earlier copy of repeated text. Replaced with `rag/chunking.py`, which records positions while cutting (checked on 50,000 random documents).
- nDCG's ideal ranking counted a chunk twice when it overlapped two answer spans.

All numbers above are from after these fixes.

## Layout

```
rag/      the system: ingest, chunking, index, router, hybrid search, rerank, answer
evals/    one script per experiment; summaries in evals/results/
tests/    scoring, chunking and retrieval logic, no Ollama needed
main.py   ask questions about a contract from the command line
```

## Run it

```bash
pip install -r requirements.txt
ollama pull mxbai-embed-large && ollama pull llama3.2
# put the LegalBench-RAG corpus/ and benchmarks/ folders under data/
python -m rag.ingest          # select 743 questions, chunk 69 documents into 5,845 chunks
python -m rag.build_index     # embed the chunks into Chroma (slow on CPU)
python main.py                # ask questions
python -m pytest tests -q     # tests
python -m evals.eval_hybrid   # reproduce the main results rows
```