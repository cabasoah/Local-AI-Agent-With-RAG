"""
Step 1b: embed the chunks from ingest.py and store them in Chroma.

Run once after ingest.py. Re-run with --rebuild whenever chunks.jsonl changes,
otherwise you'll be evaluating an old index without noticing.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DB_DIR = Path("chroma_db")
COLLECTION = "legalbench_mini"
EMBED_MODEL = "mxbai-embed-large"
BATCH_SIZE = 128


def get_vector_store() -> Chroma:
    """Shared by this script and (later) the retriever + eval code."""
    return Chroma(
        collection_name=COLLECTION,
        persist_directory=str(DB_DIR),
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
        collection_metadata={"hnsw:space": "cosine"},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="delete and rebuild the index")
    args = parser.parse_args()

    if args.rebuild and DB_DIR.exists():
        shutil.rmtree(DB_DIR)
    elif DB_DIR.exists():
        print(f"{DB_DIR} already exists. Use --rebuild to re-embed.")
        return

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    docs = [
        Document(
            page_content=c["text"],
            metadata={k: c[k] for k in ("file_path", "dataset", "start", "end")},
            id=c["id"],
        )
        for c in chunks
    ]

    store = get_vector_store()
    t0 = time.perf_counter()
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        store.add_documents(batch, ids=[d.id for d in batch])
        print(f"  embedded {min(i + BATCH_SIZE, len(docs))}/{len(docs)}", end="\r")

    elapsed = time.perf_counter() - t0
    print(f"\nindexed {len(docs)} chunks in {elapsed:.0f}s ({len(docs) / elapsed:.1f} chunks/s)")


if __name__ == "__main__":
    main()