"""
Ask questions about a contract from the command line.

    python main.py

First pick a contract (type company names, choose from the top 3 matches),
then ask questions about it. This is how a real user would use the system:
they choose the document, so nothing depends on the benchmark's
"Consider X; question" format.
"""

from rag.answer import OllamaGenerator, answer_question
from rag.build_index import get_vector_store
from evals.eval_retrieval import CHUNKS_PATH, load_jsonl
from rag.hybrid import HybridRetriever
from rag.router import DocumentRouter

TOP_K = 5


def pick_contract(router: DocumentRouter) -> str | None:
    while True:
        text = input("\nWhich contract? Type company names (q to quit): ").strip()
        if text.lower() == "q":
            return None
        options = router.route(text, top_n=3)
        for i, (fp, _) in enumerate(options, 1):
            print(f"  {i}. {fp}")
        choice = input("Pick 1-3, or press Enter to search again: ").strip()
        if choice in {"1", "2", "3"}:
            return options[int(choice) - 1][0]


def main() -> None:
    chunks = load_jsonl(CHUNKS_PATH)
    text = {c["id"]: c["text"] for c in chunks}
    router = DocumentRouter(chunks)
    retriever = HybridRetriever(chunks, get_vector_store(), router, mode="hybrid")
    generate = OllamaGenerator()

    while (file_path := pick_contract(router)) is not None:
        print(f"\nAsking about: {file_path}")
        while True:
            question = input("\nQuestion (b = other contract, q = quit): ").strip()
            if question.lower() == "q":
                return
            if question.lower() == "b" or not question:
                break
            found = [dict(c, text=text[c["id"]]) for c in retriever.search(question, file_path, TOP_K)]
            result = answer_question(question, found, generate)

            print("\n" + result["answer"])
            for n, cid in zip(result["cited_labels"], result["cited_ids"]):
                c = found[n - 1]
                print(f"\n  [C{n}] characters {c['start']:,}-{c['end']:,}: {c['text'][:160].strip()}...")
            if result["invalid_labels"]:
                print(f"\n  (cited passages that don't exist: {result['invalid_labels']})")


if __name__ == "__main__":
    main()
