"""
Step 3a: document router.

Given the document part of a query ("Consider the Merger Agreement between
'Avangrid, Inc.' and 'PNM Resources, Inc.'"), rank the documents in the corpus
by how likely the query is about them.

How: build a short "card" for each document (its file name plus the opening
~2,000 characters, where contracts name their parties), then score cards
against the query title with BM25. Company names are rare words, and BM25
weights rare words heavily, which is exactly what we want here.

No embeddings, no Ollama: builds in well under a second for 69 documents.
"""

import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

CARD_CHARS = 2000

# Words that appear in nearly every query title and carry no information
# about WHICH document is meant.
STOPWORDS = {
    "the", "a", "an", "and", "of", "to", "in", "for", "on", "by", "with", "or",
    "between", "consider", "this", "that", "is", "are", "be", "as", "its", "s",
}


def tokenize(text: str) -> list[str]:
    # [a-z0-9]+ instead of \w+ because \w includes "_", and file names like
    # "Inc._Avangrid" would otherwise produce the token "_avangrid".
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def split_query(text: str) -> tuple[str, str]:
    """'Consider the X; What is Y' -> ('Consider the X', 'What is Y').

    This is how LegalBench-RAG formats queries. In a real app the document
    would come from the user (a dropdown or a document_id), not from parsing.
    """
    title, sep, question = text.partition(";")
    return (title.strip(), question.strip()) if sep else ("", text.strip())


def file_name_text(file_path: str) -> str:
    name = file_path.split("/", 1)[-1]
    name = re.sub(r"\.(txt|pdf)", " ", name, flags=re.IGNORECASE)
    return name.replace("_", " ")


def document_opening(file_chunks: list[dict], n_chars: int) -> str:
    """Rebuild the first n_chars of a document from its chunks.

    Neighbouring chunks overlap by ~200 chars, so naively joining them would
    repeat text (and inflate BM25 term counts). Only append the part of each
    chunk that starts after what we already have.
    """
    parts, covered = [], 0
    for c in sorted(file_chunks, key=lambda c: c["start"]):
        if c["start"] >= n_chars:
            break
        if c["start"] > covered and parts:
            # The splitter drops the whitespace it split on, so there can be a
            # gap between chunks. Without a separator, "...and" + "the..."
            # becomes "andthe" and BM25 sees a word that doesn't exist.
            parts.append("\n")
        parts.append(c["text"][max(0, covered - c["start"]):])
        covered = c["end"]
    return "".join(parts)[:n_chars]


class DocumentRouter:
    def __init__(self, chunks: list[dict], card_chars: int = CARD_CHARS, unique_terms: bool = False):
        """unique_terms=True counts each word once per card. For routing we care
        WHETHER a card mentions "23andMe", not how often it says "privacy
        policy"; with raw counts, repeated common words can outvote a rare
        name that appears once."""
        by_file = defaultdict(list)
        for c in chunks:
            by_file[c["file_path"]].append(c)

        self.files = sorted(by_file)
        cards = [
            file_name_text(fp) + "\n" + document_opening(by_file[fp], card_chars)
            for fp in self.files
        ]
        tokenized = [tokenize(card) for card in cards]
        if unique_terms:
            tokenized = [list(dict.fromkeys(toks)) for toks in tokenized]
        self.bm25 = BM25Okapi(tokenized)

    def explain(self, title: str, top_n: int = 3) -> list[tuple[str, float, dict]]:
        """Break each top document's score into per-word contributions:
        {word: (times on card, points added)}. For debugging misroutes."""
        bm, out = self.bm25, []
        for fp, total in self.route(title, top_n):
            i = self.files.index(fp)
            norm = bm.k1 * (1 - bm.b + bm.b * bm.doc_len[i] / bm.avgdl)
            parts = {}
            for term in tokenize(title):
                tf = bm.doc_freqs[i].get(term, 0)
                idf = bm.idf.get(term, 0.0)
                parts[term] = (tf, idf * tf * (bm.k1 + 1) / (tf + norm) if tf else 0.0)
            out.append((fp, total, parts))
        return out

    def route(self, title: str, top_n: int = 1) -> list[tuple[str, float]]:
        """Return the top_n (file_path, score) pairs, best first."""
        scores = self.bm25.get_scores(tokenize(title))
        ranked = sorted(zip(self.files, scores), key=lambda pair: -pair[1])
        return ranked[:top_n]
