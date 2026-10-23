"""
Step 6: turn retrieved chunks into a cited answer (augment + generate).

Design choices, each for a reason:
- Passages are labelled [C1]..[C5], not with chunk IDs like
  "maud/PNM Resources, Inc._Avangrid, Inc..txt::48200". A 3B model copies a
  short label reliably and garbles a long one. The code maps labels back to
  chunk IDs, so citations can still be checked against the answer key.
- temperature=0 and a fixed seed: the same question gives the same answer, so
  an eval run can be repeated and compared.
- num_ctx=8192: Ollama's default context window can be smaller than five
  chunks plus instructions. When a prompt doesn't fit, Ollama drops the
  START of it, which is where the rules are, and does so without an error.
"""

import re

NOT_FOUND = "NOT FOUND"
LABEL_RE = re.compile(r"\[C(\d+)\]")

PROMPT = """You answer questions about a legal contract using ONLY the numbered passages below.

Rules:
1. After each sentence, cite the passage number(s) it comes from, like [C2] or [C1][C3].
2. Quote the contract's exact words where you can.
3. If the passages do not contain the answer, reply with exactly: NOT FOUND
4. Do not use any knowledge from outside the passages.

Passages:

{passages}

Question: {question}
Answer:"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    passages = "\n\n".join(f"[C{i}]\n{c['text'].strip()}" for i, c in enumerate(chunks, 1))
    return PROMPT.format(passages=passages, question=question)


def parse_answer(raw: str, chunks: list[dict]) -> dict:
    text = raw.strip()
    labels = list(dict.fromkeys(int(n) for n in LABEL_RE.findall(text)))  # unique, in order
    valid = [n for n in labels if 1 <= n <= len(chunks)]
    return {
        "answer": text,
        # Counted as a refusal only if it says NOT FOUND and cites nothing; an
        # answer that cites passages and mentions "not found" in passing is an answer.
        "not_found": NOT_FOUND in text.upper() and not valid,
        "cited_labels": valid,
        "cited_ids": [chunks[n - 1]["id"] for n in valid],
        # A label that wasn't in the prompt, e.g. [C9] when only 5 passages were given.
        "invalid_labels": [n for n in labels if n not in valid],
    }


class OllamaGenerator:
    def __init__(self, model: str = "llama3.2", num_ctx: int = 8192):
        from langchain_ollama.llms import OllamaLLM  # imported here so tests don't need Ollama

        self.num_ctx = num_ctx
        self.llm = OllamaLLM(model=model, temperature=0, seed=0, num_ctx=num_ctx)

    def __call__(self, prompt: str) -> str:
        # Rough guard: ~4 characters per token for English.
        if len(prompt) / 4 > 0.9 * self.num_ctx:
            print(f"  warning: prompt is ~{len(prompt) // 4} tokens, close to num_ctx={self.num_ctx}")
        return self.llm.invoke(prompt)


def answer_question(question: str, chunks: list[dict], generate) -> dict:
    """chunks need 'id' and 'text'. generate: any function prompt -> text."""
    return parse_answer(generate(build_prompt(question, chunks)), chunks)
