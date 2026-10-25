"""
A text splitter that records where every chunk came from.

Why not LangChain's RecursiveCharacterTextSplitter: it returns chunk TEXT, and
its add_start_index option then guesses each position with str.find(). When
text repeats (page numbers, "____" lines, boilerplate), the guess can land on
the wrong copy, and the eval would score the wrong characters. This splitter
works on positions from start to finish, so every chunk's offsets are exact by
construction; nothing is searched for afterwards.

Same idea as the recursive splitter: cut at paragraph breaks first, then line
breaks, then spaces, and split inside a word only as a last resort; then pack
the pieces into chunks of at most `size` characters, repeating up to `overlap`
characters of the previous chunk at the start of the next.
"""

SEPARATORS = ["\n\n", "\n", " ", ""]


def _pieces(text: str, start: int, end: int, separators: list[str], size: int) -> list[tuple[int, int]]:
    """Cut text[start:end] into (start, end) pieces of at most `size` characters,
    using the coarsest separator that occurs. Each separator stays attached to
    the end of its piece, so the pieces tile the text with no gaps."""
    if end - start <= size:
        return [(start, end)]
    sep = next(s for s in separators if s == "" or text.find(s, start, end) != -1)
    finer = separators[separators.index(sep) + 1:]
    if sep == "":
        return [(i, min(i + size, end)) for i in range(start, end, size)]

    out, i = [], start
    while i < end:
        j = text.find(sep, i, end)
        piece_end = end if j == -1 else j + len(sep)
        if piece_end - i <= size:
            out.append((i, piece_end))
        else:
            out.extend(_pieces(text, i, piece_end, finer, size))
        i = piece_end
    return out


def split_with_offsets(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Return (start, end) for each chunk; text[start:end] is the chunk.

    Guarantees: each chunk is at most `size` characters and starts and ends
    on non-whitespace; starts and ends strictly increase; every
    non-whitespace character of `text` is in at least one chunk.
    """
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    spans, window = [], []
    for piece in _pieces(text, 0, len(text), SEPARATORS, size):
        if window and piece[1] - window[0][0] > size:
            spans.append((window[0][0], window[-1][1]))
            # Keep the tail of this chunk (at most `overlap` chars) to start the next...
            while window and window[-1][1] - window[0][0] > overlap:
                window.pop(0)
            # ...but only as much as still fits together with the new piece.
            while window and piece[1] - window[0][0] > size:
                window.pop(0)
        window.append(piece)
    if window:
        spans.append((window[0][0], window[-1][1]))

    # Trim whitespace at both ends (adjusting offsets, not copying text).
    trimmed = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e <= s or (trimmed and e <= trimmed[-1][1]):
            continue  # empty, or adds no new text
        while trimmed and s <= trimmed[-1][0]:
            trimmed.pop()  # trimming made the previous chunk a part of this one
        trimmed.append((s, e))
    return trimmed
