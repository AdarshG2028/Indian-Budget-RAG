"""
Token counting for the chunking pipeline.

Chunks are sized so they survive the embedding model intact. The model that
matters is the *embedding* model (BAAI/bge-base-en-v1.5), not tiktoken:
sentence-transformers silently truncates anything past 512 tokens, so counting
in any other tokenizer lets oversized chunks through unnoticed.
"""
from functools import lru_cache
from typing import Callable

from .config import config


@lru_cache(maxsize=2)
def _load_counter(model_name: str) -> Callable[[str], int]:
    """
    Return a token-counting function for `model_name`.

    Falls back to tiktoken when transformers isn't installed, so ingestion-time
    tooling still runs in a slim environment. The fallback undercounts (~0.82x
    on this corpus), so it is a convenience, not a substitute.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        import tiktoken
        enc = tiktoken.get_encoding(config.ENCODING_NAME)
        return lambda text: len(enc.encode(text))

    tok = AutoTokenizer.from_pretrained(model_name)
    # add_special_tokens=False: [CLS]/[SEP] are accounted for separately via
    # config.SPECIAL_TOKEN_RESERVE, so callers can budget them once per chunk
    # rather than paying them on every intermediate paragraph measurement.
    return lambda text: len(tok.encode(text, add_special_tokens=False))


def get_token_counter() -> Callable[[str], int]:
    """Token counter matching the embedding model."""
    return _load_counter(config.EMBEDDING_MODEL)


def split_to_budget(text: str, count: Callable[[str], int], budget: int) -> list:
    """
    Split `text` into parts of at most `budget` tokens.

    Tries separators in descending order of semantic value: table cells, then
    <br> (pymupdf4llm emits these inside cells, often with no surrounding
    whitespace), then words. Falls back to a proportional character split for
    pathological content — some budget-document table cells are a single
    "word" of hundreds of <br>-joined figures, which no separator can break.
    """
    if count(text) <= budget:
        return [text]

    for sep in ('|', '<br>', ' '):
        if sep not in text:
            continue
        parts, current = [], []
        for piece in text.split(sep):
            candidate = current + [piece]
            if current and count(sep.join(candidate)) > budget:
                parts.append(sep.join(current))
                current = [piece]
            else:
                current = candidate
        if current:
            parts.append(sep.join(current))
        # Recurse: a single piece may still exceed the budget on its own.
        if all(count(p) <= budget for p in parts):
            return [p for p in parts if p.strip()]
        out = []
        for p in parts:
            out.extend(split_to_budget(p, count, budget) if count(p) > budget else [p])
        return [p for p in out if p.strip()]

    # No separator worked — cut by characters, scaled by the observed ratio.
    ratio = len(text) / max(count(text), 1)
    width = max(int(budget * ratio * 0.9), 1)
    return [text[i:i + width] for i in range(0, len(text), width)]


def cap_heading_path(path: list, count: Callable[[str], int], max_tokens: int) -> list:
    """
    Trim a heading path so it can't crowd out the chunk text.

    Most paths are ~13 tokens, but a few section headings in these documents run
    past 160 — often Devanagari, which an English-only embedding model splits
    into many low-signal tokens. Left alone those consume a third of the
    512-token window. Keeps the broadest headings first (they carry the
    document context) and truncates the first entry by words if it alone
    exceeds the cap.
    """
    if not path:
        return []

    kept, used = [], 0
    for heading in path:
        cost = count(heading)
        if used + cost > max_tokens:
            break
        kept.append(heading)
        used += cost

    if kept:
        return kept

    # First heading alone is over budget — truncate it rather than drop context.
    words = path[0].split()
    truncated = []
    for word in words:
        if count(" ".join(truncated + [word])) > max_tokens:
            break
        truncated.append(word)
    return [" ".join(truncated)] if truncated else []


def embedding_budget(header_tokens: int = 0) -> int:
    """
    Tokens available for chunk *text*, given a header of `header_tokens`.

    The embedder prepends a header (document type + title) via
    format_text_for_embedding, and the model adds [CLS]/[SEP]. Both eat into
    the 512-token window before the chunk text is even seen.
    """
    return config.EMBED_MAX_TOKENS - header_tokens - config.SPECIAL_TOKEN_RESERVE
