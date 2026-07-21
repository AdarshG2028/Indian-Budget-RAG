"""
Configuration settings for the chunking pipeline.
"""
from dataclasses import dataclass

@dataclass
class ChunkingConfig:
    # Chunk sizes are measured in EMBEDDING_MODEL tokens (see tokenizer.py),
    # not tiktoken. bge-base-en-v1.5 truncates at 512, so these must leave room
    # for the embedding header and [CLS]/[SEP].
    CHUNK_SIZE: int = 380
    CHUNK_OVERLAP: int = 150
    MIN_CHUNK_SIZE: int = 120
    MAX_CHUNK_SIZE: int = 440

    # Embedding model whose tokenizer defines the budget above. Must match
    # src/embeddings/config.py:MODEL_NAME.
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBED_MAX_TOKENS: int = 512
    SPECIAL_TOKEN_RESERVE: int = 2  # [CLS] + [SEP]
    # Ceiling on the heading path carried into the embedded text, so a few very
    # long section headings can't crowd out the chunk text itself.
    MAX_HEADING_TOKENS: int = 48

    # Fallback encoding used only when transformers is unavailable.
    ENCODING_NAME: str = "cl100k_base"

config = ChunkingConfig()
