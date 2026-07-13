"""
Configuration settings for the chunking pipeline.
"""
from dataclasses import dataclass

@dataclass
class ChunkingConfig:
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    MIN_CHUNK_SIZE: int = 300
    MAX_CHUNK_SIZE: int = 900
    
    # Optional parameters for token counting
    ENCODING_NAME: str = "cl100k_base"

config = ChunkingConfig()
