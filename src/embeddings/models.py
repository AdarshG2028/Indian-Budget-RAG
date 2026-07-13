from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import uuid

@dataclass
class ChunkData:
    """Represents an incoming chunk parsed from the JSON output of the chunker."""
    chunk_id: str
    document: str
    year: int
    text: str
    section: str = ""
    subsection: str = ""
    paragraph_start: int = 0
    paragraph_end: int = 0
    page_start: int = 0
    page_end: int = 0
    token_count: int = 0
    char_count: int = 0
    heading_path: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkData":
        """Safely initialize from a dictionary, handling missing optional fields."""
        return cls(
            chunk_id=data.get("chunk_id", ""),
            document=data.get("document", "Unknown"),
            year=data.get("year", 2026),
            text=data.get("text", ""),
            section=data.get("section", ""),
            subsection=data.get("subsection", ""),
            paragraph_start=data.get("paragraph_start", 0),
            paragraph_end=data.get("paragraph_end", 0),
            page_start=data.get("page_start", 0),
            page_end=data.get("page_end", 0),
            token_count=data.get("token_count", 0),
            char_count=data.get("char_count", 0),
            heading_path=data.get("heading_path", []),
            keywords=data.get("keywords", []),
            entities=data.get("entities", {})
        )

@dataclass
class Payload:
    """The metadata payload that will be stored alongside the embedding in the vector database."""
    document: str
    year: int
    heading_path: List[str]
    page_start: int
    page_end: int
    chunk_id: str
    text: str

@dataclass
class EmbeddedChunk:
    """Represents a fully processed chunk ready for vector database insertion."""
    id: str  # Must be a UUID string
    embedding: List[float]
    payload: Payload
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for easy JSON serialization or Vector DB insertion."""
        return {
            "id": self.id,
            "embedding": self.embedding,
            "payload": {
                "document": self.payload.document,
                "year": self.payload.year,
                "heading_path": self.payload.heading_path,
                "page_start": self.payload.page_start,
                "page_end": self.payload.page_end,
                "chunk_id": self.payload.chunk_id,
                "text": self.payload.text
            }
        }
