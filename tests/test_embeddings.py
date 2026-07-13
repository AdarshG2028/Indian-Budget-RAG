import pytest
from pathlib import Path
from typing import List

from src.embeddings.models import ChunkData, Payload, EmbeddedChunk
from src.embeddings.utils import format_text_for_embedding, generate_deterministic_uuid
from src.embeddings.batch_processor import ChunkBatchProcessor
from src.embeddings.embedder import BaseEmbedder

# --- Mock Embedder for Testing ---
class MockEmbedder(BaseEmbedder):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Return a simple 3-dimensional mock embedding for each text
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]

    def embedding_dimension(self) -> int:
        return 3

# --- Tests ---

def test_chunk_data_initialization():
    """Test safe initialization of ChunkData from dictionary."""
    data = {
        "chunk_id": "test_1",
        "document": "budget",
        "text": "Hello world"
    }
    chunk = ChunkData.from_dict(data)
    
    assert chunk.chunk_id == "test_1"
    assert chunk.document == "budget"
    assert chunk.text == "Hello world"
    assert chunk.year == 2026 # Default value
    assert chunk.heading_path == [] # Default value

def test_format_text_for_embedding():
    """Test the natural text prepending logic."""
    chunk = ChunkData(
        chunk_id="t1",
        document="budget_speech",
        year=2026,
        text="This is the actual chunk text.",
        heading_path=["Part A", "Tourism"]
    )
    
    formatted_text = format_text_for_embedding(chunk)
    
    # Document title should be capitalized and underscores removed
    assert "Budget Speech" in formatted_text
    assert "Part A" in formatted_text
    assert "Tourism" in formatted_text
    assert "This is the actual chunk text." in formatted_text
    
    # Check that they are joined by newlines correctly
    expected = "Budget Speech\nPart A\nTourism\n\nThis is the actual chunk text."
    assert formatted_text == expected

def test_generate_deterministic_uuid():
    """Test UUID generation is deterministic."""
    uuid1 = generate_deterministic_uuid("test_chunk_1", "text_A")
    uuid2 = generate_deterministic_uuid("test_chunk_1", "text_A")
    uuid3 = generate_deterministic_uuid("test_chunk_2", "text_B")
    
    assert uuid1 == uuid2
    assert uuid1 != uuid3

def test_embedder_interface():
    """Test the abstract base embedder behavior with a mock implementation."""
    embedder = MockEmbedder()
    
    assert embedder.embedding_dimension() == 3
    
    docs = ["Doc 1", "Doc 2"]
    embeddings = embedder.embed_documents(docs)
    
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 3
    assert embeddings[0] == [0.1, 0.2, 0.3]

def test_embedded_chunk_serialization():
    """Test conversion of EmbeddedChunk to dictionary for vector DB insertion."""
    payload = Payload(
        document="test_doc",
        year=2026,
        heading_path=[],
        page_start=1,
        page_end=1,
        chunk_id="chunk_1",
        text="text"
    )
    
    embedded = EmbeddedChunk(
        id="123e4567-e89b-12d3-a456-426614174000",
        embedding=[0.5, 0.5],
        payload=payload
    )
    
    data = embedded.to_dict()
    assert data["id"] == "123e4567-e89b-12d3-a456-426614174000"
    assert data["embedding"] == [0.5, 0.5]
    assert data["payload"]["document"] == "test_doc"
    assert data["payload"]["chunk_id"] == "chunk_1"
