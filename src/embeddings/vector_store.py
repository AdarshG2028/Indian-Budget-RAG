"""
QdrantStore: manages all interactions with the Qdrant vector database.

Responsibilities:
  - Connect to a Qdrant instance (local or cloud).
  - Create / verify the collection on first run.
  - Batch-upsert EmbeddedChunk objects.
  - Expose a helper to delete the collection if needed.
"""
import logging
from typing import List, Optional

from .models import EmbeddedChunk

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
    from qdrant_client.http.models import (
        Distance,
        VectorParams,
        PointStruct,
        UpdateStatus,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning(
        "qdrant-client is not installed. "
        "Install it with: uv add qdrant-client"
    )


class QdrantStore:
    """
    Wraps Qdrant client operations for the Indian Budget RAG pipeline.

    Parameters
    ----------
    collection_name : str
        Name of the Qdrant collection.
    embedding_dim : int
        Dimensionality of the embedding vectors.
    url : str, optional
        Qdrant server URL. Defaults to ``http://localhost:6333``.
    api_key : str, optional
        API key for Qdrant Cloud. Leave None for local deployments.
    distance : str
        Similarity metric – ``"Cosine"`` | ``"Dot"`` | ``"Euclid"``.
    recreate : bool
        If True, drop and recreate the collection on startup.
    """

    def __init__(
        self,
        collection_name: str,
        embedding_dim: int,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        distance: str = "Cosine",
        recreate: bool = False,
    ):
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is required. Run: uv add qdrant-client"
            )

        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self._distance_map = {
            "Cosine": Distance.COSINE,
            "Dot": Distance.DOT,
            "Euclid": Distance.EUCLID,
        }
        self._distance = self._distance_map.get(distance, Distance.COSINE)

        logger.info(f"Connecting to Qdrant at {url} …")
        self.client = QdrantClient(url=url, api_key=api_key, timeout=60)
        logger.info("Connected to Qdrant.")

        self._ensure_collection(recreate=recreate)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _ensure_collection(self, recreate: bool = False) -> None:
        """Create the collection if it doesn't exist; optionally recreate it."""
        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection_name in existing:
            if recreate:
                logger.warning(
                    f"Recreating collection '{self.collection_name}' …"
                )
                self.client.delete_collection(self.collection_name)
            else:
                logger.info(
                    f"Collection '{self.collection_name}' already exists. "
                    "Reusing it."
                )
                return

        logger.info(
            f"Creating collection '{self.collection_name}' "
            f"(dim={self.embedding_dim}, distance={self._distance.value}) …"
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=self._distance,
            ),
        )
        logger.info(f"Collection '{self.collection_name}' created successfully.")

    def delete_collection(self) -> None:
        """Permanently remove the collection from Qdrant."""
        self.client.delete_collection(self.collection_name)
        logger.info(f"Collection '{self.collection_name}' deleted.")

    # ------------------------------------------------------------------
    # Upserting
    # ------------------------------------------------------------------

    def upsert(self, embedded_chunks: List[EmbeddedChunk]) -> None:
        """
        Upsert a batch of EmbeddedChunk objects into Qdrant.

        Uses deterministic UUIDs so re-running the pipeline is idempotent.
        """
        if not embedded_chunks:
            return

        points = [
            PointStruct(
                id=chunk.id,
                vector=chunk.embedding,
                payload={
                    "document": chunk.payload.document,
                    "year": chunk.payload.year,
                    "heading_path": chunk.payload.heading_path,
                    "page_start": chunk.payload.page_start,
                    "page_end": chunk.payload.page_end,
                    "chunk_id": chunk.payload.chunk_id,
                    "text": chunk.payload.text,
                },
            )
            for chunk in embedded_chunks
        ]

        result = self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,  # block until the operation is confirmed
        )

        if result.status != UpdateStatus.COMPLETED:
            logger.error(
                f"Qdrant upsert returned unexpected status: {result.status}"
            )
        else:
            logger.debug(
                f"Upserted {len(points)} points into '{self.collection_name}'."
            )

    # ------------------------------------------------------------------
    # Info helpers
    # ------------------------------------------------------------------

    def collection_info(self) -> dict:
        """Return basic stats about the collection."""
        info = self.client.get_collection(self.collection_name)
        return {
            "collection_name": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }
