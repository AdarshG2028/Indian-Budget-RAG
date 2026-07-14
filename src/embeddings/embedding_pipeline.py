import logging
import time
from pathlib import Path
from typing import List, Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .config import BATCH_SIZE
from .embedder import BaseEmbedder
from .batch_processor import ChunkBatchProcessor
from .utils import format_text_for_embedding, generate_deterministic_uuid
from .models import EmbeddedChunk, Payload
from .vector_store import QdrantStore

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """
    Orchestrates the full embedding → vector-DB ingestion process:

    1. Iterates over chunk batches from the filesystem.
    2. Formats each chunk text for embedding.
    3. Generates embeddings via the provided embedder.
    4. Upserts the resulting EmbeddedChunk objects into Qdrant.

    Parameters
    ----------
    embedder : BaseEmbedder
        The embedding model (e.g. BGEEmbedder).
    data_dir : Path
        Root directory containing ``chunks.json`` files.
    vector_store : QdrantStore, optional
        Pre-configured Qdrant store.  If *None*, data is NOT persisted
        (useful for dry-run / testing).
    batch_size : int
        Number of chunks to embed and upsert per iteration.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        data_dir: Path,
        vector_store: Optional[QdrantStore] = None,
        batch_size: int = BATCH_SIZE,
    ):
        self.embedder = embedder
        self.processor = ChunkBatchProcessor(data_dir=data_dir, batch_size=batch_size)
        self.vector_store = vector_store
        
        self.state_file = Path(data_dir) / "processed_ids.json"
        self.processed_ids = self._load_state()

        if self.vector_store is None:
            logger.warning(
                "No QdrantStore provided – pipeline will run in DRY-RUN mode "
                "(embeddings will not be saved)."
            )

    def _load_state(self) -> set:
        """Loads the set of already processed deterministic UUIDs."""
        if self.state_file.exists():
            try:
                import json
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data)} processed IDs from {self.state_file.name}")
                    return set(data)
            except Exception as e:
                logger.warning(f"Could not load state file: {e}")
        return set()

    def _save_state(self, new_ids: List[str]):
        """Appends newly processed UUIDs to the state file."""
        self.processed_ids.update(new_ids)
        try:
            import json
            with open(self.state_file, "w") as f:
                json.dump(list(self.processed_ids), f)
        except Exception as e:
            logger.warning(f"Could not save state file: {e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_payload(self, chunk) -> Payload:
        """Converts a ChunkData object into a Payload for the Vector DB."""
        return Payload(
            document=chunk.document,
            year=chunk.year,
            heading_path=chunk.heading_path,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            chunk_id=chunk.chunk_id,
            text=chunk.text,
        )

    def _save_to_vector_db(self, embedded_chunks: List[EmbeddedChunk]) -> None:
        """Upsert a batch into Qdrant (no-op in dry-run mode)."""
        if self.vector_store is None:
            return
        self.vector_store.upsert(embedded_chunks)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the pipeline end-to-end.

        Logs a summary (batches, chunks, elapsed time, throughput) on
        completion.
        """
        logger.info(f"Starting embedding pipeline using {self.embedder.__class__.__name__}")
        logger.info(f"Embedding dimension: {self.embedder.embedding_dimension()}")

        if self.vector_store:
            logger.info(
                f"Qdrant collection: '{self.vector_store.collection_name}'"
            )

        start_time = time.time()
        total_chunks_processed = 0
        total_batches = 0

        batch_iterator = self.processor.iter_batches()

        if tqdm is not None:
            batch_iterator = tqdm(
                batch_iterator, desc="Embedding & Upserting", unit="batch"
            )

        for batch in batch_iterator:
            batch_start = time.time()
            total_batches += 1

            # 1. Filter out chunks we've already processed
            pending_chunks = []
            for chunk in batch:
                chunk_id = generate_deterministic_uuid(chunk.chunk_id, chunk.text)
                if chunk_id not in self.processed_ids:
                    pending_chunks.append((chunk, chunk_id))
            
            if not pending_chunks:
                logger.debug(f"Batch {total_batches}: all {len(batch)} chunks already processed, skipping.")
                continue

            # Unpack pending chunks
            chunks_to_process = [c[0] for c in pending_chunks]
            chunk_ids = [c[1] for c in pending_chunks]

            # 2. Format texts for embedding
            texts_to_embed = [format_text_for_embedding(chunk) for chunk in chunks_to_process]

            # 3. Generate embeddings
            embeddings = self.embedder.embed_documents(texts_to_embed)

            # 4. Build EmbeddedChunk objects
            embedded_chunks: List[EmbeddedChunk] = []
            for chunk, chunk_id, embedding in zip(chunks_to_process, chunk_ids, embeddings):
                payload = self._create_payload(chunk)
                embedded_chunks.append(
                    EmbeddedChunk(
                        id=chunk_id,
                        embedding=embedding,
                        payload=payload,
                    )
                )

            total_chunks_processed += len(embedded_chunks)

            # 5. Persist to Qdrant
            self._save_to_vector_db(embedded_chunks)
            
            # 6. Save state locally
            self._save_state(chunk_ids)

            batch_time = time.time() - batch_start
            logger.debug(
                f"Batch {total_batches}: {len(embedded_chunks)} chunks "
                f"in {batch_time:.2f}s"
            )

        elapsed = time.time() - start_time
        throughput = total_chunks_processed / elapsed if elapsed > 0 else 0

        logger.info("-" * 40)
        logger.info("PIPELINE SUMMARY")
        logger.info("-" * 40)
        logger.info(f"Total Batches Processed: {total_batches}")
        logger.info(f"Total Chunks Embedded:   {total_chunks_processed}")
        logger.info(f"Total Time Elapsed:      {elapsed:.2f}s")
        logger.info(f"Average Throughput:      {throughput:.2f} chunks/sec")

        if self.vector_store:
            try:
                info = self.vector_store.collection_info()
                logger.info(
                    f"Qdrant Collection Size:  {info['points_count']} points"
                )
            except Exception as e:
                logger.warning(f"Could not fetch collection info: {e}")

        logger.info("-" * 40)
