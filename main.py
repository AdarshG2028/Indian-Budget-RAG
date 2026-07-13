"""
main.py – Entry point for the Indian Budget RAG embedding pipeline.

Usage:
    uv run python main.py                         # embed & store (default)
    uv run python main.py --dry-run               # embed only, no Qdrant writes
    uv run python main.py --recreate              # drop & recreate collection first
    uv run python main.py --qdrant-url http://...  # override Qdrant URL
"""
import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup (must happen before any module import that uses logging)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "2026" / "chunks"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed Indian Budget chunks and store them in Qdrant."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run embedding only; do NOT write to Qdrant.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the Qdrant collection before ingesting.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Override the Qdrant server URL (e.g. http://localhost:6333).",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Override the Qdrant collection name.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Path to the directory containing chunks.json files.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- Imports (deferred so --help is instant) ----
    from src.embeddings.config import (
        QDRANT_URL,
        QDRANT_API_KEY,
        QDRANT_COLLECTION,
        QDRANT_DISTANCE,
    )
    from src.embeddings.embedder import BGEEmbedder
    from src.embeddings.vector_store import QdrantStore
    from src.embeddings.embedding_pipeline import EmbeddingPipeline

    # ---- Resolve settings ----
    qdrant_url = args.qdrant_url or QDRANT_URL
    collection_name = args.collection or QDRANT_COLLECTION
    data_dir = Path(args.data_dir)

    logger.info("=" * 60)
    logger.info("Indian Budget RAG – Embedding Pipeline")
    logger.info("=" * 60)
    logger.info(f"Data directory : {data_dir}")
    logger.info(f"Qdrant URL     : {qdrant_url}")
    logger.info(f"Collection     : {collection_name}")
    logger.info(f"Dry-run        : {args.dry_run}")
    logger.info(f"Recreate       : {args.recreate}")
    logger.info("=" * 60)

    # Validate data directory
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.error(
            "Make sure you have run the chunking pipeline first "
            "(src/batch_processor.py)."
        )
        sys.exit(1)

    # ---- Initialise embedder ----
    logger.info("Loading embedding model …")
    try:
        embedder = BGEEmbedder()
    except ImportError as e:
        logger.error(f"Could not load embedder: {e}")
        sys.exit(1)

    # ---- Initialise Qdrant store ----
    vector_store = None
    if not args.dry_run:
        try:
            vector_store = QdrantStore(
                collection_name=collection_name,
                embedding_dim=embedder.embedding_dimension(),
                url=qdrant_url,
                api_key=QDRANT_API_KEY,
                distance=QDRANT_DISTANCE,
                recreate=args.recreate,
            )
        except ImportError as e:
            logger.error(f"Could not initialise QdrantStore: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(
                f"Failed to connect to Qdrant at {qdrant_url}: {e}\n"
                "Is Qdrant running?  Start it with:\n"
                "  docker run -p 6333:6333 qdrant/qdrant"
            )
            sys.exit(1)

    # ---- Run pipeline ----
    pipeline = EmbeddingPipeline(
        embedder=embedder,
        data_dir=data_dir,
        vector_store=vector_store,
    )

    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(0)

    logger.info("Done.")


if __name__ == "__main__":
    main()
