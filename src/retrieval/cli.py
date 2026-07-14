"""
CLI for testing retrieval operations.
"""
import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embeddings.embedder import BGEEmbedder
from retrieval import DenseRetriever, QdrantVectorStore, RetrievalConfig
from embeddings.config import MODEL_NAME, DEVICE

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_filters(filter_args: list) -> Dict[str, Any]:
    """
    Parse filter arguments from CLI.
    
    Format: key=value
    Example: year=2026 document=budget_speech
    """
    filters = {}
    for arg in filter_args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Try to convert to int if possible
            try:
                value = int(value)
            except ValueError:
                pass
            filters[key] = value
        else:
            logger.warning(f"Invalid filter format: {arg}. Use key=value")
    return filters


def main():
    parser = argparse.ArgumentParser(
        description="Test retrieval operations for Indian Budget RAG"
    )
    
    parser.add_argument(
        "query",
        type=str,
        help="Query text to search for"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)"
    )
    
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Minimum similarity score threshold (default: None)"
    )
    
    parser.add_argument(
        "--filter",
        type=str,
        action="append",
        dest="filters",
        help="Metadata filters in key=value format (can be used multiple times)"
    )
    
    parser.add_argument(
        "--collection",
        type=str,
        default="indian_budget_2026",
        help="Qdrant collection name (default: indian_budget_2026)"
    )
    
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant server URL (default: from env or http://localhost:6333)"
    )
    
    parser.add_argument(
        "--qdrant-api-key",
        type=str,
        default=os.getenv("QDRANT_API_KEY"),
        help="Qdrant Cloud API key (default: from env or None)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"Embedding model name (default: {MODEL_NAME})"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help=f"Device for embedding model (default: {DEVICE})"
    )
    
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=768,
        help="Embedding dimension (default: 768 for bge-base-en-v1.5)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse filters
    filters = parse_filters(args.filters or [])
    if filters:
        logger.info(f"Applying filters: {filters}")
    
    try:
        # Initialize components
        logger.info("Initializing embedder...")
        embedder = BGEEmbedder(model_name=args.model, device=args.device)
        
        logger.info("Connecting to vector store...")
        vector_store = QdrantVectorStore(
            collection_name=args.collection,
            embedding_dim=args.embedding_dim,
            url=args.qdrant_url,
            api_key=args.qdrant_api_key
        )
        
        logger.info("Initializing retriever...")
        retriever = DenseRetriever(
            embedder=embedder,
            vector_store=vector_store,
            collection_name=args.collection
        )
        
        # Create retrieval config
        config = RetrievalConfig(
            top_k=args.top_k,
            score_threshold=args.score_threshold,
            filters=filters
        )
        
        # Perform retrieval
        logger.info(f"Searching for: '{args.query}'")
        context = retriever.retrieve_with_scores(args.query, config)
        
        # Print results
        print("\n" + "=" * 80)
        print(f"RETRIEVAL RESULTS ({len(context.results)} chunks found)")
        print("=" * 80)
        
        # Print metrics
        print(f"\nMetrics:")
        print(f"  Average similarity: {context.metrics.avg_similarity:.4f}")
        print(f"  Highest similarity: {context.metrics.highest_similarity:.4f}")
        print(f"  Lowest similarity: {context.metrics.lowest_similarity:.4f}")
        print(f"  Retrieval latency: {context.metrics.retrieval_latency:.3f}s")
        print(f"  Embedding latency: {context.metrics.embedding_latency:.3f}s")
        print(f"  Search latency: {context.metrics.search_latency:.3f}s")
        
        if not context.results:
            print("\nNo results found.")
            return
        
        print(f"\nResults:")
        for result in context.results:
            print(f"\n--- Rank {result.rank} ---")
            print(f"Score: {result.score:.4f}")
            print(f"Document: {result.document}")
            print(f"Year: {result.year}")
            print(f"Section: {result.section}")
            if result.subsection:
                print(f"Subsection: {result.subsection}")
            print(f"Pages: {result.page_start}-{result.page_end}")
            print(f"Paragraphs: {result.paragraph_start}-{result.paragraph_end}")
            print(f"Chunk ID: {result.chunk_id}")
            print(f"\nText Preview:")
            print(result.text[:500] + "..." if len(result.text) > 500 else result.text)
        
        print("\n" + "=" * 80)
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
