"""
CLI for testing and debugging reranking operations.
"""
import argparse
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embeddings.embedder import BGEEmbedder
from retrieval import DenseRetriever, QdrantVectorStore, RetrievalConfig
from reranking import CrossEncoderReranker, RerankerConfig
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


def main():
    parser = argparse.ArgumentParser(
        description="Test reranking for Indian Budget RAG"
    )
    
    # Query arguments
    parser.add_argument(
        "query",
        type=str,
        help="Query to test reranking on"
    )
    
    # Retrieval arguments
    parser.add_argument(
        "--retrieve-top-k",
        type=int,
        default=20,
        help="Number of chunks to retrieve (default: 20)"
    )
    
    # Reranking arguments
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="ms-marco-MiniLM-L-6-v2",
        help="Reranker model name (default: ms-marco-MiniLM-L-6-v2)"
    )
    
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=20,
        help="Number of chunks to rerank (default: 20)"
    )
    
    parser.add_argument(
        "--return-top-k",
        type=int,
        default=10,
        help="Number of chunks to return after reranking (default: 10)"
    )
    
    parser.add_argument(
        "--reranker-device",
        type=str,
        default="cpu",
        help="Device for reranker (default: cpu)"
    )
    
    parser.add_argument(
        "--normalize-scores",
        action="store_true",
        default=True,
        help="Normalize reranker scores (default: True)"
    )
    
    # Vector store arguments
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
    
    # Embedding arguments
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=MODEL_NAME,
        help=f"Embedding model name (default: {MODEL_NAME})"
    )
    
    parser.add_argument(
        "--embedding-device",
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
    
    # Output arguments
    parser.add_argument(
        "--show-original",
        action="store_true",
        help="Show original retrieval results"
    )
    
    parser.add_argument(
        "--show-reranked",
        action="store_true",
        help="Show reranked results"
    )
    
    parser.add_argument(
        "--show-comparison",
        action="store_true",
        help="Show side-by-side comparison"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize embedder
        logger.info("Initializing embedder...")
        embedder = BGEEmbedder(
            model_name=args.embedding_model,
            device=args.embedding_device
        )
        
        # Initialize vector store
        logger.info("Connecting to vector store...")
        vector_store = QdrantVectorStore(
            collection_name=args.collection,
            embedding_dim=args.embedding_dim,
            url=args.qdrant_url,
            api_key=args.qdrant_api_key
        )
        
        # Initialize retriever
        logger.info("Initializing retriever...")
        retriever = DenseRetriever(
            embedder=embedder,
            vector_store=vector_store,
            collection_name=args.collection
        )
        
        # Perform retrieval
        logger.info(f"Retrieving top {args.retrieve_top_k} chunks for query: '{args.query}'")
        retrieval_config = RetrievalConfig(top_k=args.retrieve_top_k)
        retrieval_context = retriever.retrieve(args.query, retrieval_config)
        
        logger.info(f"Retrieved {len(retrieval_context.results)} chunks")
        logger.info(f"Retrieval latency: {retrieval_context.metrics.retrieval_latency:.3f}s")
        
        # Show original results if requested
        if args.show_original:
            print("\n" + "=" * 80)
            print("ORIGINAL RETRIEVAL RESULTS")
            print("=" * 80)
            for i, result in enumerate(retrieval_context.results, 1):
                print(f"\n[{i}] Rank: {result.rank}")
                print(f"    Chunk ID: {result.chunk_id}")
                print(f"    Score: {result.score:.4f}")
                print(f"    Document: {result.document}")
                print(f"    Text: {result.text[:200]}...")
        
        # Initialize reranker
        logger.info(f"Initializing reranker with model: {args.reranker_model}")
        reranker_config = RerankerConfig(
            model_name=args.reranker_model,
            device=args.reranker_device,
            retrieve_top_k=args.retrieve_top_k,
            rerank_top_k=args.rerank_top_k,
            return_top_k=args.return_top_k,
            normalize_scores=args.normalize_scores,
            enable_fallback=True
        )
        reranker = CrossEncoderReranker(reranker_config)
        
        # Perform reranking
        logger.info("Reranking retrieved chunks...")
        
        # Convert retrieval results to format expected by reranker
        chunks_for_reranking = [
            {
                "chunk_id": r.chunk_id,
                "text": r.text,
                "score": r.score,
                "document": r.document,
                "year": r.year,
                "section": r.section,
                "subsection": r.subsection,
                "page_start": r.page_start,
                "page_end": r.page_end
            }
            for r in retrieval_context.results
        ]
        
        reranker_output = reranker.rerank(args.query, chunks_for_reranking)
        
        logger.info(f"Reranking latency: {reranker_output.reranking_latency:.3f}s")
        logger.info(f"Fallback used: {reranker_output.fallback_used}")
        
        # Show reranked results if requested
        if args.show_reranked:
            print("\n" + "=" * 80)
            print("RERANKED RESULTS")
            print("=" * 80)
            for result in reranker_output.results:
                print(f"\n[Rank {result.reranked_rank}]")
                print(f"    Original Rank: {result.original_rank}")
                print(f"    Chunk ID: {result.chunk_id}")
                print(f"    Original Score: {result.original_score:.4f}")
                print(f"    Reranked Score: {result.reranked_score:.4f}")
                print(f"    Score Delta: {result.score_delta:+.4f}")
                print(f"    Document: {result.metadata.get('document', 'N/A')}")
                print(f"    Text: {result.text[:200]}...")
        
        # Show comparison if requested
        if args.show_comparison:
            print("\n" + "=" * 80)
            print("RANKING COMPARISON")
            print("=" * 80)
            print(f"\n{'Chunk ID':<40} {'Original':>10} {'Reranked':>10} {'Delta':>10}")
            print("-" * 80)
            for result in reranker_output.results:
                print(f"{result.chunk_id:<40} {result.original_rank:>10} {result.reranked_rank:>10} {result.reranked_rank - result.original_rank:+10}")
        
        # Show reranker metrics
        print("\n" + "=" * 80)
        print("RERANKER METRICS")
        print("=" * 80)
        metrics = reranker.get_metrics()
        print(f"Total queries processed: {metrics.total_queries}")
        print(f"Average score improvement: {metrics.avg_score_improvement:.4f}")
        print(f"Average rank movement: {metrics.avg_rank_movement:.2f}")
        print(f"Largest rank movement: {metrics.largest_rank_movement}")
        print(f"Reordering percentage: {metrics.reordering_percentage:.1f}%")
        print(f"Fallback count: {metrics.fallback_count}")
        print(f"Total reranking latency: {metrics.reranking_latency:.3f}s")
        
        print("\n✅ Reranking test completed successfully")
        
    except Exception as e:
        logger.error(f"Reranking test failed: {e}")
        raise


if __name__ == "__main__":
    main()
