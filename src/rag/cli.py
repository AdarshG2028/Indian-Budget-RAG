"""
CLI for testing RAG pipeline operations.
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
from llm import GroqLLM, LLMConfig
from rag import RAGPipeline, RAGConfig
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
        description="Test RAG pipeline for Indian Budget RAG"
    )
    
    # Query arguments
    parser.add_argument(
        "question",
        type=str,
        help="Question to answer using RAG"
    )
    
    # Retrieval arguments
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve (default: 10)"
    )
    
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Minimum similarity score threshold (default: None)"
    )
    
    parser.add_argument(
        "--context-max-tokens",
        type=int,
        default=4000,
        help="Maximum tokens for context (default: 4000)"
    )
    
    # LLM arguments
    parser.add_argument(
        "--llm-model",
        type=str,
        default="llama-3.3-70b-versatile",
        help="Groq model name (default: llama-3.3-70b-versatile)"
    )
    
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)"
    )
    
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=1024,
        help="LLM max tokens (default: 1024)"
    )
    
    parser.add_argument(
        "--groq-api-key",
        type=str,
        default=os.getenv("GROQ_API_KEY"),
        help="Groq API key (default: from env)"
    )
    
    # Template arguments
    parser.add_argument(
        "--template",
        type=str,
        default="qa",
        choices=["qa", "summarization", "analysis"],
        help="Prompt template to use (default: qa)"
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
    
    # Reranking arguments
    parser.add_argument(
        "--enable-rerank",
        action="store_true",
        help="Enable reranking of retrieved chunks"
    )
    
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
        default=5,
        help="Number of chunks to return after reranking (default: 5)"
    )
    
    parser.add_argument(
        "--reranker-device",
        type=str,
        default="cpu",
        help="Device for reranker (default: cpu)"
    )
    
    # Output arguments
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream LLM response"
    )
    
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Show retrieved context"
    )
    
    parser.add_argument(
        "--show-citations",
        action="store_true",
        help="Show citation details"
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
        # Validate Groq API key
        if not args.groq_api_key:
            raise ValueError(
                "Groq API key is required. Set GROQ_API_KEY environment variable "
                "or use --groq-api-key argument"
            )
        
        # Initialize embedder
        logger.info("Initializing embedder...")
        embedder = BGEEmbedder(model_name=args.embedding_model, device=args.embedding_device)
        
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
        
        # Initialize LLM
        logger.info(f"Initializing Groq LLM with model: {args.llm_model}")
        llm_config = LLMConfig(
            model_name=args.llm_model,
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
            api_key=args.groq_api_key
        )
        llm = GroqLLM(llm_config)
        
        # Initialize reranker if enabled
        reranker = None
        if args.enable_rerank:
            logger.info(f"Initializing reranker with model: {args.reranker_model}")
            reranker_config = RerankerConfig(
                model_name=args.reranker_model,
                device=args.reranker_device,
                retrieve_top_k=args.top_k,
                rerank_top_k=args.rerank_top_k,
                return_top_k=args.return_top_k,
                normalize_scores=True,
                enable_fallback=True
            )
            reranker = CrossEncoderReranker(reranker_config)
            logger.info("Reranker initialized successfully")
        
        # Initialize RAG pipeline
        logger.info("Initializing RAG pipeline...")
        rag_config = RAGConfig(
            retrieval_top_k=args.top_k,
            retrieval_score_threshold=args.score_threshold,
            reranker_enabled=args.enable_rerank,
            reranker_config=None,  # Will be set from reranker if needed
            context_max_tokens=args.context_max_tokens,
            llm_temperature=args.llm_temperature,
            llm_max_tokens=args.llm_max_tokens,
            prompt_template=args.template,
            include_citations=True
        )
        pipeline = RAGPipeline(
            retriever=retriever,
            llm=llm,
            reranker=reranker,
            config=rag_config
        )
        
        # Execute RAG query
        logger.info(f"Processing question: '{args.question}'")
        
        if args.stream:
            print("\n" + "=" * 80)
            print("RAG RESPONSE (Streaming)")
            print("=" * 80)
            print()
            
            for chunk in pipeline.query_stream(args.question):
                print(chunk, end="", flush=True)
            
            print("\n")
        else:
            response = pipeline.query(args.question, stream=False)
            
            # Print results
            print("\n" + "=" * 80)
            print("RAG RESPONSE")
            print("=" * 80)
            
            print(f"\nQuestion: {args.question}")
            print(f"\nAnswer:")
            print(response.answer)
            
            # Print citations if available
            if response.citations:
                print(f"\nCitations:")
                for marker, citation in response.citations.items():
                    print(f"  {marker}: {citation['document']} ({citation['year']}), "
                          f"{citation['section']}, p.{citation['page_start']}-{citation['page_end']}")
            
            # Print metrics
            print(f"\nRetrieval Metrics:")
            print(f"  Chunks retrieved: {response.retrieval_context.metrics.chunks_returned}")
            print(f"  Avg similarity: {response.retrieval_context.metrics.avg_similarity:.4f}")
            print(f"  Retrieval latency: {response.retrieval_context.metrics.retrieval_latency:.3f}s")
            
            print(f"\nLLM Metrics:")
            print(f"  Model: {response.llm_metrics.model}")
            print(f"  Prompt tokens: {response.llm_metrics.prompt_tokens}")
            print(f"  Completion tokens: {response.llm_metrics.completion_tokens}")
            print(f"  Total tokens: {response.llm_metrics.total_tokens}")
            print(f"  LLM latency: {response.llm_metrics.llm_latency:.3f}s")
            print(f"  Total latency: {response.llm_metrics.total_latency:.3f}s")
            
            # Show context if requested
            if args.show_context:
                print(f"\nRetrieved Context ({len(response.retrieval_context.results)} chunks):")
                for result in response.retrieval_context.results:
                    print(f"\n  [{result.rank}] {result.document} ({result.year})")
                    print(f"      Section: {result.section}")
                    if result.subsection:
                        print(f"      Subsection: {result.subsection}")
                    print(f"      Pages: {result.page_start}-{result.page_end}")
                    print(f"      Score: {result.score:.4f}")
                    print(f"      Text: {result.text[:200]}...")
            
            # Show citation details if requested
            if args.show_citations and response.citations:
                print(f"\nCitation Details:")
                for marker, citation in response.citations.items():
                    print(f"\n  {marker}:")
                    print(f"    Document: {citation['document']}")
                    print(f"    Year: {citation['year']}")
                    print(f"    Section: {citation['section']}")
                    if citation['subsection']:
                        print(f"    Subsection: {citation['subsection']}")
                    print(f"    Pages: {citation['page_start']}-{citation['page_end']}")
                    print(f"    Chunk ID: {citation['chunk_id']}")
                    print(f"    Score: {citation['score']:.4f}")
        
        print("\n" + "=" * 80)
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"RAG pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
