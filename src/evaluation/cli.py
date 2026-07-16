"""
CLI for evaluation framework.
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
from evaluation import (
    EvaluationRunner,
    ExperimentTracker,
    ReportGenerator,
    DataLoader
)
from evaluation.retrieval import (
    RecallAtK,
    PrecisionAtK,
    MRR,
    NDCG,
    HitRate,
    MAP
)
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
        description="Run retrieval evaluation for Indian Budget RAG"
    )
    
    # Dataset arguments
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to evaluation dataset (JSON or CSV)"
    )
    
    # Retrieval arguments
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve (default: 10)"
    )
    
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Minimum similarity score threshold (default: None)"
    )
    
    # Metrics arguments
    parser.add_argument(
        "--k-values",
        type=int,
        nargs='+',
        default=[1, 5, 10],
        help="K values for @K metrics (default: 1 5 10)"
    )
    
    parser.add_argument(
        "--min-relevance",
        type=int,
        default=1,
        help="Minimum relevance threshold (0-3, default: 1)"
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
        default=10,
        help="Number of chunks to return after reranking (default: 10)"
    )
    
    parser.add_argument(
        "--reranker-device",
        type=str,
        default="cpu",
        help="Device for reranker (default: cpu)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/reports",
        help="Output directory for reports (default: evaluation/reports)"
    )
    
    parser.add_argument(
        "--output-formats",
        type=str,
        nargs='+',
        choices=["json", "csv", "markdown"],
        default=["json", "markdown"],
        help="Output formats (default: json markdown)"
    )
    
    parser.add_argument(
        "--save-experiment",
        action="store_true",
        help="Save experiment to experiment tracker"
    )
    
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="evaluation/experiments",
        help="Directory for experiment storage (default: evaluation/experiments)"
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
        
        # Create metrics
        logger.info("Initializing metrics...")
        metrics = []
        
        # MRR (no K parameter)
        metrics.append(MRR(min_relevance=args.min_relevance))
        
        # @K metrics for each K value
        for k in args.k_values:
            metrics.extend([
                RecallAtK(k=k, min_relevance=args.min_relevance),
                PrecisionAtK(k=k, min_relevance=args.min_relevance),
                NDCG(k=k, min_relevance=args.min_relevance),
                HitRate(k=k, min_relevance=args.min_relevance),
                MAP(k=k, min_relevance=args.min_relevance)
            ])
        
        logger.info(f"Created {len(metrics)} metrics")
        
        # Initialize evaluation runner
        logger.info("Initializing evaluation runner...")
        retrieval_config = RetrievalConfig(
            top_k=args.top_k,
            score_threshold=args.score_threshold
        )
        
        runner = EvaluationRunner(
            retriever=retriever,
            metrics=metrics,
            reranker=reranker
        )
        
        # Run evaluation
        logger.info(f"Running evaluation on dataset: {args.dataset}")
        report = runner.run(
            dataset_path=args.dataset,
            retrieval_config=retrieval_config,
            top_k=args.top_k
        )
        
        # Generate reports
        logger.info("Generating reports...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if "json" in args.output_formats:
            json_path = output_dir / f"{report.metadata.experiment_id}.json"
            ReportGenerator.generate_json(report, str(json_path))
        
        if "csv" in args.output_formats:
            csv_dir = output_dir / f"{report.metadata.experiment_id}_csv"
            ReportGenerator.generate_csv(report, str(csv_dir))
        
        if "markdown" in args.output_formats:
            md_path = output_dir / f"{report.metadata.experiment_id}.md"
            ReportGenerator.generate_markdown(report, str(md_path))
        
        # Save experiment if requested
        if args.save_experiment:
            logger.info("Saving experiment...")
            tracker = ExperimentTracker(storage_dir=args.experiment_dir)
            tracker.save_experiment(report)
        
        # Print summary
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"\nExperiment ID: {report.metadata.experiment_id}")
        print(f"Dataset: {report.metadata.dataset}")
        print(f"Queries: {len(report.query_results)}")
        print(f"\nAggregate Metrics:")
        
        # Print key metrics (filter out std)
        for metric_name, value in report.aggregate_metrics.items():
            if not metric_name.endswith("_std"):
                print(f"  {metric_name}: {value:.4f}")
        
        print(f"\nLatency Metrics:")
        for metric_name, value in report.latency_metrics.items():
            print(f"  {metric_name}: {value:.4f}s")
        
        print(f"\nFailure Analysis:")
        print(f"  Zero Recall: {report.failure_analysis['zero_recall_count']}/{report.failure_analysis['total_queries']}")
        
        if report.failure_analysis['most_missed_chunks']:
            print(f"\n  Top 5 Missed Chunks:")
            for chunk_id, count in report.failure_analysis['most_missed_chunks'][:5]:
                print(f"    {chunk_id}: {count} times")
        
        print(f"\nReports saved to: {output_dir}")
        print("\n" + "=" * 80)
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
