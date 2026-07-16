"""
Evaluation runner for orchestrating retrieval evaluation.
"""
import logging
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Type
from pathlib import Path

from retrieval import BaseRetriever, RetrievalConfig
from reranking import BaseReranker
from .models import (
    EvaluationQuery, RetrievedChunk, QueryResult, 
    ExperimentMetadata, EvaluationReport
)
from .retrieval.base import BaseMetric
from .data_loader import DataLoader

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """
    Orchestrates retrieval evaluation experiments.
    
    Responsibilities:
    - Load evaluation datasets
    - Execute retriever for each query
    - Compute metrics for each query
    - Aggregate results across queries
    - Generate experiment metadata
    - Create evaluation reports
    """
    
    def __init__(
        self,
        retriever: BaseRetriever,
        metrics: List[BaseMetric],
        reranker: Optional[BaseReranker] = None,
        experiment_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize evaluation runner.
        
        Args:
            retriever: Retrieval component to evaluate
            metrics: List of metrics to compute
            reranker: Optional reranker component
            experiment_metadata: Additional experiment metadata
        """
        self.retriever = retriever
        self.reranker = reranker
        self.metrics = metrics
        self.experiment_metadata = experiment_metadata or {}
        
        reranker_status = "enabled" if self.reranker else "disabled"
        logger.info(f"EvaluationRunner initialized with {len(metrics)} metrics, reranker: {reranker_status}")
    
    def run(
        self,
        dataset_path: str,
        retrieval_config: Optional[RetrievalConfig] = None,
        top_k: int = 10
    ) -> EvaluationReport:
        """
        Run evaluation experiment.
        
        Args:
            dataset_path: Path to evaluation dataset
            retrieval_config: Retrieval configuration
            top_k: Number of results to retrieve
            
        Returns:
            EvaluationReport with complete results
        """
        logger.info(f"Starting evaluation experiment")
        start_time = time.time()
        
        # Load dataset
        logger.info(f"Loading dataset from: {dataset_path}")
        queries = DataLoader.load(dataset_path)
        logger.info(f"Loaded {len(queries)} queries")
        
        # Set default retrieval config
        if retrieval_config is None:
            retrieval_config = RetrievalConfig(top_k=top_k)
        else:
            retrieval_config.top_k = top_k
        
        # Evaluate each query
        query_results = []
        for idx, query in enumerate(queries, 1):
            logger.info(f"Evaluating query {idx}/{len(queries)}: {query.query_id}")
            query_result = self._evaluate_query(query, retrieval_config)
            query_results.append(query_result)
        
        # Compute aggregate metrics
        aggregate_metrics = self._compute_aggregate_metrics(query_results)
        
        # Compute latency metrics
        latency_metrics = self._compute_latency_metrics(query_results)
        
        # Generate failure analysis
        failure_analysis = self._generate_failure_analysis(query_results)
        
        # Create experiment metadata
        experiment_metadata = self._create_experiment_metadata(
            dataset_path,
            retrieval_config,
            time.time() - start_time
        )
        
        # Create evaluation report
        report = EvaluationReport(
            metadata=experiment_metadata,
            query_results=query_results,
            aggregate_metrics=aggregate_metrics,
            failure_analysis=failure_analysis,
            latency_metrics=latency_metrics
        )
        
        logger.info(f"Evaluation completed in {time.time() - start_time:.2f}s")
        
        return report
    
    def _evaluate_query(
        self,
        query: EvaluationQuery,
        retrieval_config: RetrievalConfig
    ) -> QueryResult:
        """
        Evaluate a single query.
        
        Args:
            query: Evaluation query
            retrieval_config: Retrieval configuration
            
        Returns:
            QueryResult with metrics and retrieved chunks
        """
        start_time = time.time()
        
        # Execute retrieval
        retrieval_context = self.retriever.retrieve(query.query_text, retrieval_config)
        
        # Convert retrieval results to RetrievedChunk objects
        retrieved_chunks = [
            RetrievedChunk(
                chunk_id=result.chunk_id,
                rank=result.rank,
                score=result.score,
                metadata={
                    "document": result.document,
                    "year": result.year,
                    "section": result.section,
                    "subsection": result.subsection,
                    "page_start": result.page_start,
                    "page_end": result.page_end
                }
            )
            for result in retrieval_context.results
        ]
        
        # Compute metrics
        metrics = {}
        for metric in self.metrics:
            try:
                metric_value = metric.calculate(query, retrieved_chunks)
                metrics[metric.get_name()] = metric_value
            except Exception as e:
                logger.error(f"Error computing metric {metric.get_name()}: {e}")
                metrics[metric.get_name()] = 0.0
        
        # Extract latency information
        latency = retrieval_context.metrics.retrieval_latency
        embedding_latency = retrieval_context.metrics.embedding_latency
        search_latency = retrieval_context.metrics.search_latency
        
        # Extract similarity metrics
        if retrieval_context.results:
            scores = [r.score for r in retrieval_context.results]
            avg_similarity = sum(scores) / len(scores)
            highest_similarity = max(scores)
            lowest_similarity = min(scores)
        else:
            avg_similarity = 0.0
            highest_similarity = 0.0
            lowest_similarity = 0.0
        
        return QueryResult(
            query=query,
            retrieved_chunks=retrieved_chunks,
            metrics=metrics,
            latency=latency,
            embedding_latency=embedding_latency,
            search_latency=search_latency,
            chunks_returned=len(retrieved_chunks),
            avg_similarity=avg_similarity,
            highest_similarity=highest_similarity,
            lowest_similarity=lowest_similarity
        )
    
    def _compute_aggregate_metrics(self, query_results: List[QueryResult]) -> Dict[str, float]:
        """
        Compute aggregate metrics across all queries.
        
        Args:
            query_results: List of query results
            
        Returns:
            Dictionary of aggregate metric values
        """
        aggregate_metrics = {}
        
        # Get all metric names
        if not query_results:
            return aggregate_metrics
        
        metric_names = list(query_results[0].metrics.keys())
        
        # Compute mean for each metric
        for metric_name in metric_names:
            values = [qr.metrics.get(metric_name, 0.0) for qr in query_results]
            aggregate_metrics[metric_name] = sum(values) / len(values)
        
        # Compute standard deviation for each metric
        for metric_name in metric_names:
            values = [qr.metrics.get(metric_name, 0.0) for qr in query_results]
            mean = aggregate_metrics[metric_name]
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = variance ** 0.5
            aggregate_metrics[f"{metric_name}_std"] = std
        
        return aggregate_metrics
    
    def _compute_latency_metrics(self, query_results: List[QueryResult]) -> Dict[str, float]:
        """
        Compute latency metrics across all queries.
        
        Args:
            query_results: List of query results
            
        Returns:
            Dictionary of latency metrics
        """
        if not query_results:
            return {}
        
        latencies = [qr.latency for qr in query_results]
        embedding_latencies = [qr.embedding_latency for qr in query_results]
        search_latencies = [qr.search_latency for qr in query_results]
        
        return {
            "avg_latency": sum(latencies) / len(latencies),
            "avg_embedding_latency": sum(embedding_latencies) / len(embedding_latencies),
            "avg_search_latency": sum(search_latencies) / len(search_latencies),
            "total_latency": sum(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies)
        }
    
    def _generate_failure_analysis(self, query_results: List[QueryResult]) -> Dict[str, Any]:
        """
        Generate failure analysis.
        
        Args:
            query_results: List of query results
            
        Returns:
            Dictionary with failure analysis data
        """
        zero_recall_queries = [qr for qr in query_results if qr.get_zero_recall()]
        
        # Get most frequently missed chunks
        miss_counts: Dict[str, int] = {}
        for qr in query_results:
            for chunk_id in qr.get_missing_chunks():
                miss_counts[chunk_id] = miss_counts.get(chunk_id, 0) + 1
        
        most_missed = sorted(miss_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Get queries with poor ranking (first relevant chunk beyond rank 5)
        poor_ranking_queries = []
        for qr in query_results:
            relevant_ids = qr.query.get_relevant_chunk_ids()
            first_relevant_rank = None
            for chunk in qr.retrieved_chunks:
                if chunk.chunk_id in relevant_ids:
                    first_relevant_rank = chunk.rank
                    break
            
            if first_relevant_rank and first_relevant_rank > 5:
                poor_ranking_queries.append({
                    "query_id": qr.query.query_id,
                    "first_relevant_rank": first_relevant_rank
                })
        
        return {
            "zero_recall_count": len(zero_recall_queries),
            "zero_recall_queries": [qr.query.query_id for qr in zero_recall_queries],
            "most_missed_chunks": most_missed,
            "poor_ranking_queries": poor_ranking_queries,
            "total_queries": len(query_results)
        }
    
    def _create_experiment_metadata(
        self,
        dataset_path: str,
        retrieval_config: RetrievalConfig,
        duration: float
    ) -> ExperimentMetadata:
        """
        Create experiment metadata.
        
        Args:
            dataset_path: Path to dataset used
            retrieval_config: Retrieval configuration
            duration: Experiment duration in seconds
            
        Returns:
            ExperimentMetadata object
        """
        # Extract retriever info
        retriever_name = self.retriever.__class__.__name__
        
        # Try to get embedding model info
        embedding_model = "unknown"
        if hasattr(self.retriever, 'embedder'):
            if hasattr(self.retriever.embedder, 'model_name'):
                embedding_model = self.retriever.embedder.model_name
        
        # Try to get vector store info
        vector_db = "unknown"
        if hasattr(self.retriever, 'vector_store'):
            vector_db = self.retriever.vector_store.__class__.__name__
        
        # Try to get collection info
        collection = "unknown"
        if hasattr(self.retriever, 'collection_name'):
            collection = self.retriever.collection_name
        
        # Get reranker info
        reranker_name = None
        reranker_config = None
        if self.reranker:
            reranker_name = self.reranker.__class__.__name__
            if hasattr(self.reranker, 'config'):
                reranker_config = {
                    "model_name": self.reranker.config.model_name,
                    "device": self.reranker.config.device,
                    "retrieve_top_k": self.reranker.config.retrieve_top_k,
                    "rerank_top_k": self.reranker.config.rerank_top_k,
                    "return_top_k": self.reranker.config.return_top_k,
                    "normalize_scores": self.reranker.config.normalize_scores
                }
        
        return ExperimentMetadata(
            experiment_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            embedding_model=embedding_model,
            retriever=retriever_name,
            reranker=reranker_name,
            vector_database=vector_db,
            chunk_size=0,  # Not available from retriever
            chunk_overlap=0,  # Not available from retriever
            collection=collection,
            dataset=Path(dataset_path).name,
            retrieval_config={
                "top_k": retrieval_config.top_k,
                "score_threshold": retrieval_config.score_threshold,
                "filters": retrieval_config.filters
            },
            additional_metadata={
                **self.experiment_metadata,
                "reranker_config": reranker_config
            }
        )
