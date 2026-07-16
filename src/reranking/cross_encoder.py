"""
Cross-encoder reranker implementation.
"""
import logging
import time
from typing import List, Dict, Any, Optional

from .base import BaseReranker
from .models import RerankerConfig, RerankerOutput, RerankerResult
from .model_manager import get_global_model_manager

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranker using sentence-transformers.
    
    Uses cross-encoder models to re-score and rerank retrieved chunks.
    Supports batch processing for efficiency and score normalization.
    """
    
    def __init__(self, config: RerankerConfig):
        """
        Initialize cross-encoder reranker.
        
        Args:
            config: Reranker configuration
        """
        super().__init__(config)
        
        self.model_manager = get_global_model_manager()
        self.model = None
        
        try:
            self.model = self.model_manager.get_model(
                config.model_name,
                model_type="cross_encoder"
            )
            logger.info(f"CrossEncoderReranker initialized with model: {config.model_name}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            if not config.enable_fallback:
                raise
    
    def rerank(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> RerankerOutput:
        """
        Rerank a single query's retrieved chunks.
        
        Args:
            query: Original query text
            retrieved_chunks: List of retrieved chunks with metadata
            
        Returns:
            RerankerOutput with reranked results
        """
        start_time = time.time()
        
        try:
            # Limit to rerank_top_k chunks
            chunks_to_rerank = retrieved_chunks[:self.config.rerank_top_k]
            
            # Prepare query-document pairs
            pairs = [
                (query, chunk.get("text", ""))
                for chunk in chunks_to_rerank
            ]
            
            # Compute cross-encoder scores
            scores = self.model.predict(pairs)
            
            # Normalize scores if enabled
            if self.config.normalize_scores:
                scores = self._normalize_scores(scores)
            
            # Create reranked results
            results = self._create_results(
                chunks_to_rerank,
                scores,
                query
            )
            
            # Sort by reranked score and assign ranks
            results.sort(key=lambda x: x.reranked_score, reverse=True)
            for rank, result in enumerate(results, 1):
                result.reranked_rank = rank
            
            # Limit to return_top_k
            results = results[:self.config.return_top_k]
            
            # Update metrics
            self._update_metrics(results, time.time() - start_time)
            
            return RerankerOutput(
                results=results,
                query=query,
                reranking_latency=time.time() - start_time,
                fallback_used=False,
                model_name=self.config.model_name
            )
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            
            if self.config.enable_fallback:
                logger.warning("Falling back to original results")
                self._metrics.fallback_count += 1
                
                # Create results from original retrieval
                results = self._create_fallback_results(retrieved_chunks, query)
                
                return RerankerOutput(
                    results=results,
                    query=query,
                    reranking_latency=time.time() - start_time,
                    fallback_used=True,
                    model_name=self.config.model_name
                )
            else:
                raise
    
    def rerank_batch(
        self,
        queries: List[str],
        retrieved_chunks_list: List[List[Dict[str, Any]]]
    ) -> List[RerankerOutput]:
        """
        Rerank multiple queries' retrieved chunks efficiently.
        
        Args:
            queries: List of query texts
            retrieved_chunks_list: List of retrieved chunks for each query
            
        Returns:
            List of RerankerOutput for each query
        """
        outputs = []
        
        for query, chunks in zip(queries, retrieved_chunks_list):
            output = self.rerank(query, chunks)
            outputs.append(output)
        
        return outputs
    
    def validate_config(self) -> bool:
        """
        Validate that the reranker configuration is valid and the model is accessible.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Test with a minimal prediction
            test_pairs = [("test query", "test document")]
            self.model.predict(test_pairs)
            logger.info("CrossEncoderReranker configuration validated successfully")
            return True
        except Exception as e:
            logger.error(f"CrossEncoderReranker configuration validation failed: {e}")
            return False
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range using min-max normalization.
        
        Args:
            scores: Original scores
            
        Returns:
            Normalized scores
        """
        if not scores:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All scores are the same
            return [0.5] * len(scores)
        
        normalized = [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]
        
        return normalized
    
    def _create_results(
        self,
        chunks: List[Dict[str, Any]],
        scores: List[float],
        query: str
    ) -> List[RerankerResult]:
        """
        Create RerankerResult objects from chunks and scores.
        
        Args:
            chunks: Retrieved chunks
            scores: Reranked scores
            query: Original query
            
        Returns:
            List of RerankerResult objects
        """
        results = []
        
        for idx, (chunk, score) in enumerate(zip(chunks, scores)):
            original_score = chunk.get("score", 0.0)
            original_rank = idx + 1
            
            result = RerankerResult(
                chunk_id=chunk.get("chunk_id", ""),
                original_rank=original_rank,
                reranked_rank=0,  # Will be assigned after sorting
                original_score=original_score,
                reranked_score=score,
                score_delta=score - original_score,
                text=chunk.get("text", ""),
                metadata={
                    k: v for k, v in chunk.items()
                    if k not in ["chunk_id", "text", "score"]
                }
            )
            results.append(result)
        
        return results
    
    def _create_fallback_results(
        self,
        chunks: List[Dict[str, Any]],
        query: str
    ) -> List[RerankerResult]:
        """
        Create fallback results when reranking fails.
        
        Args:
            chunks: Retrieved chunks
            query: Original query
            
        Returns:
            List of RerankerResult objects with original scores
        """
        results = []
        
        for idx, chunk in enumerate(chunks[:self.config.return_top_k]):
            original_score = chunk.get("score", 0.0)
            
            result = RerankerResult(
                chunk_id=chunk.get("chunk_id", ""),
                original_rank=idx + 1,
                reranked_rank=idx + 1,
                original_score=original_score,
                reranked_score=original_score,
                score_delta=0.0,
                text=chunk.get("text", ""),
                metadata={
                    k: v for k, v in chunk.items()
                    if k not in ["chunk_id", "text", "score"]
                }
            )
            results.append(result)
        
        return results
    
    def _update_metrics(self, results: List[RerankerResult], latency: float) -> None:
        """
        Update reranking metrics.
        
        Args:
            results: Reranked results
            latency: Reranking latency
        """
        self._metrics.total_queries += 1
        self._metrics.reranking_latency += latency
        
        # Calculate score improvements
        score_improvements = [r.score_delta for r in results]
        if score_improvements:
            self._metrics.avg_score_improvement = sum(score_improvements) / len(score_improvements)
        
        # Calculate rank movements
        rank_movements = [abs(r.reranked_rank - r.original_rank) for r in results]
        if rank_movements:
            self._metrics.avg_rank_movement = sum(rank_movements) / len(rank_movements)
            self._metrics.largest_rank_movement = max(rank_movements)
        
        # Calculate reordering percentage
        reordered = sum(1 for r in results if r.reranked_rank != r.original_rank)
        self._metrics.reordering_percentage = (reordered / len(results)) * 100 if results else 0
