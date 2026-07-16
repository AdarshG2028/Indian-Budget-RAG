"""
RAG service for business logic.
"""
import logging
from typing import Optional, Dict, Any, AsyncGenerator

from rag import RAGPipeline
from .streaming import StreamingService

logger = logging.getLogger(__name__)


class RAGService:
    """
    Service for RAG operations.
    
    Handles business logic for RAG queries, both streaming and non-streaming.
    """
    
    def __init__(self, pipeline: RAGPipeline):
        """
        Initialize RAG service.
        
        Args:
            pipeline: RAG pipeline instance
        """
        self.pipeline = pipeline
    
    async def query(
        self,
        question: str,
        request_id: str,
        stream: bool = False,
        config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Execute RAG query with optional streaming.
        
        Args:
            question: User question
            request_id: Unique request identifier
            stream: Whether to stream response
            config: Optional configuration override
            
        Yields:
            SSE events if streaming, or complete response if not
        """
        streaming_service = StreamingService(request_id)
        
        try:
            if stream:
                async for event in self._query_stream(question, streaming_service, config):
                    yield event
            else:
                response = self._query_sync(question, config)
                yield response
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            await streaming_service.emit_error(
                error_code="RAG_ERROR",
                message=str(e),
                details={"question": question}
            )
            if stream:
                yield streaming_service.get_emitter().get_events()[-1]
            raise
        finally:
            await streaming_service.close()
    
    async def _query_stream(
        self,
        question: str,
        streaming_service: StreamingService,
        config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Execute RAG query with streaming.
        
        Args:
            question: User question
            streaming_service: Streaming service for event emission
            config: Optional configuration override
            
        Yields:
            SSE events
        """
        # Emit retrieval started
        await streaming_service.emit_retrieval_started(
            query=question,
            top_k=config.get("top_k", 20) if config else 20
        )
        
        # Execute RAG pipeline (non-streaming for now, will be updated later)
        response = self.pipeline.query(question, stream=False)
        
        # Emit retrieval completed
        await streaming_service.emit_retrieval_completed(
            chunks_retrieved=len(response.retrieval_context.results),
            latency=response.retrieval_context.metrics.retrieval_latency,
            top_chunks=[
                {
                    "chunk_id": r.chunk_id,
                    "score": r.score,
                    "text": r.text[:100]
                }
                for r in response.retrieval_context.results[:3]
            ]
        )
        
        # Emit reranking info if used
        if response.reranker_used:
            await streaming_service.emit_reranking_started(
                model=response.reranker_metrics.get("model_name", "unknown"),
                chunks_to_rerank=response.reranker_metrics.get("rerank_top_k", 0)
            )
            
            await streaming_service.emit_reranking_completed(
                latency=response.reranker_metrics.get("reranking_latency", 0),
                chunks_reranked=response.reranker_metrics.get("rerank_top_k", 0),
                fallback_used=response.reranker_metrics.get("fallback_count", 0) > 0
            )
        
        # Emit context ready
        await streaming_service.emit_context_ready(
            context_tokens=response.llm_metrics.prompt_tokens,
            citations_count=len(response.citations) if response.citations else 0
        )
        
        # Emit generation started
        await streaming_service.emit_generation_started(
            model=response.llm_response.model,
            prompt_tokens=response.llm_metrics.prompt_tokens
        )
        
        # Stream tokens (simulate for now, will be updated with actual streaming)
        tokens = response.answer.split()
        for i, token in enumerate(tokens):
            is_first = (i == 0)
            is_last = (i == len(tokens) - 1)
            await streaming_service.emit_token(
                token=token + " ",
                index=i,
                is_first=is_first,
                is_last=is_last
            )
            yield streaming_service.get_emitter().get_events()[-1]
        
        # Emit generation completed
        await streaming_service.emit_generation_completed(
            total_tokens=response.llm_metrics.total_tokens,
            completion_tokens=response.llm_metrics.completion_tokens,
            latency=response.llm_metrics.llm_latency
        )
        yield streaming_service.get_emitter().get_events()[-1]
    
    def _query_sync(
        self,
        question: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute RAG query synchronously.
        
        Args:
            question: User question
            config: Optional configuration override
            
        Returns:
            Complete response dictionary
        """
        response = self.pipeline.query(question, stream=False)
        
        # Convert citations to list format for API response
        citations_list = []
        if response.citations:
            if isinstance(response.citations, dict):
                # Convert dict to list of citation objects
                for chunk_id, citation_data in response.citations.items():
                    if isinstance(citation_data, dict):
                        citations_list.append({
                            "chunk_id": chunk_id,
                            **citation_data
                        })
                    else:
                        citations_list.append(citation_data)
            elif isinstance(response.citations, list):
                citations_list = response.citations
        
        return {
            "answer": response.answer,
            "citations": citations_list,
            "metrics": response.llm_metrics.to_dict(),
            "reranker_used": response.reranker_used,
            "reranker_metrics": response.reranker_metrics
        }
