"""
RAG pipeline orchestrating retrieval, context building, and LLM generation.
"""
import logging
import time
from typing import Optional, Iterator, Dict, Any
from dataclasses import dataclass

from retrieval import BaseRetriever, RetrievalConfig, RetrievalContext
from llm.base_llm import BaseLLM
from llm.models import LLMConfig, LLMResponse, LLMMetrics
from llm.context_builder import ContextBuilder, ContextBuilderConfig
from llm.prompts import PromptTemplateRegistry
from reranking import BaseReranker, RerankerConfig

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """
    Configuration for RAG pipeline.
    """
    retrieval_top_k: int = 20
    retrieval_score_threshold: Optional[float] = None
    reranker_enabled: bool = False
    reranker_config: Optional[RerankerConfig] = None
    context_max_tokens: int = 4000
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024
    prompt_template: str = "qa"
    include_citations: bool = True


@dataclass
class RAGResponse:
    """
    Complete response from RAG pipeline.
    """
    answer: str
    retrieval_context: RetrievalContext
    llm_response: LLMResponse
    llm_metrics: LLMMetrics
    citations: Optional[Dict[str, Any]] = None
    prompt_construction_time: float = 0.0
    reranker_used: bool = False
    reranker_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for logging/export."""
        return {
            "answer": self.answer,
            "retrieval_context": self.retrieval_context.to_dict(),
            "llm_response": self.llm_response.to_dict(),
            "llm_metrics": self.llm_metrics.to_dict(),
            "citations": self.citations,
            "prompt_construction_time": self.prompt_construction_time,
            "reranker_used": self.reranker_used,
            "reranker_metrics": self.reranker_metrics
        }


class RAGPipeline:
    """
    Complete RAG pipeline orchestrating retrieval, reranking, context building, and LLM generation.
    
    Pipeline flow:
    User Query → Retriever → [Optional Reranker] → Context Builder → Prompt Template → LLM → Response with citations
    
    Responsibilities:
    - Orchestrate retrieval with configurable parameters
    - Optional reranking for improved relevance
    - Build formatted context from retrieved chunks
    - Apply appropriate prompt templates
    - Generate responses via LLM
    - Include source citations
    - Track metrics across all stages
    - Handle errors gracefully
    """
    
    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        context_builder: Optional[ContextBuilder] = None,
        reranker: Optional[BaseReranker] = None,
        config: Optional[RAGConfig] = None
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            retriever: Retrieval component
            llm: LLM component
            context_builder: Optional context builder (creates default if None)
            reranker: Optional reranker component
            config: RAG pipeline configuration
        """
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker
        self.config = config or RAGConfig()
        
        # Create context builder with config
        context_config = ContextBuilderConfig(
            max_context_tokens=self.config.context_max_tokens,
            include_citations=self.config.include_citations
        )
        self.context_builder = context_builder or ContextBuilder(context_config)
        
        reranker_status = "enabled" if self.reranker else "disabled"
        logger.info(
            f"RAGPipeline initialized with template: {self.config.prompt_template}, "
            f"retrieval_top_k: {self.config.retrieval_top_k}, "
            f"reranker: {reranker_status}"
        )
    
    def query(self, question: str, stream: bool = False) -> RAGResponse:
        """
        Execute complete RAG pipeline for a question.
        
        Args:
            question: User question
            stream: Whether to stream LLM response (not implemented yet)
            
        Returns:
            RAGResponse with answer and metadata
        """
        if stream:
            raise NotImplementedError("Streaming not yet implemented")
        
        total_start = time.time()
        
        try:
            # Step 1: Retrieval
            logger.info(f"Starting retrieval for question: '{question}'")
            retrieval_config = RetrievalConfig(
                top_k=self.config.retrieval_top_k,
                score_threshold=self.config.retrieval_score_threshold
            )
            retrieval_context = self.retriever.retrieve(question, retrieval_config)
            
            if not retrieval_context.results:
                logger.warning("No results retrieved from retriever")
                return self._handle_empty_retrieval(question, retrieval_context)
            
            # Step 2: Optional Reranking
            reranker_used = False
            reranker_metrics = None
            final_results = retrieval_context.results
            
            if self.reranker and self.config.reranker_enabled:
                logger.info("Reranking retrieved chunks")
                rerank_start = time.time()
                
                try:
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
                    
                    # Perform reranking
                    reranker_output = self.reranker.rerank(question, chunks_for_reranking)
                    
                    # Convert reranked results back to RetrievalResult format
                    final_results = self._convert_reranker_results(reranker_output.results)
                    
                    reranker_used = True
                    reranker_metrics = self.reranker.get_metrics().to_dict()
                    
                    logger.info(
                        f"Reranking completed in {reranker_output.reranking_latency:.3f}s, "
                        f"fallback: {reranker_output.fallback_used}"
                    )
                    
                except Exception as e:
                    logger.error(f"Reranking failed: {e}, using original results")
                    reranker_used = False
            
            # Step 3: Context building
            logger.info("Building context from retrieved chunks")
            context_start = time.time()
            
            if self.config.include_citations:
                context, citations = self.context_builder.build_context_with_citations(
                    final_results,
                    max_tokens=self.config.context_max_tokens
                )
            else:
                context = self.context_builder.build_context(
                    final_results,
                    max_tokens=self.config.context_max_tokens
                )
                citations = None
            
            context_time = time.time() - context_start
            
            # Step 4: Prompt construction
            logger.info("Constructing prompt with template")
            prompt_start = time.time()
            
            template = PromptTemplateRegistry.get_template(self.config.prompt_template)
            
            if self.config.prompt_template == "qa":
                prompt = template.format(question=question, context=context)
            elif self.config.prompt_template == "summarization":
                prompt = template.format(context=context)
            elif self.config.prompt_template == "comparison":
                # For comparison, we'd need two contexts - not implemented yet
                raise NotImplementedError("Comparison template requires two contexts")
            elif self.config.prompt_template == "analysis":
                prompt = template.format(question=question, context=context)
            else:
                prompt = template.format(question=question, context=context)
            
            prompt_time = time.time() - prompt_start
            
            # Step 5: LLM generation
            logger.info("Generating response via LLM")
            llm_start = time.time()
            llm_response = self.llm.generate(prompt)
            llm_time = time.time() - llm_start
            
            # Step 6: Build metrics
            total_time = time.time() - total_start
            
            llm_metrics = LLMMetrics(
                prompt_construction_time=prompt_time,
                llm_latency=llm_time,
                total_latency=total_time,
                prompt_tokens=llm_response.prompt_tokens,
                completion_tokens=llm_response.completion_tokens,
                total_tokens=llm_response.total_tokens,
                model=llm_response.model
            )
            
            # Step 7: Build response
            rag_response = RAGResponse(
                answer=llm_response.text,
                retrieval_context=retrieval_context,
                llm_response=llm_response,
                llm_metrics=llm_metrics,
                citations=citations,
                prompt_construction_time=prompt_time,
                reranker_used=reranker_used,
                reranker_metrics=reranker_metrics
            )
            
            reranking_info = ""
            if reranker_used and reranker_metrics:
                reranking_info = f", reranking: {reranker_metrics.get('reranking_latency', 0):.3f}s"
            
            logger.info(
                f"RAG pipeline completed in {total_time:.3f}s "
                f"(retrieval: {retrieval_context.metrics.retrieval_latency:.3f}s"
                f"{reranking_info}, "
                f"context: {context_time:.3f}s, "
                f"prompt: {prompt_time:.3f}s, "
                f"llm: {llm_time:.3f}s)"
            )
            
            return rag_response
            
        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            raise
    
    def _convert_reranker_results(self, reranker_results) -> list:
        """
        Convert reranker results back to RetrievalResult format.
        
        Args:
            reranker_results: List of RerankerResult objects
            
        Returns:
            List of RetrievalResult objects
        """
        from retrieval.models import RetrievalResult
        
        results = []
        for rr in reranker_results:
            result = RetrievalResult(
                chunk_id=rr.chunk_id,
                document=rr.metadata.get("document", ""),
                year=rr.metadata.get("year", 0),
                section=rr.metadata.get("section", ""),
                subsection=rr.metadata.get("subsection", ""),
                paragraph_start=0,
                paragraph_end=0,
                page_start=rr.metadata.get("page_start", 0),
                page_end=rr.metadata.get("page_end", 0),
                score=rr.reranked_score,
                rank=rr.reranked_rank,
                text=rr.text,
                metadata=rr.metadata
            )
            results.append(result)
        
        return results
    
    def query_stream(self, question: str) -> Iterator[str]:
        """
        Execute RAG pipeline with streaming LLM response.
        
        Args:
            question: User question
            
        Yields:
            Response chunks as they are generated
        """
        # Step 1: Retrieval (non-streaming)
        logger.info(f"Starting retrieval for question: '{question}'")
        retrieval_config = RetrievalConfig(
            top_k=self.config.retrieval_top_k,
            score_threshold=self.config.retrieval_score_threshold
        )
        retrieval_context = self.retriever.retrieve(question, retrieval_config)
        
        if not retrieval_context.results:
            logger.warning("No results retrieved from retriever")
            yield "No relevant information found in the budget documents."
            return
        
        # Step 2: Context building
        logger.info("Building context from retrieved chunks")
        if self.config.include_citations:
            context, citations = self.context_builder.build_context_with_citations(
                retrieval_context.results,
                max_tokens=self.config.context_max_tokens
            )
        else:
            context = self.context_builder.build_context(
                retrieval_context.results,
                max_tokens=self.config.context_max_tokens
            )
            citations = None
        
        # Step 3: Prompt construction
        logger.info("Constructing prompt with template")
        template = PromptTemplateRegistry.get_template(self.config.prompt_template)
        
        if self.config.prompt_template == "qa":
            prompt = template.format(question=question, context=context)
        else:
            prompt = template.format(question=question, context=context)
        
        # Step 4: Streaming LLM generation
        logger.info("Streaming response via LLM")
        try:
            for chunk in self.llm.stream(prompt):
                yield chunk.delta
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise
    
    def _handle_empty_retrieval(
        self, 
        question: str, 
        retrieval_context: RetrievalContext
    ) -> RAGResponse:
        """
        Handle case when no results are retrieved.
        
        Args:
            question: Original question
            retrieval_context: Empty retrieval context
            
        Returns:
            RAGResponse with appropriate message
        """
        from llm.models import LLMResponse, LLMMetrics
        
        llm_response = LLMResponse(
            text="I couldn't find relevant information in the budget documents to answer your question.",
            model=self.llm.config.model_name,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency=0.0,
            finish_reason="no_results"
        )
        
        llm_metrics = LLMMetrics(
            prompt_construction_time=0.0,
            llm_latency=0.0,
            total_latency=retrieval_context.metrics.retrieval_latency,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            model=self.llm.config.model_name
        )
        
        return RAGResponse(
            answer=llm_response.text,
            retrieval_context=retrieval_context,
            llm_response=llm_response,
            llm_metrics=llm_metrics,
            citations=None,
            prompt_construction_time=0.0
        )
