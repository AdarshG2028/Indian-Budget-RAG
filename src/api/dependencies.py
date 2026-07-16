"""
Dependency injection for FastAPI application.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import Depends, Request
from typing import Generator

from .config import settings
from embeddings.embedder import BGEEmbedder
from retrieval import DenseRetriever, QdrantVectorStore
from reranking import CrossEncoderReranker, RerankerConfig
from llm import GroqLLM, LLMConfig
from rag import RAGPipeline, RAGConfig
from evaluation import EvaluationRunner
from evaluation.retrieval import RecallAtK, PrecisionAtK, MRR, NDCG, HitRate, MAP


# Global instances (lazy initialization)
_embedder: BGEEmbedder = None
_vector_store: QdrantVectorStore = None
_retriever: DenseRetriever = None
_reranker: CrossEncoderReranker = None
_llm: GroqLLM = None
_pipeline: RAGPipeline = None
_evaluation_runner: EvaluationRunner = None


def get_embedder() -> BGEEmbedder:
    """Get or create embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = BGEEmbedder(
            model_name=settings.embedding_model,
            device=settings.embedding_device
        )
    return _embedder


def get_vector_store() -> QdrantVectorStore:
    """Get or create vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = QdrantVectorStore(
            collection_name=settings.qdrant_collection,
            embedding_dim=settings.embedding_dim,
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
    return _vector_store


def get_retriever(
    embedder: BGEEmbedder = Depends(get_embedder),
    vector_store: QdrantVectorStore = Depends(get_vector_store)
) -> DenseRetriever:
    """Get or create retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = DenseRetriever(
            embedder=embedder,
            vector_store=vector_store,
            collection_name=settings.qdrant_collection
        )
    return _retriever


def get_reranker() -> CrossEncoderReranker:
    """Get or create reranker instance."""
    global _reranker
    if _reranker is None and settings.enable_reranking:
        reranker_config = RerankerConfig(
            model_name=settings.reranker_model,
            device=settings.reranker_device,
            retrieve_top_k=settings.retrieve_top_k,
            rerank_top_k=settings.rerank_top_k,
            return_top_k=settings.return_top_k,
            normalize_scores=True,
            enable_fallback=True
        )
        _reranker = CrossEncoderReranker(reranker_config)
    return _reranker


def get_llm() -> GroqLLM:
    """Get or create LLM instance."""
    global _llm
    if _llm is None:
        llm_config = LLMConfig(
            model_name=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.groq_api_key
        )
        _llm = GroqLLM(llm_config)
    return _llm


def get_rag_pipeline(
    retriever: DenseRetriever = Depends(get_retriever),
    llm: GroqLLM = Depends(get_llm),
    reranker: CrossEncoderReranker = Depends(get_reranker)
) -> RAGPipeline:
    """Get or create RAG pipeline instance."""
    global _pipeline
    if _pipeline is None:
        rag_config = RAGConfig(
            retrieval_top_k=settings.retrieve_top_k,
            reranker_enabled=settings.enable_reranking,
            context_max_tokens=settings.context_max_tokens,
            llm_temperature=settings.llm_temperature,
            llm_max_tokens=settings.llm_max_tokens,
            prompt_template=settings.prompt_template,
            include_citations=settings.include_citations
        )
        _pipeline = RAGPipeline(
            retriever=retriever,
            llm=llm,
            reranker=reranker,
            config=rag_config
        )
    return _pipeline


def get_evaluation_runner(
    retriever: DenseRetriever = Depends(get_retriever),
    reranker: CrossEncoderReranker = Depends(get_reranker)
) -> EvaluationRunner:
    """Get or create evaluation runner instance."""
    global _evaluation_runner
    if _evaluation_runner is None:
        # Create metrics
        metrics = [
            MRR(min_relevance=1),
            RecallAtK(k=1, min_relevance=1),
            RecallAtK(k=5, min_relevance=1),
            RecallAtK(k=10, min_relevance=1),
            PrecisionAtK(k=1, min_relevance=1),
            PrecisionAtK(k=5, min_relevance=1),
            PrecisionAtK(k=10, min_relevance=1),
            NDCG(k=1, min_relevance=1),
            NDCG(k=5, min_relevance=1),
            NDCG(k=10, min_relevance=1),
            HitRate(k=1, min_relevance=1),
            HitRate(k=5, min_relevance=1),
            HitRate(k=10, min_relevance=1),
            MAP(k=1, min_relevance=1),
            MAP(k=5, min_relevance=1),
            MAP(k=10, min_relevance=1)
        ]
        
        _evaluation_runner = EvaluationRunner(
            retriever=retriever,
            metrics=metrics,
            reranker=reranker
        )
    return _evaluation_runner


def get_request_id(request: Request) -> str:
    """Get request ID from request state."""
    return getattr(request.state, "request_id", "unknown")


def get_settings() -> settings:
    """Get application settings."""
    return settings
