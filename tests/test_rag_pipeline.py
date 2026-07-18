"""Regression tests for the document-type fallback logic in RAGPipeline.query."""
from rag.pipeline import RAGPipeline, RAGConfig
from retrieval.models import RetrievalContext, RetrievalMetrics, RetrievalResult
from llm.models import LLMConfig, LLMResponse


def _make_result(chunk_id, document, score):
    return RetrievalResult(
        chunk_id=chunk_id,
        document=document,
        year=2026,
        section="",
        subsection="",
        paragraph_start=0,
        paragraph_end=0,
        page_start=1,
        page_end=1,
        score=score,
        rank=0,
        text=f"text for {chunk_id}",
        metadata={},
    )


class StubRetriever:
    def __init__(self, results):
        self._results = results

    def retrieve(self, query, config=None):
        metrics = RetrievalMetrics(
            avg_similarity=0.5,
            highest_similarity=0.9,
            lowest_similarity=0.1,
            chunks_returned=len(self._results),
            retrieval_latency=0.01,
            embedding_latency=0.001,
            search_latency=0.001,
        )
        return RetrievalContext(
            query=query,
            top_k=config.top_k if config else 10,
            results=self._results,
            metrics=metrics,
        )


class StubLLM:
    def __init__(self):
        self.config = LLMConfig(model_name="stub-model")
        self.last_prompt = None

    def generate(self, prompt):
        self.last_prompt = prompt
        return LLMResponse(
            text="stub answer",
            model="stub-model",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            latency=0.0,
        )

    def validate_config(self):
        return True


def _run_pipeline(results, question):
    retriever = StubRetriever(results)
    llm = StubLLM()
    pipeline = RAGPipeline(retriever=retriever, llm=llm, config=RAGConfig(retrieval_top_k=10))
    response = pipeline.query(question)
    return response, llm


def test_detailed_allocation_query_prioritizes_non_budget_speech():
    results = [
        _make_result("bs-1", "budget_speech", 0.9),
        _make_result("bs-2", "budget_speech", 0.85),
        _make_result("exp-1", "expenditure_profile_vol1", 0.4),
    ]

    response, llm = _run_pipeline(
        results, "What are the detailed allocations for the Ministry of Health and Family Welfare?"
    )

    assert "exp-1" in llm.last_prompt
    used_ids = [r.chunk_id for r in response.retrieval_context.results]
    assert used_ids == ["bs-1", "bs-2", "exp-1"]


def test_detailed_allocation_query_falls_back_to_budget_speech_when_no_other_source():
    """Excluding budget_speech must never turn a valid retrieval into an empty context."""
    results = [
        _make_result("bs-1", "budget_speech", 0.9),
        _make_result("bs-2", "budget_speech", 0.85),
    ]

    response, llm = _run_pipeline(
        results, "What are the detailed allocations for the Ministry of Health and Family Welfare?"
    )

    assert response.answer == "stub answer"
    assert "bs-1" in llm.last_prompt


def test_non_detailed_query_is_left_untouched():
    results = [
        _make_result("bs-1", "budget_speech", 0.9),
        _make_result("exp-1", "expenditure_profile_vol1", 0.4),
    ]

    response, _ = _run_pipeline(results, "What did the budget speech announce for health?")

    used_ids = [r.chunk_id for r in response.retrieval_context.results]
    assert used_ids == ["bs-1", "exp-1"]


def test_prioritization_recomputes_ranks_for_citations():
    """Citations render from result.rank, so ranks must follow the reorder."""
    results = [
        _make_result("bs-1", "budget_speech", 0.9),
        _make_result("bs-2", "budget_speech", 0.85),
        _make_result("exp-1", "expenditure_profile_vol1", 0.4),
    ]
    for original_rank, r in enumerate(results, 1):
        r.rank = original_rank

    response, _ = _run_pipeline(
        results, "What are the detailed allocations for the Ministry of Health and Family Welfare?"
    )

    assert response.citations["[1]"]["chunk_id"] == "exp-1"
    assert sorted(response.citations.keys()) == ["[1]", "[2]", "[3]"]


class StubReranker:
    """Reranker that reverses its input order — worst case for prioritization."""

    def __init__(self):
        from reranking.models import RerankerConfig
        self.config = RerankerConfig(model_name="stub-reranker")

    def rerank(self, query, chunks):
        from reranking.models import RerankerOutput, RerankerResult
        reversed_chunks = list(reversed(chunks))
        results = [
            RerankerResult(
                chunk_id=c["chunk_id"],
                original_rank=i + 1,
                reranked_rank=i + 1,
                original_score=c["score"],
                reranked_score=1.0 - i * 0.1,
                score_delta=0.0,
                text=c["text"],
                metadata={k: v for k, v in c.items() if k not in ("chunk_id", "text", "score")},
            )
            for i, c in enumerate(reversed_chunks)
        ]
        return RerankerOutput(
            results=results,
            query=query,
            reranking_latency=0.0,
            fallback_used=False,
            model_name="stub-reranker",
        )

    def get_metrics(self):
        from reranking.models import RerankerMetrics
        return RerankerMetrics(
            reranking_latency=0.0,
            avg_score_improvement=0.0,
            avg_rank_movement=0.0,
            largest_rank_movement=0,
            reordering_percentage=0.0,
            fallback_count=0,
            total_queries=1,
            model_name="stub-reranker",
        )


def test_prioritization_survives_reranking():
    """Enabling the reranker must not silently undo the document-type fix."""
    results = [
        _make_result("exp-1", "expenditure_profile_vol1", 0.4),
        _make_result("bs-1", "budget_speech", 0.9),
    ]

    retriever = StubRetriever(results)
    llm = StubLLM()
    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        reranker=StubReranker(),
        config=RAGConfig(retrieval_top_k=10, reranker_enabled=True),
    )

    response = pipeline.query(
        "What are the detailed allocations for the Ministry of Health and Family Welfare?"
    )

    assert response.reranker_used is True
    # The stub reranker put bs-1 first; prioritization must move exp-1 back on top.
    assert response.citations["[1]"]["chunk_id"] == "exp-1"
