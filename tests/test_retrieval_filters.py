"""Regression tests for source selection on detailed allocation queries."""
from src.retrieval.filters import FilterBuilder, FilterConverter
from src.retrieval.models import RetrievalConfig
from src.retrieval.retriever import DenseRetriever
from src.rag.query_classification import requires_expenditure_sources


class StubEmbedder:
    model_name = "test"

    def embed_query(self, query):
        return [0.0]


class StubVectorStore:
    def __init__(self):
        self.search_arguments = None

    def collection_exists(self):
        return True

    def search_dense(self, **kwargs):
        self.search_arguments = kwargs
        return []


def test_detailed_allocation_phrasing_is_detected():
    assert requires_expenditure_sources(
        "What are the detailed allocations for the Ministry of Health and Family Welfare?"
    )
    assert requires_expenditure_sources("Show the Budget Estimate for Demand No. 46")
    assert requires_expenditure_sources("Give a scheme-wise breakdown for health")
    assert not requires_expenditure_sources("What did the budget speech announce for health?")


def test_retriever_does_not_filter_at_search_time():
    """
    Source selection for detailed-allocation queries happens post-retrieval
    (RAGPipeline.query, see src/rag/pipeline.py) rather than as a Qdrant-level
    filter, so that a query never comes back empty just because the only
    semantic match was budget_speech. The retriever itself must pass filters
    through unchanged.
    """
    store = StubVectorStore()
    retriever = DenseRetriever(StubEmbedder(), store, "test")
    config = RetrievalConfig(filters={"year": 2026})

    context = retriever.retrieve(
        "What are the detailed allocations for the Ministry of Health and Family Welfare?",
        config,
    )

    assert store.search_arguments["filters"] == {"year": 2026}
    assert context.filters == {"year": 2026}


def test_not_equal_filter_becomes_a_qdrant_must_not_condition():
    qdrant_filter = FilterConverter.to_qdrant_filter(
        FilterBuilder.from_dict({"document": {"ne": "budget_speech"}})
    )

    assert qdrant_filter.must is None
    assert len(qdrant_filter.must_not) == 1
    assert qdrant_filter.must_not[0].key == "document"
    assert qdrant_filter.must_not[0].match.value == "budget_speech"
