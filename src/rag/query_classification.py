"""
Query classification heuristics for source selection.

This lives in the RAG orchestration layer (not retrieval) because it encodes
domain knowledge about budget-document types, while the retrieval layer is
deliberately query-agnostic.

NOTE: keyword matching is a stopgap. Phrasings outside this list fall back
to plain semantic ranking, where narrative budget_speech chunks tend to
outscore tabular expenditure data. The durable fix is hybrid (dense+sparse)
retrieval; until then, extend this list as new phrasings show up.
"""

DETAILED_ALLOCATION_KEYWORDS = (
    "detailed allocation",
    "detailed breakdown",
    "allocation breakdown",
    "breakdown of allocation",
    "budget estimate",
    "revised estimate",
    "demand no",
    "demand number",
    "line item",
    "scheme-wise",
    "scheme wise",
    "department-wise",
    "department wise",
    "ministry-wise",
    "ministry wise",
    "how much has been allocated",
    "how much is allocated",
    "how much was allocated",
    "allocated to each",
    "outlay",
)

# Document types whose chunks are narrative prose rather than tabular data.
# Queries asking for detailed figures should prefer other sources.
NARRATIVE_DOCUMENTS = frozenset({"budget_speech"})


def requires_expenditure_sources(query: str) -> bool:
    """Return whether a query asks for tabular/detailed budget figures."""
    normalized_query = " ".join(query.lower().split())
    return any(keyword in normalized_query for keyword in DETAILED_ALLOCATION_KEYWORDS)


def is_narrative_document(document: str) -> bool:
    """Return whether a document name refers to a narrative (non-tabular) source."""
    return document in NARRATIVE_DOCUMENTS
