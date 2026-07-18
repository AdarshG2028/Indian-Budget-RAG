import uuid
from .models import ChunkData

def generate_deterministic_uuid(namespace_str: str, name: str) -> str:
    """
    Generates a deterministic UUID based on a namespace and name.
    This ensures that running the pipeline multiple times yields the same IDs,
    preventing duplicates in Qdrant.
    """
    namespace_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, namespace_str)
    return str(uuid.uuid5(namespace_uuid, name))

def format_text_for_embedding(chunk: ChunkData) -> str:
    """
    Constructs the natural semantic context for embedding without using explicit metadata labels.
    
    Example:
    Document Type: Budget Speech
    Budget Speech
    Part A
    Tourism
    
    The Tourism sector has the potential...
    """
    parts = []
    
    # 1. Document Type Detection (for better retrieval)
    doc_type = "General Document"
    if chunk.document:
        if "budget_speech" in chunk.document.lower():
            doc_type = "Budget Speech"
        elif "expenditure" in chunk.document.lower():
            doc_type = "Expenditure Budget"
        elif "receipt" in chunk.document.lower():
            doc_type = "Receipt Budget"
        elif "key_features" in chunk.document.lower():
            doc_type = "Key Features"
        elif "memorandum" in chunk.document.lower():
            doc_type = "Budget Memorandum"
        elif "glance" in chunk.document.lower():
            doc_type = "Budget at a Glance"
        
        parts.append(f"Document Type: {doc_type}")
        
        # 2. Document Name (Formatted for readability)
        doc_title = chunk.document.replace("_", " ").title()
        parts.append(doc_title)
    
    # 3. Heading Path
    if chunk.heading_path:
        # Compare against the humanized form ("budget_speech" -> "budget speech")
        # so a heading like "Budget Speech" is recognized as the document name.
        humanized_document = chunk.document.replace("_", " ").lower() if chunk.document else ""
        for heading in chunk.heading_path:
            # Avoid repeating the document name if it's already in the heading path
            if humanized_document and heading.lower() not in humanized_document:
                parts.append(heading)
            elif not humanized_document:
                parts.append(heading)
    
    # 4. Add the actual text content with some separation
    parts.append("\n" + chunk.text.strip())
    
    # Join everything with newlines
    return "\n".join(parts)
