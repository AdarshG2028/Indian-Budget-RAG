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
    Budget Speech
    Part A
    Tourism
    
    The Tourism sector has the potential...
    """
    parts = []
    
    # 1. Document Name (Formatted for readability)
    if chunk.document:
        doc_title = chunk.document.replace("_", " ").title()
        parts.append(doc_title)
    
    # 2. Heading Path
    if chunk.heading_path:
        for heading in chunk.heading_path:
            # Avoid repeating the document name if it's already in the heading path
            if heading.lower() not in doc_title.lower():
                parts.append(heading)
    
    # 3. Add the actual text content with some separation
    parts.append("\n" + chunk.text.strip())
    
    # Join everything with newlines
    return "\n".join(parts)
