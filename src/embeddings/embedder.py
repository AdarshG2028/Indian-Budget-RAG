from abc import ABC, abstractmethod
from typing import List
import logging

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .config import MODEL_NAME, NORMALIZE_EMBEDDINGS, DEVICE

logger = logging.getLogger(__name__)

class BaseEmbedder(ABC):
    """Abstract interface for embedding generation."""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of documents for indexing."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query for search/retrieval."""
        pass

    @abstractmethod
    def embedding_dimension(self) -> int:
        """Returns the dimension of the embedding space."""
        pass

class BGEEmbedder(BaseEmbedder):
    """Concrete implementation using BAAI/bge models via sentence-transformers."""
    
    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed. Please install it to use BGEEmbedder.")
            
        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = NORMALIZE_EMBEDDINGS
        
        # We can dynamically figure out the dimension by passing a dummy string
        self._dim = len(self.model.encode("dummy")[0] if isinstance(self.model.encode("dummy"), list) else self.model.encode("dummy"))
        logger.info(f"Model loaded successfully. Embedding dimension: {self._dim}")
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a batch of texts. BGE models often recommend adding specific instructions 
        to queries, but for *documents*, plain text is standard.
        """
        embeddings = self.model.encode(
            texts, 
            normalize_embeddings=self.normalize, 
            show_progress_bar=False,  # We manage our own progress bar at the pipeline level
            batch_size=len(texts) # The batch is already sized by our pipeline
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query. BGE models sometimes use a prefix for queries.
        For bge-base-en-v1.5, the recommended prefix for queries is:
        'Represent this sentence for searching relevant passages: '
        """
        query_prefix = "Represent this sentence for searching relevant passages: "
        prefixed_text = query_prefix + text
        
        embedding = self.model.encode(
            prefixed_text,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        return embedding.tolist()

    def embedding_dimension(self) -> int:
        return self._dim
