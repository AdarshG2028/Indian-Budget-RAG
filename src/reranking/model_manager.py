"""
Model manager for reranking models.
Handles downloading, caching, loading, and device management.
"""
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages reranker model lifecycle.
    
    Responsibilities:
    - Download models from HuggingFace
    - Cache models locally
    - Load models efficiently
    - Manage device selection (CPU/GPU)
    - Handle model versioning
    - Memory-efficient loading
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Initialize model manager.
        
        Args:
            cache_dir: Directory for model caching
            device: Device for model inference
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "reranking_models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self._loaded_models: Dict[str, Any] = {}
        
        logger.info(f"ModelManager initialized with cache: {self.cache_dir}, device: {self.device}")
    
    def get_model(
        self,
        model_name: str,
        model_type: str = "cross_encoder"
    ) -> Any:
        """
        Get or load a model.
        
        Args:
            model_name: Name of the model
            model_type: Type of model (cross_encoder, etc.)
            
        Returns:
            Loaded model instance
        """
        cache_key = f"{model_type}:{model_name}"
        
        if cache_key in self._loaded_models:
            logger.debug(f"Using cached model: {model_name}")
            return self._loaded_models[cache_key]
        
        logger.info(f"Loading model: {model_name}")
        model = self._load_model(model_name, model_type)
        self._loaded_models[cache_key] = model
        
        return model
    
    def _load_model(self, model_name: str, model_type: str) -> Any:
        """
        Load a model based on type.
        
        Args:
            model_name: Name of the model
            model_type: Type of model
            
        Returns:
            Loaded model instance
        """
        if model_type == "cross_encoder":
            return self._load_cross_encoder(model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _load_cross_encoder(self, model_name: str) -> Any:
        """
        Load a cross-encoder model.
        
        Args:
            model_name: Name of the cross-encoder model
            
        Returns:
            Loaded cross-encoder model
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for cross-encoder models. "
                "Install it with: uv add sentence-transformers"
            )
        
        # Determine device
        device = self.device if self.device != "cpu" else None
        
        # Load model
        model = CrossEncoder(
            model_name,
            device=device,
            cache_folder=str(self.cache_dir)
        )
        
        logger.info(f"Successfully loaded cross-encoder: {model_name}")
        return model
    
    def unload_model(self, model_name: str, model_type: str = "cross_encoder") -> None:
        """
        Unload a model from memory.
        
        Args:
            model_name: Name of the model
            model_type: Type of model
        """
        cache_key = f"{model_type}:{model_name}"
        
        if cache_key in self._loaded_models:
            del self._loaded_models[cache_key]
            logger.info(f"Unloaded model: {model_name}")
    
    def unload_all(self) -> None:
        """Unload all models from memory."""
        self._loaded_models.clear()
        logger.info("Unloaded all models")
    
    def get_cache_size(self) -> int:
        """
        Get size of model cache directory in bytes.
        
        Returns:
            Cache size in bytes
        """
        total_size = 0
        
        for item in self.cache_dir.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
        
        return total_size
    
    def clear_cache(self) -> None:
        """Clear the model cache directory."""
        import shutil
        
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cleared model cache: {self.cache_dir}")
    
    def list_cached_models(self) -> list[str]:
        """
        List all cached models.
        
        Returns:
            List of model names
        """
        models = []
        
        if self.cache_dir.exists():
            for item in self.cache_dir.iterdir():
                if item.is_dir():
                    models.append(item.name)
        
        return models


# Global model manager instance
_global_model_manager: Optional[ModelManager] = None


def get_global_model_manager() -> ModelManager:
    """
    Get the global model manager instance.
    
    Returns:
        Global ModelManager instance
    """
    global _global_model_manager
    
    if _global_model_manager is None:
        _global_model_manager = ModelManager()
    
    return _global_model_manager


def set_global_model_manager(manager: ModelManager) -> None:
    """
    Set the global model manager instance.
    
    Args:
        manager: ModelManager instance to set as global
    """
    global _global_model_manager
    _global_model_manager = manager
