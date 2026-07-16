"""
Base interface for generation/answer evaluation metrics.
This module provides architecture for future answer evaluation without implementation.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseGenerationMetric(ABC):
    """
    Abstract base class for generation/answer evaluation metrics.
    
    Future implementations may include:
    - Faithfulness (answer groundedness)
    - Answer Relevance
    - Context Precision
    - Context Recall
    - Answer Correctness
    """
    
    @abstractmethod
    def calculate(
        self,
        query: str,
        answer: str,
        context: str,
        ground_truth: Optional[str] = None
    ) -> float:
        """
        Calculate the generation metric.
        
        Args:
            query: The user query
            answer: The generated answer
            context: The retrieved context used for generation
            ground_truth: Optional ground truth answer
            
        Returns:
            Metric value as float
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this metric.
        
        Returns:
            Metric name string
        """
        pass
