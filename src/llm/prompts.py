"""
Modular prompt templates for RAG operations.
"""
from typing import Dict, Any, Optional


class PromptTemplate:
    """
    Base class for prompt templates.
    
    Provides versioning and modularity for different prompt types.
    """
    
    def __init__(self, version: str = "1.0"):
        """
        Initialize prompt template.
        
        Args:
            version: Template version for tracking
        """
        self.version = version
    
    def format(self, **kwargs) -> str:
        """
        Format the template with provided variables.
        
        Args:
            **kwargs: Variables to substitute in template
            
        Returns:
            Formatted prompt string
        """
        raise NotImplementedError("Subclasses must implement format method")


class QAPromptTemplate(PromptTemplate):
    """
    Question-answering prompt template for RAG.
    """
    
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context from Indian Budget documents. Your role is to:

1. Answer the question using only the information provided in the context
2. Be accurate and cite your sources using the descriptive document information provided in each context chunk
3. If the context doesn't contain enough information to answer the question, state that clearly
4. Provide specific details like amounts, percentages, and dates when available
5. Maintain a professional and informative tone

Each context chunk begins with descriptive source information like "[1] - Union Budget 2024-25 (2024), Ministry of Agriculture". When citing sources in your answer, use this descriptive information directly. Do NOT use the numeric markers like [1], [2] in your answer. Instead, cite the full descriptive information.

For example, instead of saying "According to [7]", say "According to the Union Budget 2024-25, Ministry of Agriculture" or "According to Budget 2026-27, Revenue Section"."""
    
    USER_PROMPT = """Context:
{context}

Question: {question}

Answer:"""
    
    def format(self, question: str, context: str) -> str:
        """
        Format QA prompt with question and context.
        
        Args:
            question: User question
            context: Retrieved context chunks
            
        Returns:
            Formatted prompt string
        """
        return f"{self.SYSTEM_PROMPT}\n\n{self.USER_PROMPT.format(question=question, context=context)}"


class SummarizationPromptTemplate(PromptTemplate):
    """
    Summarization prompt template for RAG.
    """
    
    SYSTEM_PROMPT = """You are a helpful assistant that summarizes content from Indian Budget documents. Your role is to:

1. Create a clear and concise summary of the provided context
2. Highlight key points, figures, and policy decisions
3. Maintain accuracy and avoid adding information not present in the context
4. Use professional language appropriate for budget analysis
5. Structure the summary logically with bullet points when helpful"""
    
    USER_PROMPT = """Context:
{context}

Task: Summarize the key points from the above context.

Summary:"""
    
    def format(self, context: str) -> str:
        """
        Format summarization prompt with context.
        
        Args:
            context: Retrieved context chunks
            
        Returns:
            Formatted prompt string
        """
        return f"{self.SYSTEM_PROMPT}\n\n{self.USER_PROMPT.format(context=context)}"


class ComparisonPromptTemplate(PromptTemplate):
    """
    Comparison prompt template for comparing budget items across years or sections.
    """
    
    SYSTEM_PROMPT = """You are a helpful assistant that compares information from Indian Budget documents. Your role is to:

1. Compare the provided contexts to identify similarities and differences
2. Highlight changes in allocations, policies, or priorities
3. Provide specific figures and percentages when available
4. Present the comparison in a clear, structured format
5. Use citation markers to indicate sources"""
    
    USER_PROMPT = """Context A:
{context_a}

Context B:
{context_b}

Task: Compare the information from the two contexts above.

Comparison:"""
    
    def format(self, context_a: str, context_b: str) -> str:
        """
        Format comparison prompt with two contexts.
        
        Args:
            context_a: First context for comparison
            context_b: Second context for comparison
            
        Returns:
            Formatted prompt string
        """
        return f"{self.SYSTEM_PROMPT}\n\n{self.USER_PROMPT.format(context_a=context_a, context_b=context_b)}"


class AnalysisPromptTemplate(PromptTemplate):
    """
    Analysis prompt template for deep analysis of budget content.
    """
    
    SYSTEM_PROMPT = """You are a budget analysis expert that provides detailed analysis of Indian Budget documents. Your role is to:

1. Analyze the provided context in depth
2. Identify trends, patterns, and implications
3. Provide insights on budget allocations and policy directions
4. Consider economic and fiscal implications
5. Support your analysis with specific data from the context
6. Use citation markers to reference sources"""
    
    USER_PROMPT = """Context:
{context}

Question: {question}

Analysis:"""
    
    def format(self, question: str, context: str) -> str:
        """
        Format analysis prompt with question and context.
        
        Args:
            question: Analysis question
            context: Retrieved context chunks
            
        Returns:
            Formatted prompt string
        """
        return f"{self.SYSTEM_PROMPT}\n\n{self.USER_PROMPT.format(question=question, context=context)}"


class PromptTemplateRegistry:
    """
    Registry for managing prompt templates.
    
    Provides centralized access to all prompt templates with versioning.
    """
    
    _templates: Dict[str, PromptTemplate] = {
        "qa": QAPromptTemplate(),
        "summarization": SummarizationPromptTemplate(),
        "comparison": ComparisonPromptTemplate(),
        "analysis": AnalysisPromptTemplate(),
    }
    
    @classmethod
    def get_template(cls, template_name: str, version: Optional[str] = None) -> PromptTemplate:
        """
        Get a prompt template by name.
        
        Args:
            template_name: Name of the template (qa, summarization, comparison, analysis)
            version: Optional version override
            
        Returns:
            PromptTemplate instance
        """
        template = cls._templates.get(template_name)
        
        if template is None:
            available = ", ".join(cls._templates.keys())
            raise ValueError(
                f"Unknown template: {template_name}. "
                f"Available templates: {available}"
            )
        
        if version:
            # Return a new instance with specified version
            return template.__class__(version=version)
        
        return template
    
    @classmethod
    def register_template(cls, name: str, template: PromptTemplate) -> None:
        """
        Register a new prompt template.
        
        Args:
            name: Name for the template
            template: PromptTemplate instance
        """
        cls._templates[name] = template
    
    @classmethod
    def list_templates(cls) -> Dict[str, str]:
        """
        List all available templates with their versions.
        
        Returns:
            Dictionary of template names to versions
        """
        return {
            name: template.version 
            for name, template in cls._templates.items()
        }
