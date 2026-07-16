"""
Context builder for formatting retrieved chunks into LLM prompts.
"""
import logging
from typing import List, Optional
from dataclasses import dataclass

from retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class ContextBuilderConfig:
    """
    Configuration for context building.
    """
    max_context_tokens: int = 4000
    include_citations: bool = True
    include_metadata: bool = True
    chunk_separator: str = "\n\n---\n\n"
    citation_format: str = "[{rank}] {document} ({year}), {section}, p.{page_start}-{page_end}"


class ContextBuilder:
    """
    Builds formatted context from retrieved chunks for LLM prompts.
    
    Responsibilities:
    - Format retrieved chunks with citations
    - Handle heading hierarchy and metadata
    - Manage token budgeting
    - Create structured context for LLM
    """
    
    def __init__(self, config: Optional[ContextBuilderConfig] = None):
        """
        Initialize context builder.
        
        Args:
            config: Context builder configuration
        """
        self.config = config or ContextBuilderConfig()
        logger.info(f"ContextBuilder initialized with max tokens: {self.config.max_context_tokens}")
    
    def build_context(
        self,
        results: List[RetrievalResult],
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Build formatted context from retrieved results.
        
        Args:
            results: List of retrieval results
            max_tokens: Optional override for max tokens
            
        Returns:
            Formatted context string
        """
        if not results:
            logger.warning("No results provided to context builder")
            return ""
        
        max_tokens = max_tokens or self.config.max_context_tokens
        context_parts = []
        current_tokens = 0
        
        for result in results:
            # Format individual chunk
            chunk_text = self._format_chunk(result)
            
            # Estimate tokens (rough approximation: 4 chars per token)
            chunk_tokens = len(chunk_text) // 4
            
            # Check if adding this chunk would exceed budget
            if current_tokens + chunk_tokens > max_tokens:
                logger.info(
                    f"Context token limit reached at {current_tokens} tokens, "
                    f"skipping remaining {len(results) - len(context_parts)} chunks"
                )
                break
            
            context_parts.append(chunk_text)
            current_tokens += chunk_tokens
        
        context = self.config.chunk_separator.join(context_parts)
        
        logger.info(
            f"Built context with {len(context_parts)} chunks, "
            f"estimated {current_tokens} tokens"
        )
        
        return context
    
    def _format_chunk(self, result: RetrievalResult) -> str:
        """
        Format a single retrieval result into a context chunk.
        
        Args:
            result: Retrieval result to format
            
        Returns:
            Formatted chunk string
        """
        parts = []
        
        # Add citation if enabled
        if self.config.include_citations:
            citation = self.config.citation_format.format(
                rank=result.rank,
                document=result.document,
                year=result.year,
                section=result.section,
                subsection=result.subsection if result.subsection else "",
                page_start=result.page_start,
                page_end=result.page_end
            )
            parts.append(f"Source: {citation}")
        
        # Add metadata if enabled
        if self.config.include_metadata:
            metadata_parts = []
            if result.section:
                metadata_parts.append(f"Section: {result.section}")
            if result.subsection:
                metadata_parts.append(f"Subsection: {result.subsection}")
            if result.paragraph_start and result.paragraph_end:
                metadata_parts.append(f"Paragraphs: {result.paragraph_start}-{result.paragraph_end}")
            
            if metadata_parts:
                parts.append(" | ".join(metadata_parts))
        
        # Add the actual text
        parts.append(result.text)
        
        return "\n".join(parts)
    
    def build_context_with_citations(
        self,
        results: List[RetrievalResult],
        max_tokens: Optional[int] = None
    ) -> tuple[str, dict]:
        """
        Build context and return citation mapping.
        
        Args:
            results: List of retrieval results
            max_tokens: Optional override for max tokens
            
        Returns:
            Tuple of (context string, citation mapping)
        """
        if not results:
            return "", {}
        
        max_tokens = max_tokens or self.config.max_context_tokens
        context_parts = []
        citation_map = {}
        current_tokens = 0
        
        for result in results:
            # Create citation key
            citation_key = f"[{result.rank}]"
            
            # Format chunk with citation marker
            chunk_text = self._format_chunk_with_marker(result, citation_key)
            
            # Estimate tokens
            chunk_tokens = len(chunk_text) // 4
            
            # Check token budget
            if current_tokens + chunk_tokens > max_tokens:
                logger.info(
                    f"Context token limit reached at {current_tokens} tokens, "
                    f"skipping remaining {len(results) - len(context_parts)} chunks"
                )
                break
            
            context_parts.append(chunk_text)
            
            # Store citation information
            citation_map[citation_key] = {
                "document": result.document,
                "year": result.year,
                "section": result.section,
                "subsection": result.subsection,
                "page_start": result.page_start,
                "page_end": result.page_end,
                "chunk_id": result.chunk_id,
                "score": result.score
            }
            
            current_tokens += chunk_tokens
        
        context = self.config.chunk_separator.join(context_parts)
        
        logger.info(
            f"Built context with {len(context_parts)} chunks and {len(citation_map)} citations, "
            f"estimated {current_tokens} tokens"
        )
        
        return context, citation_map
    
    def _format_chunk_with_marker(self, result: RetrievalResult, marker: str) -> str:
        """
        Format chunk with citation marker.
        
        Args:
            result: Retrieval result to format
            marker: Citation marker (e.g., "[1]")
            
        Returns:
            Formatted chunk string with marker
        """
        parts = []
        
        # Add marker at start with document information
        citation_info = f"{marker} - {result.document} ({result.year})"
        if result.section:
            citation_info += f", {result.section}"
        parts.append(citation_info)
        
        # Add additional metadata
        if self.config.include_metadata:
            metadata_parts = []
            if result.subsection:
                metadata_parts.append(f"Subsection: {result.subsection}")
            if result.page_start and result.page_end:
                metadata_parts.append(f"Pages: {result.page_start}-{result.page_end}")
            
            if metadata_parts:
                parts.append(" | ".join(metadata_parts))
        
        # Add text
        parts.append(result.text)
        
        return "\n".join(parts)
