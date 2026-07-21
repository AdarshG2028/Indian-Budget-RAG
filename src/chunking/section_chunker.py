from typing import Callable, Dict, List, Any
from .models import Chunk
from .config import config
from .metadata import MetadataCleaner
from .keyword_extractor import KeywordExtractor
from .entity_extractor import EntityExtractor
from .utils import ChunkIdGenerator, PageMapper
from .tokenizer import get_token_counter, embedding_budget, split_to_budget, cap_heading_path

class SectionChunker:
    def __init__(
        self,
        id_generator: ChunkIdGenerator,
        page_mapper: PageMapper,
        header_tokens_for: Callable[[List[str]], int] = None
    ):
        """
        Args:
            header_tokens_for: Given a heading path, returns the number of
                tokens the embedder prepends to chunks under it (see
                embeddings.utils.format_text_for_embedding). The header
                includes the heading path, so its cost varies per section and
                must be subtracted from the 512-token window per section
                rather than once per document.
        """
        self._count = get_token_counter()
        self.id_generator = id_generator
        self.page_mapper = page_mapper
        self.document = id_generator.document
        self._header_tokens_for = header_tokens_for or (lambda _: 0)
        # Budget for sections with no heading path; recomputed per section.
        self.max_tokens = self._budget_for([])

    def _budget_for(self, heading_path: List[str]) -> int:
        return min(
            config.MAX_CHUNK_SIZE,
            embedding_budget(self._header_tokens_for(heading_path))
        )

    def _count_tokens(self, text: str) -> int:
        return self._count(text)

    def _split_to_budget(self, text: str) -> List[str]:
        """
        Split `text` into parts that each fit self.max_tokens.

        A safety net for cases the paragraph-level loop can't size down: a
        single paragraph longer than the budget, or the minimum-size rule
        overshooting the maximum. Splits on whitespace, so it can cut
        mid-sentence — that only happens for text that would otherwise have
        been truncated outright, which is strictly worse.
        """
        return split_to_budget(text, self._count_tokens, self.max_tokens)

    def chunk_section(
        self, 
        paragraphs: List[Dict], 
        heading_path: List[str]
    ) -> List[Chunk]:
        """Chunk a section using sliding window approach over paragraphs."""
        chunks = []
        if not paragraphs:
            return chunks

        # Resolve the heading path once: it determines both the embedded header
        # (and therefore the text budget) and the metadata on every chunk here.
        clean_heading_path = [MetadataCleaner.clean_markdown(h) for h in heading_path]
        semantic_section = MetadataCleaner.normalize_section_name(clean_heading_path)
        section = clean_heading_path[0] if len(clean_heading_path) > 0 else semantic_section
        subsection = clean_heading_path[1] if len(clean_heading_path) > 1 else ""
        effective_path = cap_heading_path(
            clean_heading_path if clean_heading_path else [semantic_section],
            self._count_tokens,
            config.MAX_HEADING_TOKENS,
        )
        self.max_tokens = self._budget_for(effective_path)

        start_idx = 0
        
        while start_idx < len(paragraphs):
            chunk_paras = []
            current_tokens = 0
            
            for i in range(start_idx, len(paragraphs)):
                para = paragraphs[i]
                para_text = f"**{para['number']}.** {para['content']}"
                para_tokens = self._count_tokens(para_text)
                
                if chunk_paras and current_tokens + para_tokens > self.max_tokens:
                    break
                    
                chunk_paras.append(para)
                current_tokens += para_tokens
                
                if len(chunk_paras) >= 2 and current_tokens >= config.CHUNK_SIZE:
                    break
            
            # Ensure minimum chunk size by adding more paragraphs if needed
            while (current_tokens < config.MIN_CHUNK_SIZE and 
                   start_idx + len(chunk_paras) < len(paragraphs)):
                next_para = paragraphs[start_idx + len(chunk_paras)]
                para_text = f"**{next_para['number']}.** {next_para['content']}"
                para_tokens = self._count_tokens(para_text)
                
                if current_tokens + para_tokens <= self.max_tokens:
                    chunk_paras.append(next_para)
                    current_tokens += para_tokens
                else:
                    # If adding next paragraph would exceed max, break the loop
                    # But if current chunk is still too small, we need to handle it
                    if current_tokens < config.MIN_CHUNK_SIZE:
                        # Allow exceeding max to meet minimum
                        chunk_paras.append(next_para)
                        current_tokens += para_tokens
                    else:
                        break
                    
            if not chunk_paras:
                chunk_paras = [paragraphs[start_idx]]
                current_tokens = self._count_tokens(f"**{chunk_paras[0]['number']}.** {chunk_paras[0]['content']}")
                
            content_parts = [f"**{p['number']}.** {p['content']}" for p in chunk_paras]
            content = '\n\n'.join(content_parts)
            
            chunk_id = self.id_generator.generate(effective_path)
            
            # Determine page ranges
            start_line = chunk_paras[0]['line_number']
            end_line = chunk_paras[-1]['line_number']
            page_start, page_end = self.page_mapper.get_page_range(start_line, end_line)
            
            # Emit one chunk per budget-sized part. Normally a single part;
            # more only when the paragraph loop couldn't size the content down
            # (oversized single paragraph, or the minimum-size rule overshooting).
            parts = self._split_to_budget(content)
            for part_idx, part in enumerate(parts):
                part_id = chunk_id if len(parts) == 1 else f"{chunk_id}_part{part_idx + 1}"
                chunks.append(Chunk(
                    chunk_id=part_id,
                    document=self.document,
                    year=2026,
                    section=section,
                    subsection=subsection,
                    paragraph_start=chunk_paras[0]['number'],
                    paragraph_end=chunk_paras[-1]['number'],
                    page_start=page_start,
                    page_end=page_end,
                    token_count=self._count_tokens(part),
                    char_count=len(part),
                    text=part,
                    heading_path=effective_path,
                    keywords=KeywordExtractor.extract_keywords(part),
                    entities=EntityExtractor.extract_entities(part)
                ))
            
            # Advance sliding window
            if start_idx + len(chunk_paras) >= len(paragraphs):
                break
                
            # Overlap in paragraphs is determined by calculating how many paras we can drop while keeping forward progress
            # We will overlap by 1 paragraph normally, or less if chunk is tiny
            overlap = 1
            new_start_idx = start_idx + len(chunk_paras) - overlap
            if new_start_idx <= start_idx:
                new_start_idx = start_idx + 1
            start_idx = new_start_idx
            
        return chunks
