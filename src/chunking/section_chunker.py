from typing import Dict, List, Any
import tiktoken
from .models import Chunk
from .config import config
from .metadata import MetadataCleaner
from .keyword_extractor import KeywordExtractor
from .entity_extractor import EntityExtractor
from .utils import ChunkIdGenerator, PageMapper

class SectionChunker:
    def __init__(self, id_generator: ChunkIdGenerator, page_mapper: PageMapper):
        self.tokenizer = tiktoken.get_encoding(config.ENCODING_NAME)
        self.id_generator = id_generator
        self.page_mapper = page_mapper
        self.document = id_generator.document
        
    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
        
    def chunk_section(
        self, 
        paragraphs: List[Dict], 
        heading_path: List[str]
    ) -> List[Chunk]:
        """Chunk a section using sliding window approach over paragraphs."""
        chunks = []
        if not paragraphs:
            return chunks
            
        start_idx = 0
        
        while start_idx < len(paragraphs):
            chunk_paras = []
            current_tokens = 0
            
            for i in range(start_idx, len(paragraphs)):
                para = paragraphs[i]
                para_text = f"**{para['number']}.** {para['content']}"
                para_tokens = self._count_tokens(para_text)
                
                if chunk_paras and current_tokens + para_tokens > config.MAX_CHUNK_SIZE:
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
                
                if current_tokens + para_tokens <= config.MAX_CHUNK_SIZE:
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
            
            # Clean heading path for metadata
            clean_heading_path = [MetadataCleaner.clean_markdown(h) for h in heading_path]
            semantic_section = MetadataCleaner.normalize_section_name(clean_heading_path)
            
            # Extract section and subsection
            section = clean_heading_path[0] if len(clean_heading_path) > 0 else semantic_section
            subsection = clean_heading_path[1] if len(clean_heading_path) > 1 else ""
            
            # Use semantic section name for base grouping if heading_path is empty
            effective_path = clean_heading_path if clean_heading_path else [semantic_section]
            chunk_id = self.id_generator.generate(effective_path)
            
            # Determine page ranges
            start_line = chunk_paras[0]['line_number']
            end_line = chunk_paras[-1]['line_number']
            page_start, page_end = self.page_mapper.get_page_range(start_line, end_line)
            
            chunk = Chunk(
                chunk_id=chunk_id,
                document=self.document,
                year=2026,
                section=section,
                subsection=subsection,
                paragraph_start=chunk_paras[0]['number'],
                paragraph_end=chunk_paras[-1]['number'],
                page_start=page_start,
                page_end=page_end,
                token_count=current_tokens,
                char_count=len(content),
                text=content,
                keywords=KeywordExtractor.extract_keywords(content),
                entities=EntityExtractor.extract_entities(content)
            )
            
            chunks.append(chunk)
            
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
