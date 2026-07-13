from typing import Dict, List
import tiktoken
from .models import Chunk
from .config import config
from .metadata import MetadataCleaner
from .keyword_extractor import KeywordExtractor
from .entity_extractor import EntityExtractor
from .utils import ChunkIdGenerator, PageMapper

class TableChunker:
    def __init__(self, id_generator: ChunkIdGenerator, page_mapper: PageMapper):
        self.tokenizer = tiktoken.get_encoding(config.ENCODING_NAME)
        self.id_generator = id_generator
        self.page_mapper = page_mapper
        
    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
        
    def chunk_tables(
        self, 
        tables: List[Dict]
    ) -> List[Chunk]:
        """Create chunks from tables, splitting them safely by rows if they exceed token budget."""
        chunks = []
        
        for i, table in enumerate(tables):
            content = table['content']
            heading_path = table.get('heading_hierarchy', [])
            
            # Clean heading path
            clean_heading_path = [MetadataCleaner.clean_markdown(h) for h in heading_path]
            semantic_section = MetadataCleaner.normalize_section_name(clean_heading_path)
            
            # Extract section and subsection
            section = clean_heading_path[0] if len(clean_heading_path) > 0 else semantic_section
            subsection = clean_heading_path[1] if len(clean_heading_path) > 1 else ""
            
            # Use semantic section name if empty
            effective_path = clean_heading_path if clean_heading_path else [semantic_section]
            base_chunk_id = self.id_generator.generate(effective_path + ["table"])
            
            # Determine page ranges
            start_line = table['start_line']
            end_line = table['end_line']
            page_start, page_end = self.page_mapper.get_page_range(start_line, end_line)
            
            # Check if table fits in one chunk
            total_tokens = self._count_tokens(content)
            
            if total_tokens <= config.MAX_CHUNK_SIZE:
                chunk = self._create_table_chunk(
                    content, section, subsection, base_chunk_id, 
                    page_start, page_end, total_tokens
                )
                chunks.append(chunk)
            else:
                # Split table row by row, keeping header
                lines = content.strip().split('\n')
                if len(lines) < 3:
                    # Can't meaningfully split, just keep it together
                    chunk = self._create_table_chunk(
                        content, section, subsection, base_chunk_id, 
                        page_start, page_end, total_tokens
                    )
                    chunks.append(chunk)
                    continue
                    
                # Assume first two rows are header and separator
                header = '\n'.join(lines[:2]) + '\n'
                current_part = header
                current_tokens = self._count_tokens(header)
                part_idx = 1
                
                # Precalculate how many parts for the metadata
                # Since we don't know yet, we'll assign section_chunk_index iteratively
                
                for row in lines[2:]:
                    row_tokens = self._count_tokens(row + '\n')
                    
                    if current_tokens + row_tokens > config.MAX_CHUNK_SIZE and current_part != header:
                        # Yield current part
                        chunk = self._create_table_chunk(
                            current_part.strip(), section, subsection, f"{base_chunk_id}_part{part_idx}",
                            page_start, page_end, current_tokens
                        )
                        chunks.append(chunk)
                        
                        # Start new part with header
                        current_part = header + row + '\n'
                        current_tokens = self._count_tokens(header) + row_tokens
                        part_idx += 1
                    else:
                        current_part += row + '\n'
                        current_tokens += row_tokens
                
                # Add final part
                if current_part != header:
                    chunk = self._create_table_chunk(
                        current_part.strip(), section, subsection, f"{base_chunk_id}_part{part_idx}",
                        page_start, page_end, current_tokens
                    )
                    chunks.append(chunk)
                    
        return chunks

    def _create_table_chunk(
        self, content: str, section: str, subsection: str, chunk_id: str, 
        page_start: int, page_end: int, token_count: int
    ) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            document="budget_speech",
            year=2026,
            section=section,
            subsection=subsection,
            paragraph_start=0,
            paragraph_end=0,
            page_start=page_start,
            page_end=page_end,
            token_count=token_count,
            char_count=len(content),
            text=content,
            keywords=KeywordExtractor.extract_keywords(content),
            entities=EntityExtractor.extract_entities(content)
        )
