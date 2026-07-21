from typing import Callable, Dict, List
from .models import Chunk
from .config import config
from .metadata import MetadataCleaner
from .keyword_extractor import KeywordExtractor
from .entity_extractor import EntityExtractor
from .utils import ChunkIdGenerator, PageMapper
from .tokenizer import get_token_counter, embedding_budget, split_to_budget, cap_heading_path

class TableChunker:
    def __init__(
        self,
        id_generator: ChunkIdGenerator,
        page_mapper: PageMapper,
        header_tokens_for: Callable[[List[str]], int] = None
    ):
        """
        Args:
            header_tokens_for: Given a heading path, returns the tokens the
                embedder prepends to chunks under it; subtracted from the
                512-token window. See SectionChunker for details.
        """
        self._count = get_token_counter()
        self.id_generator = id_generator
        self.page_mapper = page_mapper
        self.document = id_generator.document
        self._header_tokens_for = header_tokens_for or (lambda _: 0)
        self.max_tokens = self._budget_for([])

    def _budget_for(self, heading_path: List[str]) -> int:
        return min(
            config.MAX_CHUNK_SIZE,
            embedding_budget(self._header_tokens_for(heading_path))
        )

    def _count_tokens(self, text: str) -> int:
        return self._count(text)

    def _split_row_to_budget(self, row: str, budget: int = None) -> List[str]:
        """
        Split a single table row wider than `budget` (default: the full chunk
        budget).

        Rare, but a wide row is otherwise unsplittable by the row loop and
        would be truncated by the embedder instead.
        """
        return split_to_budget(
            row, self._count_tokens, self.max_tokens if budget is None else budget
        )

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
            effective_path = cap_heading_path(
                effective_path, self._count_tokens, config.MAX_HEADING_TOKENS
            )
            base_chunk_id = self.id_generator.generate(effective_path + ["table"])

            # The embedded header carries the heading path, so the text budget
            # depends on it.
            self.max_tokens = self._budget_for(effective_path)
            
            # Determine page ranges
            start_line = table['start_line']
            end_line = table['end_line']
            page_start, page_end = self.page_mapper.get_page_range(start_line, end_line)
            
            # Check if table fits in one chunk
            total_tokens = self._count_tokens(content)
            
            if total_tokens <= self.max_tokens:
                chunk = self._create_table_chunk(
                    content, section, subsection, base_chunk_id,
                    page_start, page_end, total_tokens, effective_path
                )
                chunks.append(chunk)
            else:
                parts = self._split_table(content)
                for i, part in enumerate(parts):
                    chunks.append(self._create_table_chunk(
                        part, section, subsection,
                        base_chunk_id if len(parts) == 1 else f"{base_chunk_id}_part{i + 1}",
                        page_start, page_end, self._count_tokens(part), effective_path
                    ))

        return chunks

    def _split_table(self, content: str) -> List[str]:
        """
        Split an over-budget table into parts that each fit self.max_tokens.

        Repeats the header row on every part so each chunk stays self-describing
        — but only when the header is small enough to be worth its cost. Budget
        documents contain tables whose header row alone runs to hundreds of
        tokens; repeating those would leave no room for data (and previously
        produced parts that overflowed the model's window regardless).
        """
        lines = [ln for ln in content.strip().split('\n') if ln.strip()]

        header = ''
        rows = lines
        if len(lines) >= 3:
            candidate = '\n'.join(lines[:2])
            # Only repeat the header if it leaves at least half the budget for data.
            if self._count_tokens(candidate) <= self.max_tokens // 2:
                header = candidate
                rows = lines[2:]

        header_tokens = self._count_tokens(header + '\n') if header else 0
        parts: List[str] = []
        current: List[str] = []
        current_tokens = header_tokens

        for raw_row in rows:
            # A row wider than the remaining budget can't fit even alone.
            for row in self._split_row_to_budget(raw_row, self.max_tokens - header_tokens):
                row_tokens = self._count_tokens(row + '\n')
                if current and current_tokens + row_tokens > self.max_tokens:
                    parts.append('\n'.join(([header] if header else []) + current))
                    current = [row]
                    current_tokens = header_tokens + row_tokens
                else:
                    current.append(row)
                    current_tokens += row_tokens

        if current:
            parts.append('\n'.join(([header] if header else []) + current))
        return parts or [content]

    def _create_table_chunk(
        self, content: str, section: str, subsection: str, chunk_id: str,
        page_start: int, page_end: int, token_count: int,
        heading_path: List[str] = None
    ) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            document=self.document,
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
            heading_path=heading_path or [],
            keywords=KeywordExtractor.extract_keywords(content),
            entities=EntityExtractor.extract_entities(content)
        )
