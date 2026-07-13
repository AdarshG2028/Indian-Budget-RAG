import re
from typing import Dict

class ChunkIdGenerator:
    """Generates deterministic chunk IDs."""
    
    def __init__(self, document: str, year: int):
        self.document = document
        self.year = year
        self.counters: Dict[str, int] = {}
    
    def generate(self, heading_path: list[str]) -> str:
        """Generate a deterministic chunk ID."""
        key = "_".join(heading_path) if heading_path else "unknown"
        clean_key = self._clean_for_id(key)
        
        if clean_key not in self.counters:
            self.counters[clean_key] = 1
        else:
            self.counters[clean_key] += 1
            
        return f"{self.document}_{self.year}_{clean_key}_{self.counters[clean_key]:03d}"
    
    @staticmethod
    def _clean_for_id(text: str) -> str:
        """Clean text for use in ID."""
        text = re.sub(r'[^a-zA-Z0-9]', '_', text)
        text = re.sub(r'_+', '_', text)
        return text.strip('_').lower()


class PageMapper:
    """Maps line numbers to page numbers in a markdown document."""
    
    def __init__(self, markdown_text: str):
        self.line_to_page = {}
        self._build_mapping(markdown_text)
        
    def _build_mapping(self, text: str):
        lines = text.split('\n')
        current_page = 1
        
        # Some Markdown outputs just have lines with isolated page numbers
        # e.g., "2", "3", or "Page 2"
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            # Try to match standalone digits as page numbers, but avoid small numbers that might be paragraph numbers
            # If the parser writes just a single number on a line, it's typically a page number.
            if re.match(r'^\d+$', line_stripped):
                try:
                    p = int(line_stripped)
                    # rudimentary check to ensure it's a plausible page number increment
                    if p == current_page + 1:
                        current_page = p
                except ValueError:
                    pass
            
            self.line_to_page[i] = current_page
            
    def get_page_range(self, start_line: int, end_line: int) -> tuple[int, int]:
        """Get the start and end page for a given range of line numbers."""
        start_page = self.line_to_page.get(start_line, 1)
        end_page = self.line_to_page.get(end_line, start_page)
        return start_page, end_page
