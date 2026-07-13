import re
from typing import List

class MetadataCleaner:
    @staticmethod
    def clean_markdown(text: str) -> str:
        """Remove markdown formatting from text."""
        if not text:
            return ""
        # Remove markdown heading markers
        text = re.sub(r'^#+\s*', '', text)
        # Remove bold/italic markers
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()

    @staticmethod
    def normalize_section_name(headings: List[str]) -> str:
        """Analyze headings to determine the semantic section name."""
        for heading in headings:
            cleaned = MetadataCleaner.clean_markdown(heading).lower()
            if "contents" in cleaned or "index" in cleaned:
                return "Table of Contents"
            if "annexure" in cleaned:
                return "Annexure"
            if "appendix" in cleaned:
                return "Appendix"
        return "Unknown"
