import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chunking.metadata import MetadataCleaner
from chunking.utils import PageMapper, ChunkIdGenerator
from chunking.table_chunker import TableChunker

def test_metadata_cleaner():
    assert MetadataCleaner.clean_markdown("**Introduction**") == "Introduction"
    assert MetadataCleaner.clean_markdown("# **Part A**") == "Part A"
    assert MetadataCleaner.clean_markdown("<u>Part A</u>") == "Part A"
    
def test_normalize_section_name():
    assert MetadataCleaner.normalize_section_name(["# **CONTENTS**"]) == "Table of Contents"
    assert MetadataCleaner.normalize_section_name(["Annexure to Part-A"]) == "Annexure"
    assert MetadataCleaner.normalize_section_name(["Introduction"]) == "Unknown"

def test_page_mapper():
    md = """
Some text on page 1
2
Some text on page 2
**1.** A paragraph on page 2.
3
Text on page 3.
"""
    mapper = PageMapper(md.strip())
    # Line 0: "Some text on page 1" -> page 1
    # Line 1: "2" -> page 2
    # Line 2: "Some text on page 2" -> page 2
    # Line 4: "3" -> page 3
    assert mapper.get_page_range(0, 0) == (1, 1)
    assert mapper.get_page_range(2, 3) == (2, 2)
    assert mapper.get_page_range(2, 5) == (2, 3)

def test_chunk_id_generator():
    gen = ChunkIdGenerator("budget", 2026)
    assert gen.generate(["Part A"]) == "budget_2026_part_a_001"
    assert gen.generate(["Part A"]) == "budget_2026_part_a_002"
    assert gen.generate(["PART B", "Indirect Taxes"]) == "budget_2026_part_b_indirect_taxes_001"

def test_table_splitting():
    gen = ChunkIdGenerator("budget", 2026)
    mapper = PageMapper("text\n2")
    chunker = TableChunker(gen, mapper)
    
    # We will patch max chunk size to force split for test
    from chunking import config
    original_max = config.config.MAX_CHUNK_SIZE
    config.config.MAX_CHUNK_SIZE = 10 # very small, forcing split
    
    table_content = "| Col 1 | Col 2 |\n|---|---|\n| a | b |\n| c | d |"
    tables = [{"content": table_content, "start_line": 0, "end_line": 3, "heading_hierarchy": ["Table Test"]}]
    
    chunks = chunker.chunk_tables(tables)
    
    assert len(chunks) == 2
    assert "Col 1" in chunks[0].text
    assert "a | b" in chunks[0].text
    assert "Col 1" in chunks[1].text
    assert "c | d" in chunks[1].text
    
    config.config.MAX_CHUNK_SIZE = original_max
