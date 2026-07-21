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

@pytest.fixture
def patch_max_chunk_size():
    """Temporarily shrink the chunk budget to force splitting in tests."""
    from chunking import config
    original = config.config.MAX_CHUNK_SIZE

    def _apply(value):
        config.config.MAX_CHUNK_SIZE = value
        return value

    yield _apply
    config.config.MAX_CHUNK_SIZE = original


def _make_chunker():
    # Built after the budget is patched: max_tokens is resolved in __init__.
    return TableChunker(ChunkIdGenerator("budget", 2026), PageMapper("text\n2"))


def test_table_splitting_repeats_header(patch_max_chunk_size):
    """A split table keeps its header on every part, so each chunk stands alone."""
    patch_max_chunk_size(40)
    chunker = _make_chunker()

    rows = "\n".join(f"| r{i} | v{i} |" for i in range(6))
    table_content = f"| Col 1 | Col 2 |\n|---|---|\n{rows}"
    tables = [{"content": table_content, "start_line": 0, "end_line": 7,
               "heading_hierarchy": ["Table Test"]}]

    chunks = chunker.chunk_tables(tables)

    assert len(chunks) > 1
    for chunk in chunks:
        assert "Col 1" in chunk.text
        assert chunker._count_tokens(chunk.text) <= chunker.max_tokens
    # No data row is dropped by the split
    combined = " ".join(c.text for c in chunks)
    for i in range(6):
        assert f"r{i}" in combined


def test_table_splitting_drops_oversized_header(patch_max_chunk_size):
    """
    When the header alone would consume most of the budget, it is not repeated.

    Budget documents contain tables whose header row runs to hundreds of
    tokens; repeating those leaves no room for data and produces parts that
    overflow the embedding window regardless.
    """
    patch_max_chunk_size(30)
    chunker = _make_chunker()

    wide_header = "| " + " | ".join(f"LongColumnName{i}" for i in range(10)) + " |"
    table_content = f"{wide_header}\n|---|---|\n| a | b |\n| c | d |"
    tables = [{"content": table_content, "start_line": 0, "end_line": 3,
               "heading_hierarchy": ["Table Test"]}]

    chunks = chunker.chunk_tables(tables)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunker._count_tokens(chunk.text) <= chunker.max_tokens


def test_chunks_never_exceed_embedding_window(patch_max_chunk_size):
    """
    The invariant the chunker exists to guarantee: nothing the embedder sees
    is over its window, even for content with no natural split points.
    """
    patch_max_chunk_size(50)
    chunker = _make_chunker()

    # A single "word" of <br>-joined figures: no spaces, no pipes to split on.
    unsplittable = "| " + "<br>".join(str(i * 1.5) for i in range(200)) + " |"
    tables = [{"content": f"| A | B |\n|---|---|\n{unsplittable}",
               "start_line": 0, "end_line": 2, "heading_hierarchy": ["Wide"]}]

    chunks = chunker.chunk_tables(tables)

    assert chunks
    for chunk in chunks:
        assert chunker._count_tokens(chunk.text) <= chunker.max_tokens
