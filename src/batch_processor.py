from pathlib import Path
import json
from typing import Dict, List, Tuple
import pymupdf4llm
import fitz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "2026"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "2026"

# Skip non-PDF files
SKIP_FILES = {"ChatGPT Installer.exe"}


def parse_pdf(pdf_path: Path, output_dir: Path) -> Tuple[Dict, str, List]:
    """Parse a single PDF using pymupdf4llm."""
    print(f"\nParsing: {pdf_path.name}")
    
    # Convert PDF to markdown
    md_text = pymupdf4llm.to_markdown(str(pdf_path))
    pages_md = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    
    # Get document metadata
    doc = fitz.open(pdf_path)
    
    metadata = {
        "source_file": pdf_path.name,
        "source_path": str(pdf_path),
        "page_count": len(doc),
        "parser": "pymupdf4llm",
        "total_pages": len(pages_md),
        "file_size": pdf_path.stat().st_size
    }
    
    # Save complete markdown
    (output_dir / f"{pdf_path.stem}_pymupdf4llm.md").write_text(
        md_text,
        encoding="utf-8"
    )
    
    # Save page-by-page markdown as JSON
    pages_data = []
    for i, page_md in enumerate(pages_md, start=1):
        pages_data.append({
            "page": i,
            "markdown": page_md,
            "char_count": len(page_md)
        })
    
    with open(output_dir / f"{pdf_path.stem}_pymupdf4llm_pages.json", "w", encoding="utf-8") as f:
        json.dump({
            "metadata": metadata,
            "pages": pages_data
        }, f, indent=2, ensure_ascii=False)
    
    # Save metadata
    with open(output_dir / f"{pdf_path.stem}_pymupdf4llm_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    doc.close()
    
    print(f"  Pages: {len(pages_md)}")
    print(f"  Total chars: {len(md_text):,}")
    print(f"  Avg page chars: {sum(len(p) for p in pages_md) / len(pages_md):.0f}")
    
    return metadata, md_text, pages_data


def chunk_pdf(md_text: str, pdf_name: str, output_dir: Path) -> Dict:
    """Chunk a parsed PDF using the chunking pipeline."""
    print(f"Chunking: {pdf_name}")
    
    # Import chunking modules
    import sys
    sys.path.insert(0, str(PROCESSED_DIR.parent.parent / "src"))
    from chunking.main import parse_markdown_structure, group_by_section
    from chunking.utils import PageMapper, ChunkIdGenerator
    from chunking.section_chunker import SectionChunker
    from chunking.table_chunker import TableChunker
    from chunking.config import config
    from dataclasses import asdict
    
    # Initialize components
    page_mapper = PageMapper(md_text)
    id_generator = ChunkIdGenerator(document=pdf_name.replace(".pdf", "").replace("-", "_"), year=2026)
    section_chunker = SectionChunker(id_generator, page_mapper)
    table_chunker = TableChunker(id_generator, page_mapper)
    
    # Parse structure
    paragraphs, headings, tables = parse_markdown_structure(md_text)
    
    print(f"  Paragraphs: {len(paragraphs)}")
    print(f"  Headings: {len(headings)}")
    print(f"  Tables: {len(tables)}")
    
    # Group by section
    sections = group_by_section(paragraphs)
    
    # Merge very small sections
    section_keys = list(sections.keys())
    for i in range(len(section_keys) - 1):
        current_key = section_keys[i]
        next_key = section_keys[i + 1]
        
        if len(sections[current_key]["paragraphs"]) < 3:
            sections[next_key]["paragraphs"] = sections[current_key]["paragraphs"] + sections[next_key]["paragraphs"]
            sections[next_key]["heading_path"] = sections[current_key]["heading_path"]
            del sections[current_key]
    
    # Chunk sections
    all_chunks = []
    for section_key, section_data in sections.items():
        chunks = section_chunker.chunk_section(
            paragraphs=section_data["paragraphs"],
            heading_path=section_data["heading_path"]
        )
        all_chunks.extend(chunks)
    
    # Chunk tables
    table_chunks = table_chunker.chunk_tables(tables)
    all_chunks.extend(table_chunks)
    
    # Save chunks
    chunks_output_dir = output_dir / "chunks" / pdf_name.replace(".pdf", "")
    chunks_output_dir.mkdir(parents=True, exist_ok=True)
    
    chunks_data = [asdict(chunk) for chunk in all_chunks]
    
    with open(chunks_output_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    # Save statistics
    stats = {
        "source_file": pdf_name,
        "total_chunks": len(all_chunks),
        "text_chunks": len(all_chunks) - len(table_chunks),
        "table_chunks": len(table_chunks),
        "total_tokens": sum(c.token_count for c in all_chunks),
        "avg_tokens": sum(c.token_count for c in all_chunks) / len(all_chunks) if all_chunks else 0,
        "max_tokens": max(c.token_count for c in all_chunks) if all_chunks else 0,
        "min_tokens": min(c.token_count for c in all_chunks) if all_chunks else 0
    }
    
    with open(chunks_output_dir / "chunk_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Avg tokens: {stats['avg_tokens']:.0f}")
    
    return stats


def process_all_pdfs():
    """Process all PDFs in the raw directory."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all PDF files
    pdf_files = [f for f in RAW_DIR.glob("*.pdf") if f.name not in SKIP_FILES]
    
    print(f"Found {len(pdf_files)} PDF files to process")
    print("=" * 60)
    
    all_stats = []
    
    for pdf_path in pdf_files:
        try:
            chunks_file = PROCESSED_DIR / "chunks" / pdf_path.stem / "chunks.json"
            if chunks_file.exists():
                print(f"Skipping already processed: {pdf_path.name}")
                continue
                
            # Parse PDF
            metadata, md_text, pages_data = parse_pdf(pdf_path, PROCESSED_DIR)
            
            # Chunk PDF
            chunk_stats = chunk_pdf(md_text, pdf_path.name, PROCESSED_DIR)
            
            all_stats.append({
                "file": pdf_path.name,
                "parse_metadata": metadata,
                "chunk_stats": chunk_stats
            })
            
            print(f"[SUCCESS] Completed: {pdf_path.name}")
            
        except Exception as e:
            print(f"[FAILED] Failed: {pdf_path.name} - {str(e)}")
            all_stats.append({
                "file": pdf_path.name,
                "error": str(e)
            })
    
    # Save overall statistics
    with open(PROCESSED_DIR / "batch_processing_stats.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    
    successful = [s for s in all_stats if "error" not in s]
    failed = [s for s in all_stats if "error" in s]
    
    print(f"Successfully processed: {len(successful)}/{len(pdf_files)}")
    print(f"Failed: {len(failed)}/{len(pdf_files)}")
    
    if successful:
        total_chunks = sum(s["chunk_stats"]["total_chunks"] for s in successful)
        total_tokens = sum(s["chunk_stats"]["total_tokens"] for s in successful)
        print(f"Total chunks generated: {total_chunks}")
        print(f"Total tokens: {total_tokens:,}")
    
    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  - {f['file']}: {f['error']}")


if __name__ == "__main__":
    process_all_pdfs()
