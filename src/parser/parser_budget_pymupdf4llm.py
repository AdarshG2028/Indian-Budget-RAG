from pathlib import Path
import json
from typing import Dict, List, Any
import pymupdf4llm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

pdf_path = PROJECT_ROOT / "data" / "raw" / "2026" / "Budget_Speech.pdf"
output_dir = PROJECT_ROOT / "data" / "processed" / "2026"
output_dir.mkdir(parents=True, exist_ok=True)


def parse_budget_pdf():
    """Parse Budget PDF using pymupdf4llm with structural extraction."""
    
    # Convert PDF to markdown using pymupdf4llm
    md_text = pymupdf4llm.to_markdown(str(pdf_path))
    
    # Also get page-by-page markdown
    pages_md = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    
    # Get document metadata
    import fitz
    doc = fitz.open(pdf_path)
    
    metadata = {
        "source_file": pdf_path.name,
        "source_path": str(pdf_path),
        "page_count": len(doc),
        "parser": "pymupdf4llm",
        "total_pages": len(pages_md)
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
    
    print(f"Parsed {len(pages_md)} pages using pymupdf4llm")
    print(f"Total markdown length: {len(md_text)} characters")
    print(f"Average page length: {sum(len(p) for p in pages_md) / len(pages_md):.0f} characters")
    
    # Analyze structure
    heading_count = md_text.count('#')
    table_count = md_text.count('|') // 10  # Rough estimate
    print(f"Estimated headings: {heading_count}")
    print(f"Estimated tables: {table_count}")
    
    return metadata, md_text, pages_data


if __name__ == "__main__":
    parse_budget_pdf()
