from pathlib import Path
import json
import re
from typing import Dict, List, Tuple
from dataclasses import asdict

from .utils import ChunkIdGenerator, PageMapper
from .section_chunker import SectionChunker
from .table_chunker import TableChunker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
markdown_path = PROJECT_ROOT / "data" / "processed" / "2026" / "Budget_Speech_pymupdf4llm.md"
output_dir = PROJECT_ROOT / "data" / "processed" / "2026" / "chunks"

def is_valid_heading(text: str) -> bool:
    """Check if text is a valid heading (not a date, page number, etc.)."""
    text_clean = text.strip('**').strip()
    
    # Remove HTML tags
    text_clean = re.sub(r'<[^>]+>', '', text_clean).strip()
    
    # Filter out dates
    if re.match(r'^\d{1,2}(st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)', text_clean, re.IGNORECASE):
        return False
    # Filter out standalone page numbers
    if re.match(r'^\d+$', text_clean):
        return False
    # Filter out very short headings
    if len(text_clean) < 3:
        return False
    # Filter out common non-heading patterns (names, titles, dates)
    invalid_patterns = [
        r'^Hon', r'^Speaker', r'^Minister', 
        r'^Nirmala', r'^Sitharaman', r'^February', r'^January', 
        r'^March', r'^April', r'^May', r'^June', r'^July', 
        r'^August', r'^September', r'^October', r'^November', r'^December',
        r'^GOVERNMENT OF INDIA', r'^BUDGET', r'^SPEECH', r'^OF',
        r'^CONTENTS', r'^Page No'
    ]
    for pattern in invalid_patterns:
        if re.match(pattern, text_clean, re.IGNORECASE):
            return False
    # Filter out headings that are just formatting
    if re.match(r'^[<>\-_]+$', text_clean):
        return False
    return True

def parse_markdown_structure(md_text: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Parse markdown into paragraphs, headings, and tables."""
    lines = md_text.split('\n')
    
    paragraphs = []
    headings = []
    tables = []
    
    current_heading = {"level": 0, "text": "", "line_number": 0}
    heading_hierarchy = []
    
    current_table = []
    in_table = False
    table_start_line = 0
    
    for i, line in enumerate(lines):
        # Check for headings
        heading_match = re.match(r'^(#{1,4})\s+(.+)$', line)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            
            if not is_valid_heading(text):
                continue
            
            # Update heading hierarchy - reset for level 1, replace for same level, append for deeper
            if level == 1:
                heading_hierarchy = [text]
            elif level == 2:
                if len(heading_hierarchy) >= 1:
                    heading_hierarchy = heading_hierarchy[:1] + [text]
                else:
                    heading_hierarchy = [text]
            elif level == 3:
                if len(heading_hierarchy) >= 2:
                    heading_hierarchy = heading_hierarchy[:2] + [text]
                elif len(heading_hierarchy) >= 1:
                    heading_hierarchy = heading_hierarchy[:1] + [text]
                else:
                    heading_hierarchy = [text]
            elif level == 4:
                if len(heading_hierarchy) >= 3:
                    heading_hierarchy = heading_hierarchy[:3] + [text]
                elif len(heading_hierarchy) >= 2:
                    heading_hierarchy = heading_hierarchy[:2] + [text]
                elif len(heading_hierarchy) >= 1:
                    heading_hierarchy = heading_hierarchy[:1] + [text]
                else:
                    heading_hierarchy = [text]
            
            headings.append({
                "level": level,
                "text": text,
                "line_number": i,
                "hierarchy": heading_hierarchy.copy()
            })
            current_heading = {"level": level, "text": text, "line_number": i}
            continue
        
        # Check for numbered paragraphs
        para_patterns = [
            r'^\*\*(\d+)\.\*\*\s*(.+)$',  # **1.** text
            r'^\*\*(\d+)\.\s*(.+)$',  # **1.** text (without closing **)
            r'^\*\*(\d+)\s*\.<mark>\s*(.+)</mark>',  # **1.** <mark>text</mark>
            r'^<mark>\*\*(\d+)\.\*\*</mark>\s*(.+)$',  # <mark>**1.**</mark> text
            r'^\*\*(\d+)\.<mark>\s*(.+)</mark>',  # **1.<mark> text</mark>
            r'^<mark>\*\*(\d+)\.</mark>\s*(.+)$',  # <mark>**1.</mark> text
        ]
        
        para_match = None
        para_num = None
        content = None
        
        for pattern in para_patterns:
            match = re.match(pattern, line)
            if match:
                para_match = match
                para_num = int(match.group(1))
                content = match.group(2)
                break
        
        if para_match:
            paragraphs.append({
                "number": para_num,
                "content": content,
                "line_number": i,
                "heading_hierarchy": heading_hierarchy.copy(),
                "heading_level": current_heading["level"],
                "heading_text": current_heading["text"]
            })
            continue
        
        # Check for tables
        line_stripped = line.strip()
        
        if line_stripped.startswith('|') and line_stripped.count('|') >= 2:
            if not in_table:
                in_table = True
                table_start_line = i
            current_table.append(line)
        elif in_table and '|' in line_stripped:
            current_table.append(line)
        elif in_table:
            if line_stripped == '':
                if len(current_table) > 0 and current_table[-1].strip() == '':
                    if len(current_table) >= 2:
                        tables.append({
                            "content": '\n'.join(current_table),
                            "start_line": table_start_line,
                            "end_line": i - 1,
                            "heading_hierarchy": heading_hierarchy.copy()
                        })
                    current_table = []
                    in_table = False
                else:
                    current_table.append(line)
            else:
                if len(current_table) >= 2:
                    tables.append({
                        "content": '\n'.join(current_table),
                        "start_line": table_start_line,
                        "end_line": i - 1,
                        "heading_hierarchy": heading_hierarchy.copy()
                    })
                current_table = []
                in_table = False
    
    if in_table and len(current_table) >= 2:
        tables.append({
            "content": '\n'.join(current_table),
            "start_line": table_start_line,
            "end_line": len(lines) - 1,
            "heading_hierarchy": heading_hierarchy.copy()
        })
    
    return paragraphs, headings, tables

def group_by_section(paragraphs: List[Dict]) -> Dict[str, List[Dict]]:
    """Group paragraphs by their complete valid heading hierarchy."""
    sections = {}
    
    for para in paragraphs:
        valid_hierarchy = [h for h in para['heading_hierarchy'] if is_valid_heading(h)]
        key = "|".join(valid_hierarchy) if valid_hierarchy else "Unknown"
        
        if key not in sections:
            sections[key] = {
                "heading_path": valid_hierarchy,
                "paragraphs": []
            }
        sections[key]["paragraphs"].append(para)
    
    # Merge very small sections into adjacent sections
    section_keys = list(sections.keys())
    for i in range(len(section_keys) - 1):
        current_key = section_keys[i]
        next_key = section_keys[i + 1]
        
        if len(sections[current_key]["paragraphs"]) < 3:
            # Merge current into next
            sections[next_key]["paragraphs"] = sections[current_key]["paragraphs"] + sections[next_key]["paragraphs"]
            sections[next_key]["heading_path"] = sections[current_key]["heading_path"]  # Keep the first heading
            del sections[current_key]
        
    return sections

def chunk_budget_document():
    """Main pipeline to parse markdown and generate chunks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    md_text = markdown_path.read_text(encoding='utf-8')
    
    page_mapper = PageMapper(md_text)
    id_generator = ChunkIdGenerator(document="budget_speech", year=2026)
    
    section_chunker = SectionChunker(id_generator, page_mapper)
    table_chunker = TableChunker(id_generator, page_mapper)
    
    paragraphs, headings, tables = parse_markdown_structure(md_text)
    
    sections = group_by_section(paragraphs)
    
    all_chunks = []
    
    for section_key, section_data in sections.items():
        chunks = section_chunker.chunk_section(
            paragraphs=section_data["paragraphs"],
            heading_path=section_data["heading_path"]
        )
        all_chunks.extend(chunks)
        
    table_chunks = table_chunker.chunk_tables(tables)
    all_chunks.extend(table_chunks)
    
    # Save output
    chunks_data = [asdict(chunk) for chunk in all_chunks]
    
    with open(output_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
    print(f"Total chunks generated: {len(all_chunks)}")

if __name__ == "__main__":
    chunk_budget_document()
