from pathlib import Path
import json

import fitz


PROJECT_ROOT = Path(__file__).resolve().parents[2]

pdf_path = PROJECT_ROOT / "data" / "raw" / "2026" / "Budget_Speech.pdf"

output_dir = PROJECT_ROOT / "data" / "processed" / "2026"
output_dir.mkdir(parents=True, exist_ok=True)

doc = fitz.open(pdf_path)

pages = []
markdown = []

for page_number, page in enumerate(doc, start=1):
    text = page.get_text()

    pages.append(
        {
            "page": page_number,
            "text": text
        }
    )

    markdown.append(f"# Page {page_number}\n\n{text}")

metadata = {
    "source_file": pdf_path.name,
    "source_path": str(pdf_path),
    "page_count": len(doc),
    "title": doc.metadata.get("title"),
    "author": doc.metadata.get("author"),
    "subject": doc.metadata.get("subject"),
    "creator": doc.metadata.get("creator"),
    "producer": doc.metadata.get("producer")
}

(output_dir / f"{pdf_path.stem}.md").write_text(
    "\n\n".join(markdown),
    encoding="utf-8"
)

with open(output_dir / f"{pdf_path.stem}.json", "w", encoding="utf-8") as f:
    json.dump(pages, f, indent=2, ensure_ascii=False)

with open(output_dir / f"{pdf_path.stem}_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

doc.close()