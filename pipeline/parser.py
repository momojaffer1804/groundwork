#Phase 1 parser : PyMuPDF text extraction.

#Goal for this phase: 
# 1. prove we can pull clean-enough text out of a research paper pdf
# 2. just extracting raw data 
# 3. this gets replaced by docling(converts complex docs into clean text) in phase 2

import re
import sys
from pathlib import Path

import fitz  # PyMuPDF


def _strip_page_numbers(text: str) -> str:
    """
    Strip lines that are just a standalone page number (common PDF
    footer/header pattern). Left unstripped, these leak into chunks
    as stray digits that can get misread as real numeric answers
    (e.g. a page number "15" getting extracted as the answer to
    "how many attention heads" instead of the actual answer, "8").
    """
    lines = text.split("\n")
    cleaned = [line for line in lines if not re.fullmatch(r"\s*\d{1,4}\s*", line)]
    return "\n".join(cleaned)


def extract_text(pdf_path: str) -> str:
    """
    Extract raw text from a PDF, page by page, concatenated with
    page-break markers so we can tell where pages split during
    debugging. Standalone page-number lines are stripped out.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"No PDF found at {pdf_path}")

    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        text = _strip_page_numbers(text)
        pages.append(f"\n--- Page {page_num} ---\n{text}")
    doc.close()

    return "".join(pages)
if __name__ == "__main__":
    # Usage: python pipeline/parser.py sample_papers/your_paper.pdf
    if len(sys.argv) != 2:
        print("Usage: python pipeline/parser.py <path_to_pdf>")
        sys.exit(1)

    full_text = extract_text(sys.argv[1])

    print(full_text)
    print("\n\n=====  SANITY CHECK =====")
    print(f"Total characters extracted: {len(full_text)}")
    print(f"Total pages: {full_text.count('--- Page')}")