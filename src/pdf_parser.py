"""PDF Parser module using PyMuPDF to extract structured document sections.

Preserves document -> section (clause / article / heading) -> text relationships,
page numbers, and tabular data.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any
import pandas as pd
import pymupdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOC_INDEX_PATH = PROJECT_ROOT / "ground_truth" / "document_index.csv"

# Heading detection regex patterns
HEADING_PATTERNS = [
    # 제1조, 제 1 조, 제1조 [목적], 제1조 검수기준불명확
    re.compile(r"^(제\s*\d+\s*조(?:\s*\[[^\]]+\]|\s*\([^\)]+\)|\s*[^\n]+)?)$"),
    # 1. 목적, 1) 개요, 1.2 범위
    re.compile(r"^(\d+(?:\.\d+)*[\.\)]\s*[^\n]+)$"),
    # 부칙, 부 칙
    re.compile(r"^(부\s*칙(?:\s*[^\n]+)?)$"),
    # 서문, 전문, 총칙 등
    re.compile(r"^(서\s*문|전\s*문|목\s*차|개\s*요|총\s*칙)$"),
    # 대괄호 제목: [개요], [보안 원칙]
    re.compile(r"^(\[[^\]]{2,30}\])$"),
]


def is_heading_line(line: str) -> bool:
    """Checks if a given line matches section heading patterns."""
    text = line.strip()
    if not text or len(text) > 80:
        return False
    for pat in HEADING_PATTERNS:
        if pat.match(text):
            return True
    return False


def format_table_to_markdown(table_data: list[list[Any]]) -> str:
    """Formats 2D table data extracted from PyMuPDF into a Markdown table string."""
    if not table_data:
        return ""
    lines = []
    # Clean cells
    cleaned_rows = []
    for row in table_data:
        cleaned_rows.append([str(cell).replace("\n", " ").strip() if cell is not None else "" for cell in row])

    if not cleaned_rows:
        return ""

    header = cleaned_rows[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for row in cleaned_rows[1:]:
        # Pad or trim row to match header length
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[: len(header)]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def parse_single_pdf(
    file_path: Path,
    doc_id: str,
    title: str,
    category: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parses a single PDF into structured sections with metadata."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    doc = pymupdf.open(file_path)
    page_count = len(doc)
    sections: list[dict[str, Any]] = []
    empty_pages: list[int] = []

    current_heading = "[문서 서두]"
    current_page = 1
    current_texts: list[str] = []
    current_has_table = False

    section_seq = 1

    def flush_section():
        nonlocal section_seq, current_texts, current_has_table
        full_text = "\n".join(current_texts).strip()
        if full_text or current_heading != "[문서 서두]":
            sections.append({
                "section_id": f"{doc_id}-S{section_seq:02d}",
                "document_id": doc_id,
                "document_title": title,
                "category": category,
                "page": current_page,
                "heading": current_heading,
                "text": full_text,
                "has_table": current_has_table,
            })
            section_seq += 1
        current_texts = []
        current_has_table = False

    for page_idx in range(page_count):
        page_num = page_idx + 1
        page = doc[page_idx]

        # Check for tables on this page
        page_tables_md = []
        try:
            tabs = page.find_tables()
            if tabs and tabs.tables:
                for tab in tabs.tables:
                    extracted = tab.extract()
                    if extracted:
                        table_md = format_table_to_markdown(extracted)
                        if table_md:
                            page_tables_md.append(table_md)
        except Exception as e:
            logger.debug("Table detection error on page %d of %s: %s", page_num, doc_id, e)

        # Extract text blocks
        blocks = page.get_text("blocks")
        if not blocks:
            empty_pages.append(page_num)
            continue

        for block in blocks:
            block_text = block[4].strip()
            if not block_text:
                continue

            # Split block into lines to detect headings
            lines = [l.strip() for l in block_text.splitlines() if l.strip()]
            for line in lines:
                if is_heading_line(line):
                    # Flush previous section
                    flush_section()
                    current_heading = line
                    current_page = page_num
                else:
                    current_texts.append(line)

        # Append any table markdown extracted for this page
        if page_tables_md:
            current_has_table = True
            current_texts.append("\n[표 데이터]\n" + "\n\n".join(page_tables_md))

    # Flush final section
    flush_section()

    doc_summary = {
        "document_id": doc_id,
        "title": title,
        "category": category,
        "filename": file_path.name,
        "page_count": page_count,
        "section_count": len(sections),
        "empty_pages": empty_pages,
    }

    return sections, doc_summary


def parse_all_documents() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parses all 60 documents specified in document_index.csv and writes JSON output."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DOC_INDEX_PATH.exists():
        raise FileNotFoundError(f"Document index not found: {DOC_INDEX_PATH}")

    df = pd.read_csv(DOC_INDEX_PATH)
    logger.info("Starting parsing for %d documents...", len(df))

    all_sections: list[dict[str, Any]] = []
    doc_summaries: list[dict[str, Any]] = []
    failed_docs: list[str] = []

    category_folder_map = {
        "contract": "contracts",
        "policy": "policies",
        "security": "security",
        "guide": "guides",
    }

    for _, row in df.iterrows():
        doc_id = str(row["document_id"]).strip()
        category = str(row["category"]).strip()
        filename = str(row["filename"]).strip()
        title = str(row["title"]).strip()

        folder_name = category_folder_map.get(category, category)
        pdf_path = PROJECT_ROOT / "data" / "source_documents" / folder_name / filename

        try:
            sections, summary = parse_single_pdf(
                file_path=pdf_path,
                doc_id=doc_id,
                title=title,
                category=category,
            )
            all_sections.extend(sections)
            doc_summaries.append(summary)
        except Exception as e:
            logger.error("[%s] Parsing failed for %s: %s", doc_id, filename, e)
            failed_docs.append(doc_id)

    output_payload = {
        "total_documents": len(df),
        "parsed_documents": len(doc_summaries),
        "total_sections": len(all_sections),
        "failed_count": len(failed_docs),
        "failed_documents": failed_docs,
        "documents": doc_summaries,
        "sections": all_sections,
    }

    output_file = OUTPUTS_DIR / "parsed_documents.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    logger.info("Saved %d sections from %d documents to %s", len(all_sections), len(doc_summaries), output_file)
    return all_sections, doc_summaries


if __name__ == "__main__":
    parse_all_documents()
