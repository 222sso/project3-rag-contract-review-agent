"""Hierarchical chunking module for contract and policy documents.

Preserves Document -> Parent Section (Clause/Article) -> Child Chunk hierarchy.
Splits long sections using sentence boundaries and overlap while keeping
clause boundaries intact.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "project_config.json"
PARSED_DOCS_PATH = PROJECT_ROOT / "outputs" / "parsed_documents.json"
CHUNKS_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "chunks.json"


def load_config() -> dict[str, Any]:
    """Loads configuration settings from project_config.json."""
    if not CONFIG_PATH.exists():
        logger.warning("Config file not found at %s, using default values.", CONFIG_PATH)
        return {
            "chunking": {
                "parent_target_chars": 2400,
                "child_target_chars": 700,
                "child_overlap_chars": 120,
            }
        }
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def split_text_into_sentences(text: str) -> list[str]:
    """Splits text into cohesive sentence units based on newlines and punctuation."""
    lines = text.splitlines()
    sentences = []
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        # Split on sentence enders followed by whitespace, keeping delimiters
        parts = re.split(r"(?<=[.!?])\s+", line_str)
        for p in parts:
            p_str = p.strip()
            if p_str:
                sentences.append(p_str)
    return sentences


def create_child_chunks(
    text: str,
    heading: str,
    child_target_chars: int = 700,
    child_overlap_chars: int = 120,
) -> list[str]:
    """Splits a section's text into child chunks respecting target size and overlap.

    If the text fits within child_target_chars, returns a single chunk.
    Otherwise, splits along sentence boundaries with overlap.
    """
    clean_text = text.strip()
    full_content = f"{heading}\n{clean_text}" if clean_text and heading != clean_text else heading

    if len(full_content) <= child_target_chars:
        return [full_content]

    sentences = split_text_into_sentences(clean_text)
    if not sentences:
        return [full_content]

    chunks = []
    current_sentences = []
    current_len = len(heading) + 1  # heading + newline

    for sent in sentences:
        sent_len = len(sent) + 1  # sentence + newline
        if current_sentences and (current_len + sent_len > child_target_chars):
            # Form current chunk with heading prefix for context
            chunk_body = "\n".join(current_sentences)
            chunks.append(f"{heading}\n{chunk_body}")

            # Calculate overlap for next chunk
            overlap_sentences = []
            overlap_len = 0
            for prev_sent in reversed(current_sentences):
                if overlap_len + len(prev_sent) <= child_overlap_chars:
                    overlap_sentences.insert(0, prev_sent)
                    overlap_len += len(prev_sent) + 1
                else:
                    break
            
            current_sentences = overlap_sentences
            current_len = len(heading) + 1 + sum(len(s) + 1 for s in current_sentences)

        current_sentences.append(sent)
        current_len += sent_len

    if current_sentences:
        chunk_body = "\n".join(current_sentences)
        chunks.append(f"{heading}\n{chunk_body}")

    return chunks


def process_hierarchical_chunking() -> dict[str, Any]:
    """Performs hierarchical chunking across all parsed documents and saves chunks.json."""
    config = load_config()
    chunking_cfg = config.get("chunking", {})
    child_target_chars = chunking_cfg.get("child_target_chars", 700)
    child_overlap_chars = chunking_cfg.get("child_overlap_chars", 120)

    if not PARSED_DOCS_PATH.exists():
        raise FileNotFoundError(f"Parsed documents not found at {PARSED_DOCS_PATH}. Run pdf_parser first.")

    with open(PARSED_DOCS_PATH, "r", encoding="utf-8") as f:
        parsed_data = json.load(f)

    sections = parsed_data.get("sections", [])
    logger.info("Processing %d parent sections for hierarchical chunking...", len(sections))

    all_chunks: list[dict[str, Any]] = []
    doc_chunk_counts: dict[str, int] = {}

    for section in sections:
        doc_id = section["document_id"]
        doc_title = section["document_title"]
        category = section["category"]
        parent_section = section["heading"]
        page = section["page"]
        text = section.get("text", "")
        has_table = section.get("has_table", False)

        child_contents = create_child_chunks(
            text=text,
            heading=parent_section,
            child_target_chars=child_target_chars,
            child_overlap_chars=child_overlap_chars,
        )

        for child_idx, content in enumerate(child_contents):
            doc_chunk_counts[doc_id] = doc_chunk_counts.get(doc_id, 0) + 1
            chunk_seq = doc_chunk_counts[doc_id]
            chunk_id = f"{doc_id}-C{chunk_seq:03d}"

            chunk_record = {
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "document_title": doc_title,
                "category": category,
                "parent_section": parent_section,
                "page": page,
                "child_index": child_idx,
                "content": content,
                "char_length": len(content),
                "metadata": {
                    "has_table": has_table,
                    "section_id": section.get("section_id"),
                    "page": page,
                    "category": category,
                    "document_title": doc_title,
                },
            }
            all_chunks.append(chunk_record)

    # Calculate statistics
    lengths = [c["char_length"] for c in all_chunks]
    total_children = len(all_chunks)
    avg_len = sum(lengths) / total_children if total_children > 0 else 0
    max_len = max(lengths) if lengths else 0
    min_len = min(lengths) if lengths else 0
    empty_chunks = sum(1 for c in all_chunks if not c["content"].strip())
    short_chunks_under_20 = sum(1 for l in lengths if l < 20)

    stats = {
        "total_documents": len(parsed_data.get("documents", [])),
        "total_parent_sections": len(sections),
        "total_child_chunks": total_children,
        "average_chars": round(avg_len, 2),
        "max_chars": max_len,
        "min_chars": min_len,
        "empty_chunks_count": empty_chunks,
        "short_chunks_under_20_count": short_chunks_under_20,
    }

    output_payload = {
        "stats": stats,
        "chunks": all_chunks,
    }

    CHUNKS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    logger.info("Hierarchical chunking complete: %s", json.dumps(stats, ensure_ascii=False))
    return output_payload


if __name__ == "__main__":
    process_hierarchical_chunking()
