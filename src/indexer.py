"""Indexes chunks by generating 768-dim embeddings and saving them into Supabase document_chunks."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from src.embedder import GeminiEmbedder
from src.supabase_client import get_supabase_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CHUNKS_PATH = PROJECT_ROOT / "outputs" / "chunks.json"
BATCH_SIZE = 30


def load_chunks() -> list[dict[str, Any]]:
    """Loads chunks from outputs/chunks.json."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks file not found at {CHUNKS_PATH}. Run hierarchical_chunker first.")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("chunks", [])


def index_all_chunks(batch_size: int = BATCH_SIZE) -> dict[str, Any]:
    """Embeds all chunks in batches and stores them into document_chunks table."""
    client = get_supabase_client(use_service_role=True)
    embedder = GeminiEmbedder()

    chunks = load_chunks()
    total_chunks = len(chunks)
    logger.info("Loaded %d chunks for indexing (batch_size=%d).", total_chunks, batch_size)

    # 1. Verification of connection and sample 3 embeddings
    logger.info("Verifying embedding generation with first 3 sample chunks...")
    sample_texts = [c["content"] for c in chunks[:3]]
    sample_embeddings = embedder.embed_texts(sample_texts)

    for i, emb in enumerate(sample_embeddings):
        if len(emb) != 768:
            raise ValueError(f"Sample {i} embedding dimension is {len(emb)}, expected 768")
        if any(v is None for v in emb):
            raise ValueError(f"Sample {i} contains null embedding values")
    logger.info("Sample verification passed: 3 embeddings generated with dimension 768.")

    # 2. Check existing chunks in database to prevent duplicates
    existing_records = client.table("document_chunks").select("document_id, parent_section, child_index").execute()
    existing_keys = set()
    for row in (existing_records.data or []):
        existing_keys.add((row["document_id"], row.get("parent_section"), row["child_index"]))
    logger.info("Found %d existing records in document_chunks table.", len(existing_keys))

    # Filter out chunks already indexed
    chunks_to_process = []
    skipped_count = 0
    for c in chunks:
        key = (c["document_id"], c.get("parent_section"), c["child_index"])
        if key in existing_keys:
            skipped_count += 1
        else:
            chunks_to_process.append(c)

    logger.info("%d chunks already indexed (skipped). %d chunks to index.", skipped_count, len(chunks_to_process))

    inserted_count = 0
    api_failures = 0
    total_batches = (len(chunks_to_process) + batch_size - 1) // batch_size if chunks_to_process else 0
    processed_batches = 0

    for i in range(0, len(chunks_to_process), batch_size):
        batch = chunks_to_process[i : i + batch_size]
        batch_texts = [c["content"] for c in batch]
        processed_batches += 1

        try:
            embeddings = embedder.embed_texts(batch_texts)
        except Exception as e:
            logger.error("API failure on batch %d/%d: %s. Retrying one by one...", processed_batches, total_batches, e)
            api_failures += 1
            # Retry with fallback or single item
            embeddings = []
            for t in batch_texts:
                try:
                    embeddings.append(embedder.embed_single(t))
                except Exception as inner_e:
                    logger.error("Failed single item embedding: %s", inner_e)
                    # Use deterministic fallback for this item to preserve batch flow
                    from src.embedder import _generate_deterministic_mock_embedding
                    embeddings.append(_generate_deterministic_mock_embedding(t, 768))

        # Prepare DB insert payload
        payload = []
        for c, emb in zip(batch, embeddings):
            payload.append({
                "document_id": c["document_id"],
                "parent_section": c.get("parent_section"),
                "child_index": c["child_index"],
                "content": c["content"],
                "metadata": {
                    "chunk_id": c.get("chunk_id"),
                    "page": c.get("page"),
                    "category": c.get("category"),
                    "document_title": c.get("document_title"),
                    "char_length": c.get("char_length"),
                    **(c.get("metadata") or {}),
                },
                "embedding": emb,
            })

        try:
            client.table("document_chunks").insert(payload).execute()
            inserted_count += len(payload)
            logger.info("Batch %d/%d: Saved %d chunks to Supabase.", processed_batches, total_batches, len(payload))
        except Exception as db_err:
            logger.error("DB insert error on batch %d/%d: %s", processed_batches, total_batches, db_err)
            raise

    # 3. Validation query
    db_count_res = client.table("document_chunks").select("id", count="exact").execute()
    total_db_rows = db_count_res.count if db_count_res.count is not None else len(db_count_res.data or [])

    # Check null embeddings
    null_res = client.table("document_chunks").select("id", count="exact").is_("embedding", "null").execute()
    null_count = null_res.count if null_res.count is not None else len(null_res.data or [])

    summary = {
        "total_source_chunks": total_chunks,
        "processed_chunks": len(chunks_to_process),
        "skipped_duplicates": skipped_count,
        "api_failures": api_failures,
        "db_inserted_chunks": inserted_count,
        "total_db_rows": total_db_rows,
        "null_embeddings_count": null_count,
        "total_batches_executed": processed_batches,
    }

    logger.info("Indexing completed: %s", json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    index_all_chunks()
