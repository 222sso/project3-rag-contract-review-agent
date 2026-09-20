"""Uploads 60 PDF documents to Supabase Storage and records metadata into documents table."""

import json
import logging
import os
from pathlib import Path
import pandas as pd
from src.supabase_client import get_supabase_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUCKET_NAME = "rag-documents"

CATEGORY_TO_STORAGE_DIR = {
    "contract": "contracts",
    "policy": "policies",
    "security": "security",
    "guide": "guides",
}


def ensure_private_bucket(client, bucket_name: str = BUCKET_NAME):
    """Ensures that a private storage bucket exists."""
    try:
        buckets = client.storage.list_buckets()
        bucket_dict = {b.name: b for b in buckets}
        if bucket_name in bucket_dict:
            b = bucket_dict[bucket_name]
            is_public = getattr(b, "public", False)
            logger.info("Bucket '%s' exists (public=%s).", bucket_name, is_public)
            if is_public:
                logger.warning("Warning: Bucket '%s' is public. Ensuring private access is advised.", bucket_name)
            return
        
        logger.info("Bucket '%s' does not exist. Creating private bucket...", bucket_name)
        client.storage.create_bucket(bucket_name, options={"public": False})
        logger.info("Private bucket '%s' created successfully.", bucket_name)
    except Exception as e:
        logger.error("Error checking or creating bucket '%s': %s", bucket_name, e)
        raise


def get_existing_storage_files(client, bucket_name: str, folder: str) -> dict[str, int]:
    """Lists files in a specific folder within the bucket. Returns {filename: size}."""
    existing = {}
    try:
        items = client.storage.from_(bucket_name).list(path=folder)
        for item in items:
            name = item.get("name") if isinstance(item, dict) else getattr(item, "name", None)
            metadata = item.get("metadata") if isinstance(item, dict) else getattr(item, "metadata", None)
            size = None
            if isinstance(metadata, dict):
                size = metadata.get("size")
            if name:
                existing[name] = size
    except Exception as e:
        logger.warning("Could not list files in '%s/%s': %s", bucket_name, folder, e)
    return existing


def upload_documents_and_record_metadata():
    """Main execution function to upload documents and insert records into PostgreSQL."""
    client = get_supabase_client(use_service_role=True)
    ensure_private_bucket(client, BUCKET_NAME)

    doc_index_path = PROJECT_ROOT / "ground_truth" / "document_index.csv"
    if not doc_index_path.exists():
        raise FileNotFoundError(f"document_index.csv not found at {doc_index_path}")

    df = pd.read_csv(doc_index_path)
    logger.info("Loaded %d documents from index.", len(df))

    # Pre-fetch existing documents table records to avoid duplicate inserts
    db_docs = client.table("documents").select("document_id, storage_path").execute()
    db_doc_map = {row["document_id"]: row for row in (db_docs.data or [])}

    # Cache storage files per category folder
    storage_cache = {}
    for cat_dir in set(CATEGORY_TO_STORAGE_DIR.values()):
        storage_cache[cat_dir] = get_existing_storage_files(client, BUCKET_NAME, cat_dir)

    success_count = 0
    duplicate_count = 0
    failure_count = 0

    category_stats = {"contract": 0, "policy": 0, "security": 0, "guide": 0}

    for _, row in df.iterrows():
        doc_id = str(row["document_id"]).strip()
        category = str(row["category"]).strip()
        filename = str(row["filename"]).strip()
        title = str(row["title"]).strip()

        folder_name = CATEGORY_TO_STORAGE_DIR.get(category, category)
        local_file_path = PROJECT_ROOT / "data" / "source_documents" / folder_name / filename
        storage_path = f"{folder_name}/{filename}"

        if not local_file_path.exists():
            logger.error("[%s] Local file not found: %s", doc_id, local_file_path)
            failure_count += 1
            continue

        file_size = local_file_path.stat().st_size

        try:
            # Check if file already exists in storage with same name and size
            already_in_storage = False
            existing_folder_files = storage_cache.get(folder_name, {})
            if filename in existing_folder_files:
                existing_size = existing_folder_files[filename]
                if existing_size is None or existing_size == file_size:
                    already_in_storage = True

            if not already_in_storage:
                logger.info("[%s] Uploading %s to Storage (%s)...", doc_id, filename, storage_path)
                with open(local_file_path, "rb") as f:
                    file_bytes = f.read()
                client.storage.from_(BUCKET_NAME).upload(
                    path=storage_path,
                    file=file_bytes,
                    file_options={"content-type": "application/pdf", "upsert": "false"},
                )
                existing_folder_files[filename] = file_size
            else:
                logger.info("[%s] File %s already exists in storage with identical size. Skipping storage upload.", doc_id, storage_path)

            # Insert / update metadata in documents table
            doc_metadata = {
                "filename": filename,
                "file_size_bytes": file_size,
                "original_category": category,
            }

            db_payload = {
                "document_id": doc_id,
                "title": title,
                "category": category,
                "storage_path": storage_path,
                "metadata": doc_metadata,
            }

            client.table("documents").upsert(db_payload, on_conflict="document_id").execute()

            category_stats[category] = category_stats.get(category, 0) + 1
            if already_in_storage and doc_id in db_doc_map:
                duplicate_count += 1
            else:
                success_count += 1

        except Exception as e:
            logger.error("[%s] Failed to process document %s: %s", doc_id, filename, e)
            failure_count += 1

    summary = {
        "total_attempted": len(df),
        "success": success_count,
        "duplicate_skipped": duplicate_count,
        "failed": failure_count,
        "category_stats": category_stats,
    }
    logger.info("Upload process completed: %s", json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    upload_documents_and_record_metadata()
