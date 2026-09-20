"""Document chunk retriever module using Supabase match_document_chunks RPC."""

import json
import logging
from pathlib import Path
from typing import Any
import pandas as pd
from src.embedder import GeminiEmbedder
from src.supabase_client import get_supabase_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "project_config.json"


def load_retrieval_config() -> dict[str, Any]:
    """Loads retrieval configuration settings."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            return cfg.get("retrieval", {"top_k": 8, "use_metadata_filter": True})
    return {"top_k": 8, "use_metadata_filter": True}


class DocumentRetriever:
    """Retriever class for semantic chunk search via Supabase pgvector RPC."""

    def __init__(self, embedder: GeminiEmbedder | None = None, min_similarity: float | None = None):
        self.client = get_supabase_client(use_service_role=True)
        self.embedder = embedder or GeminiEmbedder()
        self.config = load_retrieval_config()
        self.default_top_k = self.config.get("top_k", 8)
        self.min_similarity = min_similarity

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_category: str | None = None,
        min_similarity: float | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieves top_k relevant chunks matching the query.

        Args:
            query: The user search query or contract review question.
            top_k: Number of chunks to retrieve (defaults to config value: 8).
            filter_category: Optional category filter ('contract', 'policy', 'security', 'guide').
            min_similarity: Minimum cosine similarity threshold (optional).

        Returns:
            list[dict]: List of retrieved chunk dictionaries sorted by similarity descending.
        """
        k = top_k or self.default_top_k
        threshold = min_similarity if min_similarity is not None else self.min_similarity

        # 1. Embed query into 768-dim vector
        query_vector = self.embedder.embed_single(query)

        # 2. Call Supabase RPC
        params = {
            "query_embedding": query_vector,
            "match_count": k,
            "filter_category": filter_category,
        }

        try:
            response = self.client.rpc("match_document_chunks", params).execute()
            raw_results = response.data or []
        except Exception as e:
            logger.error("RPC match_document_chunks error: %s", e)
            raise

        # 3. Format and enrich results
        formatted_results = []
        for row in raw_results:
            sim = float(row.get("similarity", 0.0))
            if threshold is not None and sim < threshold:
                continue

            metadata = row.get("metadata") or {}
            formatted_results.append({
                "id": row.get("id"),
                "document_id": row.get("document_id"),
                "document_title": metadata.get("document_title", ""),
                "category": metadata.get("category", ""),
                "section": row.get("parent_section") or metadata.get("section_id", ""),
                "page": metadata.get("page", 1),
                "similarity": round(sim, 4),
                "content": row.get("content", ""),
                "metadata": metadata,
            })

        return formatted_results


def run_sample_evaluations(questions: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    """Runs retrieval for a set of evaluation questions and returns summarized results."""
    retriever = DocumentRetriever()

    if not questions:
        # Load first 5 questions from evaluation_questions.csv as standard test set
        eval_path = PROJECT_ROOT / "ground_truth" / "evaluation_questions.csv"
        if eval_path.exists():
            df_eval = pd.read_csv(eval_path)
            questions = df_eval.head(5).to_dict(orient="records")
        else:
            questions = [
                {"question_id": "Q001", "question": "자동갱신 또는 장기 해지통보 조건을 찾아줘.", "target_document": "contract_01.pdf"},
                {"question_id": "Q002", "question": "개인정보 재위탁과 관련된 조항이 있는지 확인해줘.", "target_document": "contract_02.pdf"},
                {"question_id": "Q003", "question": "손해배상 책임에 상한이 있는지 확인해줘.", "target_document": "contract_03.pdf"},
                {"question_id": "Q004", "question": "이 계약서의 가장 높은 위험 조항은 무엇이며 근거를 제시해줘.", "target_document": "contract_04.pdf"},
                {"question_id": "Q005", "question": "지식재산권 귀속 조건을 요약해줘.", "target_document": "contract_05.pdf"},
            ]

    eval_results = []
    for q_item in questions:
        qid = q_item.get("question_id", "Q")
        q_text = q_item.get("question", "")
        target_doc = q_item.get("target_document", "")

        logger.info("Executing retrieval for [%s]: '%s' (Target: %s)", qid, q_text, target_doc)
        matches = retriever.retrieve(query=q_text, top_k=5)

        eval_results.append({
            "question_id": qid,
            "question": q_text,
            "target_document": target_doc,
            "top_5_matches": matches[:5],
        })

    return eval_results


if __name__ == "__main__":
    results = run_sample_evaluations()
    for r in results:
        print(f"\n==========================================")
        print(f"[{r['question_id']}] Question: {r['question']}")
        print(f"Target Document: {r['target_document']}")
        print(f"------------------------------------------")
        if not r["top_5_matches"]:
            print("  (No matching chunks found)")
        for idx, m in enumerate(r["top_5_matches"], 1):
            print(f"  {idx}. [{m['document_id']}] {m['section']} (p.{m['page']}) | Similarity: {m['similarity']}")
            content_preview = m['content'].replace('\n', ' ')[:80]
            print(f"     Content: {content_preview}...")
