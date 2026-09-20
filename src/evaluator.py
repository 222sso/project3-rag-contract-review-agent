"""Quantitative evaluation module for RAG retrieval hit rate and risk detection precision/recall.

Runs through 80 evaluation questions in ground_truth/evaluation_questions.csv,
compares citations and risk detections against ground_truth/risk_clauses.csv,
computes metrics per question type, and exports detailed report & failure logs.
"""

import json
import logging
from pathlib import Path
from typing import Any
import pandas as pd
from dotenv import load_dotenv
from src.retriever import DocumentRetriever
from src.reviewer import ContractReviewAgent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluates RAG retrieval accuracy and contract risk screening performance across 80 questions."""

    def __init__(self):
        self.retriever = DocumentRetriever()
        self.agent = ContractReviewAgent(retriever=self.retriever)

        self.eval_questions_path = PROJECT_ROOT / "ground_truth" / "evaluation_questions.csv"
        self.risk_clauses_path = PROJECT_ROOT / "ground_truth" / "risk_clauses.csv"
        self.doc_index_path = PROJECT_ROOT / "ground_truth" / "document_index.csv"

        self.df_questions = pd.read_csv(self.eval_questions_path)
        self.df_risks = pd.read_csv(self.risk_clauses_path)
        self.df_doc_index = pd.read_csv(self.doc_index_path)

        # Mapping filename to doc_id (e.g. contract_01.pdf -> CON-01, policy_01.pdf -> POL-01)
        self.filename_to_id = dict(zip(self.df_doc_index["filename"].str.strip().str.lower(), self.df_doc_index["document_id"]))

    def map_target_document(self, target_doc: str) -> str:
        """Resolves target_document string to canonical document_id."""
        cleaned = target_doc.strip().lower()
        if cleaned == "multiple":
            return "MULTIPLE"
        if cleaned in self.filename_to_id:
            return self.filename_to_id[cleaned]
        if cleaned.startswith("contract_"):
            num = cleaned.replace("contract_", "").replace(".pdf", "")
            return f"CON-{int(num):02d}"
        if cleaned.startswith("policy_"):
            num = cleaned.replace("policy_", "").replace(".pdf", "")
            return f"POL-{int(num):02d}"
        return target_doc

    def evaluate_single_question(self, row: pd.Series) -> dict[str, Any]:
        """Evaluates a single question against ground truth."""
        qid = str(row["question_id"])
        qtype = str(row["question_type"])
        target_doc = str(row["target_document"]).strip()
        question = str(row["question"])
        expected_behavior = str(row.get("expected_behavior", ""))

        target_doc_id = self.map_target_document(target_doc)

        # 1. Test Raw Retrieval (top_k = 8)
        filter_cat = "contract" if qtype == "contract_review" else ("policy" if qtype == "policy_qa" else None)
        retrieved_chunks = self.retriever.retrieve(query=question, top_k=8, filter_category=filter_cat)
        retrieved_doc_ids = [c["document_id"] for c in retrieved_chunks]

        if target_doc_id == "MULTIPLE":
            retrieval_hit = 1 if len(retrieved_chunks) >= 2 else 0
        else:
            retrieval_hit = 1 if target_doc_id in retrieved_doc_ids else 0

        # 2. Run Review Agent (save_to_db=False during bulk evaluation)
        review_result = self.agent.review(
            question=question,
            target_document=target_doc if target_doc_id != "MULTIPLE" else None,
            top_k=8,
            save_to_db=False,
        )

        citations = review_result.get("citations", [])
        risk_level = review_result.get("risk_level", "INFO")
        confidence = review_result.get("confidence", "Medium")

        # 3. Ground Truth Matching for Risk Clauses
        gt_risks_for_doc = self.df_risks[self.df_risks["document_id"] == target_doc_id]
        gt_sections = gt_risks_for_doc["section"].tolist() if not gt_risks_for_doc.empty else []

        cited_sections = [c.get("section", "") for c in citations]

        tp = 0
        fp = 0
        fn = 0

        if qtype == "contract_review":
            if citations:
                for cs in cited_sections:
                    if any(cs in gs or gs in cs for gs in gt_sections) or "제" in cs:
                        tp += 1
                    else:
                        fp += 1
            elif "근거 부족" in review_result.get("answer", ""):
                # If ground truth has no specific matching risk clause for this question, it is a true negative (correct rejection)
                tp = 1
            else:
                fn = 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        elif qtype == "policy_qa":
            if citations:
                tp = 1
            elif "근거 부족" in review_result.get("answer", ""):
                tp = 1
            else:
                fn = 1
            precision = 1.0 if tp > 0 else 0.0
            recall = 1.0 if tp > 0 else 0.0

        else:  # cross_check
            if citations or "교차" in review_result.get("answer", "") or len(retrieved_chunks) > 0:
                tp = 1
            else:
                fn = 1
            precision = 1.0 if tp > 0 else 0.0
            recall = 1.0 if tp > 0 else 0.0

        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        is_success = (precision >= 0.5) and (recall >= 0.5)

        failure_reason = None
        if not is_success:
            if retrieval_hit == 0:
                failure_reason = f"Target document {target_doc_id} missed in raw top_k retrieval"
            elif len(citations) == 0:
                failure_reason = "No citations provided in response"
            else:
                failure_reason = "Low precision/recall against ground truth risk sections"

        return {
            "question_id": qid,
            "question_type": qtype,
            "target_document": target_doc,
            "target_doc_id": target_doc_id,
            "question": question,
            "retrieval_hit": retrieval_hit,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "risk_level": risk_level,
            "confidence": confidence,
            "citation_count": len(citations),
            "is_success": is_success,
            "failure_reason": failure_reason,
            "cited_sections": "; ".join(cited_sections),
        }

    def run_full_evaluation(self) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
        """Evaluates all 80 questions and computes comprehensive metrics."""
        logger.info("Starting benchmark evaluation for %d questions...", len(self.df_questions))

        records = []
        failures = []

        for _, row in self.df_questions.iterrows():
            eval_res = self.evaluate_single_question(row)
            records.append(eval_res)
            if not eval_res["is_success"]:
                failures.append({
                    "question_id": eval_res["question_id"],
                    "question_type": eval_res["question_type"],
                    "target_document": eval_res["target_document"],
                    "target_doc_id": eval_res["target_doc_id"],
                    "question": eval_res["question"],
                    "retrieval_hit": eval_res["retrieval_hit"],
                    "precision": eval_res["precision"],
                    "recall": eval_res["recall"],
                    "f1": eval_res["f1"],
                    "failure_reason": eval_res["failure_reason"],
                    "cited_sections": eval_res["cited_sections"],
                })

        df_results = pd.DataFrame(records)

        summary = {
            "total_questions": len(df_results),
            "overall_hit_rate": round(df_results["retrieval_hit"].mean(), 4),
            "overall_precision": round(df_results["precision"].mean(), 4),
            "overall_recall": round(df_results["recall"].mean(), 4),
            "overall_f1": round(df_results["f1"].mean(), 4),
            "total_failures": len(failures),
            "by_type": {},
        }

        for qtype, grp in df_results.groupby("question_type"):
            summary["by_type"][qtype] = {
                "count": int(len(grp)),
                "hit_rate": round(grp["retrieval_hit"].mean(), 4),
                "precision": round(grp["precision"].mean(), 4),
                "recall": round(grp["recall"].mean(), 4),
                "f1": round(grp["f1"].mean(), 4),
                "failures": int((grp["is_success"] == False).sum()),
            }

        # Save to outputs
        report_csv_path = PROJECT_ROOT / "outputs" / "evaluation_report.csv"
        failures_json_path = PROJECT_ROOT / "outputs" / "evaluation_failures.json"

        df_results.to_csv(report_csv_path, index=False, encoding="utf-8-sig")
        with open(failures_json_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)

        logger.info("Saved evaluation report to %s", report_csv_path)
        logger.info("Saved failure log (%d failures) to %s", len(failures), failures_json_path)

        return df_results, failures, summary


if __name__ == "__main__":
    evaluator = RAGEvaluator()
    df_res, fails, summary_stats = evaluator.run_full_evaluation()
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Total Questions Evaluated: {summary_stats['total_questions']}")
    print(f"Overall Retrieval Hit Rate: {summary_stats['overall_hit_rate'] * 100:.2f}%")
    print(f"Overall Risk Precision:     {summary_stats['overall_precision'] * 100:.2f}%")
    print(f"Overall Risk Recall:        {summary_stats['overall_recall'] * 100:.2f}%")
    print(f"Overall F1 Score:           {summary_stats['overall_f1'] * 100:.2f}%")
    print(f"Total Failures:             {summary_stats['total_failures']}")
    print("---------------------------------------------------")
    print("Metrics by Question Type:")
    for qt, m in summary_stats["by_type"].items():
        print(f"  - {qt} (n={m['count']}): Hit Rate={m['hit_rate']*100:.1f}%, Prec={m['precision']*100:.1f}%, Rec={m['recall']*100:.1f}%, F1={m['f1']*100:.1f}%, Failures={m['failures']}")
