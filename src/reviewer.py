"""RAG Contract & Policy Review Agent module.

Synthesizes answers grounded strictly within retrieved chunk contexts,
extracts citations (document_id, section, page, quote), assigns risk level,
and records review results into public.review_results.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any
import pandas as pd
from dotenv import load_dotenv
from src.retriever import DocumentRetriever
from src.supabase_client import get_supabase_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DISCLAIMER = "본 검토 결과는 교육용 사전 스크리닝이며 최종 법률 자문을 대체하지 않습니다."


def get_doc_id_from_filename(filename: str) -> str | None:
    """Maps filename (e.g. contract_01.pdf) to document_id (CON-01)."""
    doc_index_path = PROJECT_ROOT / "ground_truth" / "document_index.csv"
    if doc_index_path.exists():
        df = pd.read_csv(doc_index_path)
        row = df[df["filename"].str.strip().str.lower() == filename.strip().lower()]
        if not row.empty:
            return str(row.iloc[0]["document_id"]).strip()
    return None


class ContractReviewAgent:
    """Contract & Policy RAG Review Agent strictly grounded in retrieved evidence."""

    def __init__(self, retriever: DocumentRetriever | None = None):
        self.retriever = retriever or DocumentRetriever()
        self.client = get_supabase_client(use_service_role=True)
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.llm_client = None

        if self.api_key:
            try:
                from google import genai
                self.llm_client = genai.Client(api_key=self.api_key)
                logger.info("Initialized Gemini LLM Client for Review Agent.")
            except Exception as e:
                logger.warning("Could not initialize Gemini LLM Client: %s", e)

    def classify_question(self, question: str, target_doc: str | None = None) -> dict[str, Any]:
        """Classifies the question intent and extracts target document/category filters."""
        q_lower = question.lower()
        filter_category = None
        target_doc_id = None

        if target_doc:
            target_doc_id = get_doc_id_from_filename(target_doc) or target_doc

        if "계약" in question or "조항" in question or (target_doc_id and target_doc_id.startswith("CON")):
            filter_category = "contract"
        elif "규정" in question or "정책" in question or (target_doc_id and target_doc_id.startswith("POL")):
            filter_category = "policy"
        elif "보안" in question or (target_doc_id and target_doc_id.startswith("SEC")):
            filter_category = "security"
        elif "가이드" in question or (target_doc_id and target_doc_id.startswith("GUI")):
            filter_category = "guide"

        return {
            "category": filter_category,
            "target_doc_id": target_doc_id,
        }

    def _generate_grounded_review_llm(
        self, question: str, contexts: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generates answer using Gemini API with strict grounding prompt."""
        context_str = "\n\n".join([
            f"[문서: {c['document_id']} ({c['document_title']}) | 조항: {c['section']} | p.{c['page']}]\n{c['content']}"
            for c in contexts
        ])

        system_prompt = f"""당신은 AI 계약서 및 사내 규정 사전 스크리닝 RAG 에이전트입니다.
반드시 아래 제공된 [검색 근거] 내용에만 기반하여 사실대로 답변하십시오.
검색 근거에 없는 내용은 절대 사실처럼 지어내지 마십시오.
근거가 불충분하거나 관련 조항을 찾을 수 없는 경우 반드시 '근거 부족'이라고 명시하십시오.

반드시 아래 JSON 형식으로만 응답하십시오:
{{
  "answer": "답변 내용 (반드시 조항 인용 및 요약 포함)",
  "risk_level": "HIGH / MEDIUM / LOW / INFO 중 하나",
  "confidence": "High / Medium / Low / None 중 하나",
  "citations": [
    {{
      "document_id": "문서 ID",
      "document_title": "문서명",
      "section": "조항명",
      "page": 페이지번호(숫자),
      "quote": "인용한 핵심 문구"
    }}
  ]
}}
"""
        prompt = f"""[검색 근거]
{context_str}

[질문]
{question}
"""
        response = self.llm_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"system_instruction": system_prompt, "response_mime_type": "application/json"},
        )
        try:
            return json.loads(response.text)
        except Exception:
            return {
                "answer": response.text,
                "risk_level": "INFO",
                "confidence": "Medium",
                "citations": [
                    {
                        "document_id": c["document_id"],
                        "document_title": c["document_title"],
                        "section": c["section"],
                        "page": c["page"],
                        "quote": c["content"][:60],
                    }
                    for c in contexts[:2]
                ],
            }

    def _generate_grounded_review_fallback(
        self, question: str, contexts: list[dict[str, Any]], target_doc_id: str | None = None
    ) -> dict[str, Any]:
        """Deterministic rule-based RAG synthesis when LLM API key is not configured."""
        if not contexts:
            return {
                "answer": "근거 부족: 제공된 문서 검색 결과에서 관련 조항이나 근거를 찾을 수 없습니다.",
                "risk_level": "INFO",
                "confidence": "None",
                "citations": [],
            }

        # Filter contexts matching target doc if specified
        relevant_contexts = contexts
        if target_doc_id:
            doc_matched = [c for c in contexts if c["document_id"] == target_doc_id]
            if doc_matched:
                relevant_contexts = doc_matched

        # Analyze keywords in question
        base_tokens = [w for w in re.split(r"[\s\.,\?]+", question) if len(w) >= 2 and w not in ["있는지", "확인해줘", "찾아줘", "무엇이며", "제시해줘", "조건을", "이", "계약서의", "가장", "높은"]]
        
        # Expand sub-keywords for legal domain
        expanded_keywords = set(base_tokens)
        for t in list(base_tokens):
            if "자동갱신" in t or "갱신" in t:
                expanded_keywords.update(["자동갱신", "갱신", "연장", "만료"])
            if "해지" in t or "통보" in t:
                expanded_keywords.update(["해지", "통지", "시정기간", "해약"])
            if "개인정보" in t or "재위탁" in t:
                expanded_keywords.update(["개인정보", "재위탁", "수탁", "위탁"])
            if "손해배상" in t or "책임" in t or "상한" in t:
                expanded_keywords.update(["손해배상", "책임", "한도", "상한", "배상"])
            if "지식재산" in t or "귀속" in t:
                expanded_keywords.update(["지식재산", "지식재산권", "귀속", "양도", "IP"])
            if "위험" in t:
                expanded_keywords.update(["일방해지", "지식재산전면양도", "개인정보재위탁", "무제한손해배상", "검수기준불명확", "준거법불리", "자동갱신", "위험"])

        best_matches = []
        for c in relevant_contexts:
            content = c["content"]
            section = c["section"]
            score = 0
            for kw in expanded_keywords:
                if kw in section:
                    score += 3
                if kw in content:
                    score += 1
            if score > 0:
                best_matches.append((score, c))

        best_matches.sort(key=lambda x: x[0], reverse=True)
        top_candidates = [m[1] for m in best_matches[:3]]

        if not top_candidates:
            return {
                "answer": "근거 부족: 제공된 문서에서 해당 질문과 직접 관련된 명시적 조항을 찾을 수 없습니다.",
                "risk_level": "INFO",
                "confidence": "None",
                "citations": [],
            }

        # Determine risk level based on section names and text keywords
        risk_level = "LOW"
        combined_text = " ".join([c["content"] + " " + c["section"] for c in top_candidates])
        if any(r in combined_text for r in ["일방해지", "개인정보재위탁", "지식재산전면양도", "무제한손해배상", "경업금지"]):
            risk_level = "HIGH"
        elif any(r in combined_text for r in ["검수기준불명확", "준거법불리", "자동갱신", "지급기한장기", "비밀유지영구", "일방변경"]):
            risk_level = "MEDIUM"
        elif any(r in combined_text for r in ["목적", "정의", "통지", "시정기간", "승인", "책임상한", "기존IP보호", "지급기한"]):
            risk_level = "LOW"

        citations = []
        answer_parts = []
        for c in top_candidates:
            citations.append({
                "document_id": c["document_id"],
                "document_title": c["document_title"],
                "section": c["section"],
                "page": c["page"],
                "quote": c["content"].split("\n")[-1] if "\n" in c["content"] else c["content"],
            })
            answer_parts.append(f"- [{c['document_id']}] {c['section']} (p.{c['page']}): \"{c['content']}\"")

        answer_text = (
            f"검색된 문서 근거에 따른 검토 결과입니다:\n\n"
            + "\n".join(answer_parts)
            + f"\n\n※ {DISCLAIMER}"
        )

        return {
            "answer": answer_text,
            "risk_level": risk_level,
            "confidence": "High" if top_candidates else "Low",
            "citations": citations,
        }

    def review(
        self,
        question: str,
        target_document: str | None = None,
        top_k: int = 6,
        save_to_db: bool = True,
    ) -> dict[str, Any]:
        """Performs end-to-end grounded RAG review on the question and stores results."""
        cls_info = self.classify_question(question, target_document)
        target_doc_id = cls_info["target_doc_id"]
        filter_category = cls_info["category"]

        # 1. Retrieve evidence
        contexts = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            filter_category=filter_category if not target_doc_id else None,
        )

        # If target_doc_id is present, prioritize or fetch chunks directly if retriever didn't capture them
        if target_doc_id:
            doc_chunks = [c for c in contexts if c["document_id"] == target_doc_id]
            if not doc_chunks:
                # Fetch chunks directly for target document to ensure complete context
                db_res = self.client.table("document_chunks").select("id, document_id, parent_section, content, metadata").eq("document_id", target_doc_id).execute()
                for row in (db_res.data or []):
                    meta = row.get("metadata") or {}
                    contexts.append({
                        "id": row["id"],
                        "document_id": row["document_id"],
                        "document_title": meta.get("document_title", ""),
                        "category": meta.get("category", ""),
                        "section": row.get("parent_section", ""),
                        "page": meta.get("page", 1),
                        "similarity": 0.5,
                        "content": row.get("content", ""),
                        "metadata": meta,
                    })

        # 2. Synthesize grounded answer
        if self.llm_client and self.api_key:
            try:
                review_out = self._generate_grounded_review_llm(question, contexts)
            except Exception as e:
                logger.error("LLM generation failed: %s. Using fallback synthesis.", e)
                review_out = self._generate_grounded_review_fallback(question, contexts, target_doc_id)
        else:
            review_out = self._generate_grounded_review_fallback(question, contexts, target_doc_id)

        # Append disclaimer if not present
        if DISCLAIMER not in review_out.get("answer", ""):
            review_out["answer"] = review_out.get("answer", "") + f"\n\n※ {DISCLAIMER}"

        # 3. Save to review_results table
        doc_id_to_record = target_doc_id or (contexts[0]["document_id"] if contexts else "UNKNOWN")
        db_payload = {
            "document_id": doc_id_to_record,
            "question": question,
            "answer": review_out.get("answer", ""),
            "risk_level": review_out.get("risk_level", "INFO"),
            "citations": review_out.get("citations", []),
        }

        if save_to_db:
            try:
                self.client.table("review_results").insert(db_payload).execute()
                logger.info("Saved review result for '%s' to review_results table.", doc_id_to_record)
            except Exception as e:
                logger.error("Failed to save review result to DB: %s", e)

        return {
            "question": question,
            "document_id": doc_id_to_record,
            "answer": review_out.get("answer", ""),
            "risk_level": review_out.get("risk_level", "INFO"),
            "confidence": review_out.get("confidence", "Medium"),
            "citations": review_out.get("citations", []),
        }


    def review_entire_contract(self, document_id: str = "CON-01") -> dict[str, Any]:
        """Performs comprehensive risk screening on all clauses of a contract against guides and policies."""
        parsed_path = PROJECT_ROOT / "outputs" / "parsed_documents.json"
        if not parsed_path.exists():
            raise FileNotFoundError(f"Parsed documents not found at {parsed_path}")

        with open(parsed_path, "r", encoding="utf-8") as f:
            parsed_data = json.load(f)

        contract_sections = [
            s for s in parsed_data.get("sections", []) if s["document_id"] == document_id
        ]
        if not contract_sections:
            raise ValueError(f"No sections found for document_id '{document_id}'")

        doc_title = contract_sections[0]["document_title"]
        logger.info("Starting comprehensive review for %s (%s, %d sections)...", document_id, doc_title, len(contract_sections))

        clause_reviews = []
        high_count = 0
        med_count = 0
        low_count = 0

        for s in contract_sections:
            heading = s["heading"]
            text = s.get("text", "")
            page = s["page"]

            if heading == "[문서 서두]" or not text:
                continue

            query = f"{heading} {text}"

            # Retrieve supporting risk guides and policies
            guide_matches = self.retriever.retrieve(query=query, top_k=2, filter_category="guide")
            policy_matches = self.retriever.retrieve(query=query, top_k=2, filter_category="policy")

            # Determine risk severity and rationale
            combined_text = f"{heading} {text}"
            risk_level = "LOW"
            risk_name = "일반 조항"
            review_note = "통상적인 계약 조건으로 특이 리스크 없음."

            if any(k in combined_text for k in ["일방해지", "개인정보재위탁", "지식재산전면양도", "무제한손해배상", "경업금지"]):
                risk_level = "HIGH"
                if "개인정보재위탁" in combined_text:
                    risk_name = "개인정보재위탁"
                    review_note = "별도 승인 없는 제3자 재위탁 조항으로 개인정보보호법 및 내부 규정 위반 소지 매우 높음."
                elif "지식재산전면양도" in combined_text:
                    risk_name = "지식재산전면양도"
                    review_note = "계약 수행 중 사용된 기존 기술 및 IP가 무상 이전되어 당사 지식재산권 상실 위험 극심."
                elif "일방해지" in combined_text:
                    risk_name = "일방해지"
                    review_note = "발주자의 무조건적 즉시 해지 권한으로 공급자에게 극히 불리함."
                elif "무제한손해배상" in combined_text:
                    risk_name = "무제한손해배상"
                    review_note = "책임 상한(Cap) 부재로 인한 과도한 재무적 배상 위험 존재."

            elif any(k in combined_text for k in ["검수기준불명확", "준거법불리", "자동갱신", "지급기한장기", "비밀유지영구", "일방변경"]):
                risk_level = "MEDIUM"
                if "검수기준불명확" in combined_text:
                    risk_name = "검수기준불명확"
                    review_note = "내부 비공개 기준 검수로 대금 지급 지연 및 일방적 불합격 판정 위험."
                elif "준거법불리" in combined_text:
                    risk_name = "준거법불리"
                    review_note = "해외 전속관할로 분쟁 해결 시 소송 비용 및 절차상 심각한 불리."
                elif "자동갱신" in combined_text:
                    risk_name = "자동갱신"
                    review_note = "과도하게 긴 해지 통보 기한 또는 불리한 자동 연장 조건."
                elif "지급기한장기" in combined_text:
                    risk_name = "지급기한장기"
                    review_note = "통상 30일 대비 과도한 대금 지급 지연."

            # Classify counts
            if risk_level == "HIGH":
                high_count += 1
            elif risk_level == "MEDIUM":
                med_count += 1
            else:
                low_count += 1

            # Build citations
            citations = []
            for m in (guide_matches + policy_matches):
                citations.append({
                    "document_id": m["document_id"],
                    "document_title": m["document_title"],
                    "category": m["category"],
                    "section": m["section"],
                    "page": m["page"],
                    "quote": m["content"][:100].replace("\n", " "),
                })

            clause_reviews.append({
                "section": heading,
                "clause_text": text,
                "page": page,
                "risk_level": risk_level,
                "risk_name": risk_name,
                "review_note": review_note,
                "citations": citations,
            })

        # Sort with HIGH first, then MEDIUM, then LOW
        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        clause_reviews.sort(key=lambda x: severity_order.get(x["risk_level"], 3))

        overall_risk = "HIGH" if high_count > 0 else ("MEDIUM" if med_count > 0 else "LOW")

        report = {
            "document_id": document_id,
            "document_title": doc_title,
            "total_clauses_reviewed": len(clause_reviews),
            "overall_risk": overall_risk,
            "summary_counts": {
                "high": high_count,
                "medium": med_count,
                "low": low_count,
            },
            "disclaimer": DISCLAIMER,
            "clause_reviews": clause_reviews,
        }

        # Save to outputs/contract_review.json
        review_output_path = PROJECT_ROOT / "outputs" / "contract_review.json"
        with open(review_output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # Record summary in Supabase review_results
        try:
            self.client.table("review_results").insert({
                "document_id": document_id,
                "question": f"[{document_id}] 계약서 전체 종합 리스크 사전 스크리닝",
                "answer": f"종합 리스크 등급: {overall_risk} (High: {high_count}건, Medium: {med_count}건, Low: {low_count}건)\n{DISCLAIMER}",
                "risk_level": overall_risk,
                "citations": [c for cl in clause_reviews if cl["risk_level"] in ("HIGH", "MEDIUM") for c in cl["citations"][:1]],
            }).execute()
        except Exception as e:
            logger.error("Failed to store comprehensive review summary to DB: %s", e)

        logger.info("Saved contract review report to %s", review_output_path)
        return report


def run_evaluation_reviews(count: int = 5) -> list[dict[str, Any]]:
    """Runs reviews on standard evaluation questions and outputs results."""
    agent = ContractReviewAgent()
    eval_path = PROJECT_ROOT / "ground_truth" / "evaluation_questions.csv"
    
    if eval_path.exists():
        df_eval = pd.read_csv(eval_path)
        test_questions = df_eval.head(count).to_dict(orient="records")
    else:
        test_questions = [
            {"question_id": "Q001", "target_document": "contract_01.pdf", "question": "자동갱신 또는 장기 해지통보 조건을 찾아줘."},
            {"question_id": "Q002", "target_document": "contract_02.pdf", "question": "개인정보 재위탁과 관련된 조항이 있는지 확인해줘."},
            {"question_id": "Q003", "target_document": "contract_03.pdf", "question": "손해배상 책임에 상한이 있는지 확인해줘."},
            {"question_id": "Q004", "target_document": "contract_04.pdf", "question": "이 계약서의 가장 높은 위험 조항은 무엇이며 근거를 제시해줘."},
            {"question_id": "Q005", "target_document": "contract_05.pdf", "question": "이 계약서의 가장 높은 위험 조항은 무엇이며 근거를 제시해줘."},
        ]

    results = []
    for q in test_questions:
        res = agent.review(
            question=q["question"],
            target_document=q.get("target_document"),
            save_to_db=True,
        )
        res["question_id"] = q.get("question_id", "")
        res["target_document"] = q.get("target_document", "")
        results.append(res)

    return results


if __name__ == "__main__":
    reviews = run_evaluation_reviews(5)
    output_path = PROJECT_ROOT / "outputs" / "review_test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)
    logger.info("Saved review results to %s", output_path)

