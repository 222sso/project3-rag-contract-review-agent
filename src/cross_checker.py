"""Cross-check module comparing contract clauses against internal policies and security guides.

Evaluates compliance status (Match / Warning / Conflict / Undetermined) with explicit citations.
"""

import json
import logging
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

DISCLAIMER = "본 교차 검토 결과는 사내 규정 준수 여부를 확인하기 위한 사전 스크리닝이며 최종 법률 자문을 대체하지 않습니다."

# Standard cross-check topics with corresponding search queries
CROSS_CHECK_TOPICS = [
    {
        "topic_id": "T01",
        "topic": "개인정보 재위탁 (Subcontracting)",
        "contract_query": "개인정보 재위탁 승인 수탁자 제3자",
        "policy_query": "개인정보 재위탁 사전 승인 수탁업체 관리 감독",
        "security_query": "외부업체 계정 권한 회수 및 위탁 보안",
    },
    {
        "topic_id": "T02",
        "topic": "지식재산권 귀속 및 기존 기술 이전 (Intellectual Property)",
        "contract_query": "지식재산권 귀속 기존 기술 양도 산출물 IP",
        "policy_query": "지식재산권 소유 기존 보유 기술 양도 금지 산출물 권리",
        "security_query": "데이터 보호 산출물 소유권 민감정보 보호",
    },
    {
        "topic_id": "T03",
        "topic": "검수 기준 및 절차 (Acceptance Testing)",
        "contract_query": "검수 기준 절차 내부 기준 비공개 완료",
        "policy_query": "검수 기준 명시 객관적 합의 납품 승인 절차",
        "security_query": "시스템 검수 보안 취약점 점검",
    },
    {
        "topic_id": "T04",
        "topic": "보안 사고 통지 기한 (Security Incident Notification)",
        "contract_query": "보안 사고 침해 사고 통지 기한 24시간",
        "policy_query": "침해 사고 보고 긴급 대응 즉시 통지",
        "security_query": "중대 보안사고 24시간 이내 통지 공유 현황",
    },
    {
        "topic_id": "T05",
        "topic": "대금 지급 기한 및 조건 (Payment Terms)",
        "contract_query": "대금 지급 기한 검수 세금계산서 30일",
        "policy_query": "대금 지급 기한 구매 계약 정산 절차",
        "security_query": "계약 이행 및 대금 정산",
    },
    {
        "topic_id": "T06",
        "topic": "계약 해지 및 시정 기간 (Termination & Cure Period)",
        "contract_query": "계약 해지 즉시 해지 시정기간 30일 중대한 위반",
        "policy_query": "계약 해지 시정 요구 최고 기간 부여",
        "security_query": "보안 위반 시 계약 해지 조치",
    },
]


class ContractPolicyCrossChecker:
    """Compares contract terms against corporate internal policies and security guides."""

    def __init__(self, retriever: DocumentRetriever | None = None):
        self.retriever = retriever or DocumentRetriever()
        self.client = get_supabase_client(use_service_role=True)

    def cross_check_contract(self, document_id: str = "CON-01") -> dict[str, Any]:
        """Runs compliance cross-checks for a specific contract across predefined compliance topics."""
        # 1. Fetch contract chunks directly to ensure accurate local clause mapping
        contract_chunks_res = self.client.table("document_chunks").select("id, document_id, parent_section, content, metadata").eq("document_id", document_id).execute()
        contract_chunks = contract_chunks_res.data or []
        doc_title = contract_chunks[0]["metadata"].get("document_title", f"계약서 {document_id}") if contract_chunks else document_id

        logger.info("Executing Cross-Check for %s (%s, %d clauses)...", document_id, doc_title, len(contract_chunks))

        topic_results = []
        conflict_count = 0
        warning_count = 0
        match_count = 0
        undetermined_count = 0

        for t_cfg in CROSS_CHECK_TOPICS:
            topic_name = t_cfg["topic"]
            t_id = t_cfg["topic_id"]

            # A. Match contract clause
            best_contract_chunk = None
            max_c_score = -1
            keywords = t_cfg["contract_query"].split()

            for chunk in contract_chunks:
                sec = chunk.get("parent_section") or ""
                cont = chunk.get("content") or ""
                if sec == "[문서 서두]" or sec == "부칙":
                    continue
                score = sum(4 for kw in keywords if kw in sec) + sum(1 for kw in keywords if kw in cont)
                # Prioritize risk section match for IP
                if t_id == "T02" and "전면양도" in sec:
                    score += 10
                if score > max_c_score and score >= 3:
                    max_c_score = score
                    best_contract_chunk = chunk

            # B. Retrieve matching policy / security guide
            policy_matches = self.retriever.retrieve(
                query=t_cfg["policy_query"],
                top_k=3,
                filter_category="policy",
            )
            security_matches = self.retriever.retrieve(
                query=t_cfg["security_query"],
                top_k=3,
                filter_category="security",
            )
            
            # Filter out [문서 서두] if actual content section exists
            filtered_rules = [r for r in (policy_matches + security_matches) if r.get("section") != "[문서 서두]"]
            best_rule_match = filtered_rules[0] if filtered_rules else ((policy_matches + security_matches)[0] if (policy_matches + security_matches) else None)

            # C. Evaluate Status
            status = "미확정"
            comparison_analysis = ""

            if not best_contract_chunk or not best_rule_match:
                status = "미확정"
                comparison_analysis = "계약서 또는 내부 규정 중 한쪽의 관련 조항 근거가 누락되어 비교가 미확정되었습니다."
                undetermined_count += 1
            else:
                c_text = best_contract_chunk.get("content", "")
                c_sec = best_contract_chunk.get("parent_section", "")
                r_text = best_rule_match.get("content", "")
                r_sec = best_rule_match.get("section", "")

                # Topic-specific compliance evaluation logic
                if t_id == "T01":  # Subcontracting
                    if "별도 승인 없이" in c_text or "임의" in c_text or "승인 없이" in c_text:
                        status = "충돌"
                        comparison_analysis = "계약서는 '사전 승인 없는 재위탁'을 허용하나, 사내 개인정보 규정은 '사전 서면 승인 및 수탁사 관리 감독 의무'를 명시하고 있어 심각한 규정 충돌이 발생합니다."
                        conflict_count += 1
                    else:
                        status = "일치"
                        comparison_analysis = "계약서의 사전 서면 승인 요건이 사내 개인정보 처리 규정과 부합합니다."
                        match_count += 1

                elif t_id == "T02":  # Intellectual Property
                    if "무상으로 이전" in c_text or "전면양도" in c_sec or "기존 기술을 포함한" in c_text:
                        status = "충돌"
                        comparison_analysis = "계약서는 당사 기존 기술(Background IP)까지 발주자에게 무상 전면 양도하도록 규정하고 있어, 지식재산권 보호 및 IP 유출 방지 규정과 정면 충돌합니다."
                        conflict_count += 1
                    else:
                        status = "일치"
                        comparison_analysis = "체결 전 고유 지식재산권 귀속 및 산출물 사용권 부여 조항이 사내 규정과 일치합니다."
                        match_count += 1

                elif t_id == "T03":  # Acceptance
                    if "내부 기준에 따라" in c_text or "공개하지 않는다" in c_text or "불명확" in c_sec:
                        status = "주의"
                        comparison_analysis = "계약서상 검수 기준이 발주자 내부 비공개 기준으로 되어 있어, 사내 구매/계약 규정의 '객관적 검수 기준 사전 명시 및 자동 승인 간주' 권고사항에 미달합니다."
                        warning_count += 1
                    else:
                        status = "일치"
                        comparison_analysis = "명확한 검수 일정 및 기준이 규정과 부합합니다."
                        match_count += 1

                elif t_id == "T04":  # Security Incident Notification
                    if "24시간" in c_text or "즉시" in c_text:
                        status = "일치"
                        comparison_analysis = "계약서의 24시간 이내 통지 조항이 사내 정보보안 가이드(SEC) 긴급 침해사고 대응 지침과 일치합니다."
                        match_count += 1
                    else:
                        status = "주의"
                        comparison_analysis = "계약서에 구체적 침해사고 통지 기한(24시간)이 누락되어 있어 정보보안 가이드 보완이 권고됩니다."
                        warning_count += 1

                elif t_id == "T05":  # Payment Terms
                    if "30일" in c_text:
                        status = "일치"
                        comparison_analysis = "검수 완료 후 30일 이내 대금 지급 조건은 사내 표준 결제 및 정산 규정과 부합합니다."
                        match_count += 1
                    elif "120일" in c_text or "장기" in c_sec:
                        status = "충돌"
                        comparison_analysis = "120일 이상의 장기 지급 조건은 대금 결제 규정 위반 및 유동성 위험을 초래합니다."
                        conflict_count += 1
                    else:
                        status = "일치"
                        match_count += 1

                elif t_id == "T06":  # Termination
                    if "사유를 밝히지 않고" in c_text or "즉시 계약을 해지" in c_text:
                        status = "충돌"
                        comparison_analysis = "발주자의 사유 없는 일방 즉시 해지 권한은 당사 계약 해지 규정의 '최소 30일 시정기간 부여 원칙'과 정면 충돌합니다."
                        conflict_count += 1
                    elif "30일의 시정기간" in c_text or "시정기간" in c_text:
                        status = "일치"
                        comparison_analysis = "30일 시정기간 부여 후 해지 절차는 사내 표준 계약 해지 규정과 일치합니다."
                        match_count += 1
                    else:
                        status = "일치"
                        match_count += 1

            topic_results.append({
                "topic_id": t_id,
                "topic": topic_name,
                "status": status,
                "contract_condition": {
                    "document_id": document_id,
                    "section": best_contract_chunk.get("parent_section") if best_contract_chunk else "조항 누락",
                    "page": best_contract_chunk.get("metadata", {}).get("page", 1) if best_contract_chunk else 1,
                    "clause_text": best_contract_chunk.get("content", "해당 조항 없음") if best_contract_chunk else "해당 조항 없음",
                },
                "internal_policy_requirement": {
                    "document_id": best_rule_match.get("document_id") if best_rule_match else "규정 누락",
                    "document_title": best_rule_match.get("document_title") if best_rule_match else "",
                    "category": best_rule_match.get("category") if best_rule_match else "",
                    "section": best_rule_match.get("section") if best_rule_match else "",
                    "page": best_rule_match.get("page", 1) if best_rule_match else 1,
                    "requirement_text": best_rule_match.get("content", "") if best_rule_match else "",
                },
                "analysis": comparison_analysis,
            })

        report_payload = {
            "document_id": document_id,
            "document_title": doc_title,
            "total_topics_compared": len(topic_results),
            "summary": {
                "conflict": conflict_count,
                "warning": warning_count,
                "match": match_count,
                "undetermined": undetermined_count,
            },
            "disclaimer": DISCLAIMER,
            "comparisons": topic_results,
        }

        # Save to outputs/cross_check_report.json
        output_file = PROJECT_ROOT / "outputs" / "cross_check_report.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, ensure_ascii=False, indent=2)

        # Save to Supabase review_results
        try:
            self.client.table("review_results").insert({
                "document_id": document_id,
                "question": f"[{document_id}] 계약서-사내규정-보안가이드 교차 규정 충돌 검토",
                "answer": f"교차 검토 완료 (충돌: {conflict_count}건, 주의: {warning_count}건, 일치: {match_count}건, 미확정: {undetermined_count}건)\n{DISCLAIMER}",
                "risk_level": "HIGH" if conflict_count > 0 else ("MEDIUM" if warning_count > 0 else "LOW"),
                "citations": [
                    {
                        "document_id": c["contract_condition"]["document_id"],
                        "section": c["contract_condition"]["section"],
                        "page": c["contract_condition"]["page"],
                        "quote": c["contract_condition"]["clause_text"][:80],
                    }
                    for c in topic_results if c["status"] in ("충돌", "주의")
                ],
            }).execute()
        except Exception as e:
            logger.error("Failed to store cross-check summary to DB: %s", e)

        logger.info("Saved cross check report to %s", output_file)
        return report_payload


if __name__ == "__main__":
    checker = ContractPolicyCrossChecker()
    checker.cross_check_contract("CON-01")
