"""Streamlit Interactive Web Dashboard for Project 3: Contract Review RAG Agent.

Provides:
1. Interactive Contract Risk Screening with Citations & Ground Truth comparison
2. Policy & Security Compliance Cross-Checking
3. Semantic Vector Search & QA with Citation Verification
4. Quantitative RAG Evaluation Benchmark Dashboard (80 Questions)
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load root environment
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from src.cross_checker import ContractPolicyCrossChecker
from src.retriever import DocumentRetriever
from src.reviewer import ContractReviewAgent

st.set_page_config(
    page_title="AI Contract & Policy RAG Reviewer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .badge-high { background-color: #FEE2E2; color: #991B1B; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-medium { background-color: #FEF3C7; color: #92400E; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-low { background-color: #D1FAE5; color: #065F46; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .disclaimer-box { background-color: #F3F4F6; border-left: 4px solid #6B7280; padding: 10px 14px; font-size: 0.9rem; color: #374151; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_agent_and_retriever():
    retriever = DocumentRetriever()
    reviewer = ContractReviewAgent(retriever=retriever)
    cross_checker = ContractPolicyCrossChecker(retriever=retriever)
    return retriever, reviewer, cross_checker


@st.cache_data
def load_ground_truth():
    gt_docs = pd.read_csv(PROJECT_ROOT / "ground_truth" / "document_index.csv")
    gt_risks = pd.read_csv(PROJECT_ROOT / "ground_truth" / "risk_clauses.csv")
    gt_evals = pd.read_csv(PROJECT_ROOT / "outputs" / "evaluation_report.csv") if (PROJECT_ROOT / "outputs" / "evaluation_report.csv").exists() else pd.DataFrame()
    return gt_docs, gt_risks, gt_evals


retriever, reviewer, cross_checker = get_agent_and_retriever()
df_docs, df_risks, df_evals = load_ground_truth()

# Sidebar Navigation
st.sidebar.title("⚖️ Contract RAG Agent")
menu = st.sidebar.radio(
    "메뉴 선택",
    ["📋 계약서 종합 사전 검토", "🔄 규정/보안 교차 검증", "🔍 RAG 시맨틱 검색 & 질의", "📊 RAG 벤치마크 평가 대시보드"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**시스템 정보**")
st.sidebar.markdown("- **Vector DB**: Supabase pgvector (768-d)")
st.sidebar.markdown("- **HNSW Index**: m=16, ef_construction=64")
st.sidebar.markdown("- **Embedding**: Gemini / Deterministic Fallback")
st.sidebar.markdown("- **문서 인덱스**: 60개 문서 (577개 청크)")

# -------------------------------------------------------------
# TAB 1: 계약서 종합 사전 검토
# -------------------------------------------------------------
if menu == "📋 계약서 종합 사전 검토":
    st.markdown('<div class="main-header">📋 계약서 종합 위험조항 사전 스크리닝</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">계약서의 모든 조항을 분석하여 독소 조항 및 고위험 요소를 High/Medium/Low로 자동 분류합니다.</div>', unsafe_allow_html=True)

    contract_docs = df_docs[df_docs["category"] == "contract"].to_dict("records")
    doc_options = {f"{d['document_id']} - {d['document_title']} ({d['filename']})": d["document_id"] for d in contract_docs}
    
    selected_doc_label = st.selectbox("검토 대상 계약서 선택", list(doc_options.keys()))
    selected_doc_id = doc_options[selected_doc_label]

    if st.button("🚀 계약서 정밀 스크리닝 실행", type="primary"):
        with st.spinner(f"{selected_doc_id} 전체 조항 분석 및 리스크 가이드 매칭 중..."):
            review_res = reviewer.review_entire_contract(document_id=selected_doc_id)

        # Summary Metrics
        st.markdown("### 📊 위험도 진단 결과")
        col1, col2, col3, col4 = st.columns(4)
        counts = review_res.get("summary_counts", {})
        col1.metric("총 검토 조항", f"{review_res.get('total_clauses_reviewed', 0)}개")
        col2.metric("🔴 High Risk", f"{counts.get('high', 0)}건")
        col3.metric("🟡 Medium Risk", f"{counts.get('medium', 0)}건")
        col4.metric("🟢 Low Risk", f"{counts.get('low', 0)}건")

        # Clause Reviews
        st.markdown("### 📑 조항별 세부 분석 내역")
        for c in review_res.get("clause_reviews", []):
            risk_badge = (
                '<span class="badge-high">🔴 HIGH</span>' if c["risk_level"] == "HIGH"
                else ('<span class="badge-medium">🟡 MEDIUM</span>' if c["risk_level"] == "MEDIUM"
                else '<span class="badge-low">🟢 LOW</span>')
            )
            with st.expander(f"{c['section']} (p.{c['page']}) - {c['risk_name']} | {c['risk_level']}", expanded=(c["risk_level"] in ("HIGH", "MEDIUM"))):
                st.markdown(f"**위험 등급**: {risk_badge}", unsafe_allow_html=True)
                st.markdown(f"**조항 원문**:\n> {c['clause_text']}")
                st.markdown(f"**검토 의견 / 권고사항**:\n{c['review_note']}")
                
                if c.get("citations"):
                    st.markdown("**참조 근거 (Citations)**:")
                    for cit in c["citations"]:
                        st.markdown(f"- `[{cit.get('document_id')}]` {cit.get('document_title')} (p.{cit.get('page')}) : *\"{cit.get('quote')}\"*")

        st.markdown(f'<div class="disclaimer-box">⚠️ {review_res.get("disclaimer", "")}</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 2: 규정/보안 교차 검증
# -------------------------------------------------------------
elif menu == "🔄 규정/보안 교차 검증":
    st.markdown('<div class="main-header">🔄 계약 조건 vs 사내 규정/보안 가이드 교차 검증</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">계약서의 주요 의무 조건이 사내 규정(POL) 및 보안 가이드(SEC)와 충돌하는지 자동 대조합니다.</div>', unsafe_allow_html=True)

    contract_docs = df_docs[df_docs["category"] == "contract"].to_dict("records")
    doc_options = {f"{d['document_id']} - {d['document_title']}": d["document_id"] for d in contract_docs}
    selected_doc_label = st.selectbox("교차 검증 대상 계약서 선택", list(doc_options.keys()))
    selected_doc_id = doc_options[selected_doc_label]

    if st.button("🔍 컴플라이언스 교차 검증 실행", type="primary"):
        with st.spinner("사내 규정 및 보안 가이드 대조 검색 중..."):
            cross_res = cross_checker.cross_check_contract(document_id=selected_doc_id)

        st.markdown("### 📊 규정 준수 통계")
        summary = cross_res.get("summary", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🚨 충돌 (Conflict)", f"{summary.get('conflict', 0)}건")
        c2.metric("⚠️ 주의 (Warning)", f"{summary.get('warning', 0)}건")
        c3.metric("✅ 일치 (Match)", f"{summary.get('match', 0)}건")
        c4.metric("❓ 미확정 (Undetermined)", f"{summary.get('undetermined', 0)}건")

        # Table Display
        st.markdown("### 📋 항목별 대조표")
        comparisons = cross_res.get("comparisons", [])
        table_rows = []
        for item in comparisons:
            cc = item["contract_condition"]
            ip = item["internal_policy_requirement"]
            table_rows.append({
                "주제 (Topic)": item["topic"],
                "상태 (Status)": item["status"],
                "계약서 조건": f"[{cc['section']}] {cc['clause_text'][:80]}...",
                "사내 규정 요구": f"[{ip['document_id']} {ip['document_title']}] {ip['requirement_text'][:80]}...",
                "비교 분석": item["analysis"],
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
        st.markdown(f'<div class="disclaimer-box">⚠️ {cross_res.get("disclaimer", "")}</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 3: RAG 시맨틱 검색 & 질의
# -------------------------------------------------------------
elif menu == "🔍 RAG 시맨틱 검색 & 질의":
    st.markdown('<div class="main-header">🔍 RAG 시맨틱 검색 & 법률/규정 Q&A</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">자연어 질문을 입력하면 pgvector가 60개 문서 중 관련 근거를 검색하고 인용과 함께 답변합니다.</div>', unsafe_allow_html=True)

    col_q, col_cat = st.columns([3, 1])
    with col_q:
        user_query = st.text_input("질문 입력", "계약서에서 개인정보 재위탁 시 어떤 사전 절차가 필요한가요?")
    with col_cat:
        cat_filter = st.selectbox("문서 카테고리 필터", ["전체", "contracts", "policies", "security", "guides"])

    filter_val = None if cat_filter == "전체" else cat_filter

    if st.button("🔎 검색 및 답변 생성", type="primary"):
        with st.spinner("임베딩 생성 및 pgvector 검색 중..."):
            filter_cat_mapped = "contract" if filter_val == "contracts" else ("policy" if filter_val == "policies" else filter_val)
            res = reviewer.review(question=user_query, top_k=8, filter_category=filter_cat_mapped)

        st.markdown("### 💬 에이전트 검토 답변")
        st.info(res.get("answer", ""))

        col_r, col_c = st.columns(2)
        col_r.metric("판정 위험도", res.get("risk_level", "INFO"))
        col_c.metric("확신 수준", res.get("confidence", "Medium"))

        st.markdown("### 📑 인용 근거 (Citations)")
        citations = res.get("citations", [])
        if citations:
            for idx, cit in enumerate(citations, 1):
                st.markdown(f"**[{idx}] {cit.get('document_id')}** ({cit.get('document_title', '')}) - `{cit.get('section', '')}` (p.{cit.get('page', 1)})")
                st.markdown(f"> \"{cit.get('quote', '')}\"")
        else:
            st.warning("명시적 인용 근거가 부족합니다.")

# -------------------------------------------------------------
# TAB 4: RAG 벤치마크 평가 대시보드
# -------------------------------------------------------------
elif menu == "📊 RAG 벤치마크 평가 대시보드":
    st.markdown('<div class="main-header">📊 80문항 RAG 벤치마크 성능 평가 대시보드</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ground Truth (80개 평가 질문, 192개 위험조항) 기반 정량적 성능 지표 분석</div>', unsafe_allow_html=True)

    if not df_evals.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 평가 문항", f"{len(df_evals)}개")
        c2.metric("Retrieval Hit Rate", f"{df_evals['retrieval_hit'].mean()*100:.1f}%")
        c3.metric("Risk Precision", f"{df_evals['precision'].mean()*100:.1f}%")
        c4.metric("Risk Recall / F1", f"{df_evals['recall'].mean()*100:.1f}%")

        st.markdown("### 📈 유형별 성능 비교")
        type_summary = df_evals.groupby("question_type")[["retrieval_hit", "precision", "recall", "f1"]].mean() * 100
        st.dataframe(type_summary.style.format("{:.1f}%"), use_container_width=True)

        st.markdown("### 📋 전체 평가 데이터 상세")
        st.dataframe(df_evals, use_container_width=True)
    else:
        st.warning("outputs/evaluation_report.csv 파일이 없습니다. src/evaluator.py를 먼저 실행해주세요.")
