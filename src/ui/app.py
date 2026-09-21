import json
import sys
from pathlib import Path
import streamlit as st

# Setup path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import os
import requests
from src.config import settings
from src.models.claim import ClaimCase
from src.models.decision import AdjudicationResult
from src.agents.workflow import ClaimAdjudicationPipeline
from src.rag.retriever import HybridRetriever

st.set_page_config(
    page_title="Policy-Aware Claim Decision Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .reportview-container {
        background: #f8fafc;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-admissible {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-limits {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-not-admissible {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-needs-review {
        background-color: #ffedd5;
        color: #9a3412;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .citation-box {
        border-left: 4px solid #3b82f6;
        background-color: #f8fafc;
        padding: 12px;
        margin-bottom: 12px;
        border-radius: 0 6px 6px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    retriever = HybridRetriever(settings.policy_chunks_path)
    return ClaimAdjudicationPipeline(retriever=retriever)


@st.cache_data
def load_public_cases():
    local_file = root_dir / "data" / "candidate_data" / "public_test_cases.json"
    if local_file.exists():
        with open(local_file, "r", encoding="utf-8") as f:
            return json.load(f)
    parent_file = root_dir.parent / "candidate_data" / "public_test_cases.json"
    if parent_file.exists():
        with open(parent_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@st.cache_data
def load_candidate_cases():
    cases_file = root_dir / "eval" / "candidate_test_cases.json"
    if cases_file.exists():
        with open(cases_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


pipeline = get_pipeline()
public_cases = load_public_cases()
candidate_cases = load_candidate_cases()

all_preloaded = {c["case_id"]: c for c in public_cases}
for c in candidate_cases:
    all_preloaded[c["case_id"]] = c

st.title("🛡️ Health Insurance Claim Decision Engine")
st.caption("Policy-Aware Multi-Agent RAG Adjudication Engine • Universal Sompo Health Insurance")

# Sidebar for case selection
with st.sidebar:
    st.header("Case Selector")
    input_mode = st.radio("Input Source", ["Preloaded Cases", "Paste Custom JSON", "Upload JSON File"])

    selected_case_data = None

    if input_mode == "Preloaded Cases":
        case_options = list(all_preloaded.keys())
        case_id = st.selectbox("Select Case ID", case_options, index=0)
        selected_case_data = all_preloaded.get(case_id)

        if selected_case_data:
            st.markdown(f"**Diagnosis:** {selected_case_data['treatment']['diagnosis']}")
            st.markdown(f"**Sum Insured:** INR {selected_case_data['sum_insured_inr']:,}")
            st.markdown(f"**Coverage Duration:** {selected_case_data['continuous_coverage_months']} months")

    elif input_mode == "Paste Custom JSON":
        json_input = st.text_area("Paste Claim JSON", height=300)
        if json_input:
            try:
                selected_case_data = json.loads(json_input)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    elif input_mode == "Upload JSON File":
        uploaded_file = st.file_uploader("Choose JSON File", type=["json"])
        if uploaded_file:
            try:
                selected_case_data = json.load(uploaded_file)
            except Exception as e:
                st.error(f"Error parsing file: {e}")

    st.divider()
    with st.expander("⚙️ Backend API Configuration"):
        default_backend = os.getenv("BACKEND_API_URL", "")
        try:
            if hasattr(st, "secrets") and "BACKEND_API_URL" in st.secrets:
                default_backend = str(st.secrets["BACKEND_API_URL"])
        except Exception:
            pass

        backend_url = st.text_input(
            "Remote Backend URL (Optional)",
            value=default_backend,
            placeholder="e.g. https://claim-engine-api.onrender.com",
            help="If provided, adjudication requests will be sent to this deployed FastAPI backend. Leave blank to run in-memory."
        )
        if backend_url.strip():
            try:
                h_res = requests.get(f"{backend_url.strip().rstrip('/')}/health", timeout=3)
                if h_res.status_code == 200:
                    st.caption("🟢 Backend API Connected")
                else:
                    st.caption(f"🟡 Backend returned status {h_res.status_code}")
            except Exception:
                st.caption("🔴 Cannot reach Backend API")

    run_button = st.button("🚀 Run Adjudication", type="primary", use_container_width=True)

# Main content
if selected_case_data:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📄 Claim Input Details")
        st.json(selected_case_data, expanded=False)

    if run_button or "current_result" in st.session_state:
        if run_button:
            with st.spinner("Executing Multi-Agent Adjudication Workflow..."):
                try:
                    case_obj = ClaimCase(**selected_case_data)
                    if backend_url and backend_url.strip():
                        api_endpoint = f"{backend_url.strip().rstrip('/')}/analyze"
                        resp = requests.post(api_endpoint, json=case_obj.model_dump(), timeout=60)
                        if resp.status_code != 200:
                            st.error(f"Backend API Error ({resp.status_code}): {resp.text}")
                            st.stop()
                        result = AdjudicationResult(**resp.json())
                    else:
                        result = pipeline.run(case_obj)
                    st.session_state.current_result = result
                except Exception as e:
                    st.error(f"Execution Error: {str(e)}")
                    st.stop()

        result = st.session_state.current_result

        with col2:
            st.subheader("⚖️ Adjudication Verdict")

            # Status Badge
            status_val = result.decision.value
            badge_class = {
                "ADMISSIBLE": "badge-admissible",
                "ADMISSIBLE_WITH_LIMITS": "badge-limits",
                "NOT_ADMISSIBLE": "badge-not-admissible",
                "NEEDS_REVIEW": "badge-needs-review",
                "PARTIALLY_ADMISSIBLE": "badge-limits"
            }.get(status_val, "badge-limits")

            st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
                <span class="{badge_class}" style="font-size: 1.25rem;">{status_val}</span>
                <span style="font-weight: 600; color: #475569;">Confidence: {int(result.confidence * 100)}%</span>
            </div>
            """, unsafe_allow_html=True)

            if status_val == "NEEDS_REVIEW":
                st.warning("⚠️ **ABSTENTION TRIGGERED**: The system cannot safely adjudicate this claim due to missing critical documentation or unverified hospital statutory criteria.")

        st.divider()

        # Detailed Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Findings & Limits",
            "🔍 Policy Evidence & Citations",
            "🛡️ Guardrail Validation",
            "⏱️ Execution Trace"
        ])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Key Findings")
                if result.key_findings:
                    for kf in result.key_findings:
                        st.markdown(f"- {kf}")
                else:
                    st.info("No critical findings noted.")

                if result.missing_evidence:
                    st.markdown("#### ⚠️ Missing Evidence / Gaps")
                    for me in result.missing_evidence:
                        st.markdown(f"- **{me}**")

            with c2:
                st.markdown("#### Applicable Limits & Deductions")
                if result.applicable_limits:
                    for al in result.applicable_limits:
                        st.markdown(f"- ⚖️ {al}")
                else:
                    st.info("No policy deductions or sub-limits applied.")

        with tab2:
            st.markdown(f"#### Authoritative Policy Citations ({len(result.citations)} Cited Clauses)")
            for idx, cit in enumerate(result.citations):
                st.markdown(f"""
                <div class="citation-box">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="font-weight: 700; color: #1e40af;">[{cit.chunk_id}] {cit.section}</span>
                        <span style="font-size: 0.85rem; color: #64748b;">Source: {cit.source} • Page {cit.page}</span>
                    </div>
                    <p style="margin: 0; color: #1e293b;">{cit.claim}</p>
                </div>
                """, unsafe_allow_html=True)

        with tab3:
            v_status = result.validation.status
            if v_status == "PASS":
                st.success("✅ **Guardrail Status: PASS** — Every material decision statement is strictly supported by policy citations.")
            else:
                st.error("❌ **Guardrail Status: FAIL** — Unsupported assertions detected.")

            if result.validation.unsupported_claims:
                st.markdown("##### Flagged Unsupported Claims:")
                for uc in result.validation.unsupported_claims:
                    st.markdown(f"- {uc}")

        with tab4:
            st.markdown("#### Agent Execution Trace & Latencies")
            trace_data = []
            for t in result.trace:
                trace_data.append({
                    "Agent": t.agent,
                    "Action": t.action,
                    "Elapsed (ms)": t.timestamp_ms,
                    "Metadata": json.dumps(t.details) if t.details else ""
                })
            st.dataframe(trace_data, use_container_width=True)

else:
    st.info("👈 Please select or paste a claim case from the sidebar to begin.")
