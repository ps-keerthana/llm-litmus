import streamlit as st
import json
import glob
import os
import time
import urllib.request as _urllib_req
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Central Configuration
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

# Page setup
st.set_page_config(
    page_title="LLM-Litmus | RAG Evaluation Platform",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Modern UI Design Tokens & Glassmorphism CSS ────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero Banner Styling */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
        border: 1px solid rgba(129, 140, 248, 0.25);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(79, 70, 229, 0.3);
    }
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(90deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        color: #cbd5e1;
        font-size: 14px;
        font-weight: 500;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(129, 140, 248, 0.4);
    }
    .metric-card-title {
        font-size: 11px;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .metric-card-val {
        font-size: 32px;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .metric-card-sub {
        font-size: 12px;
        color: #64748b;
        font-weight: 500;
    }

    /* Mode Banner */
    .mode-banner-fast {
        background: rgba(49, 46, 129, 0.35);
        border: 1px solid rgba(129, 140, 248, 0.3);
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 20px;
        color: #c7d2fe;
        font-size: 13px;
    }
    .mode-banner-judge {
        background: rgba(6, 78, 59, 0.35);
        border: 1px solid rgba(52, 211, 153, 0.3);
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 20px;
        color: #a7f3d0;
        font-size: 13px;
    }

    /* Badges */
    .badge-pass {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
    }
    .badge-fail {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.3);
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
    }

    /* Sidebar Clean Styling */
    .css-1d351ef, [data-testid="stSidebar"] {
        background-color: #090d16;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)


# ── Normalize Schema for Legacy/Mock Runs ──────────────────
def normalize_run_data(run: dict) -> dict:
    if not run:
        return run

    summary_defaults = {
        "pass_rate": 0.0,
        "passed": 0,
        "total_questions": 0,
        "avg_retrieval_hit_rate": run.get("avg_relevancy", 1.0),
        "avg_faithfulness": 1.0,
        "hallucination_rate_avg": 0.0,
        "p50_latency_sec": 0.0,
        "p95_latency_sec": 0.0,
        "p99_latency_sec": 0.0,
        "avg_latency_sec": 0.0,
        "avg_cost_usd": 0.0,
        "total_cost_usd": 0.0,
        "git_commit_hash": run.get("commit_sha", "unknown"),
        "branch": "unknown",
        "mode": "unknown",
        "embedding_model": "unknown",
        "llm_model": "unknown",
        "timestamp": run.get("run_timestamp", run.get("timestamp", ""))
    }
    for k, v in summary_defaults.items():
        if k not in run:
            run[k] = v

    normalized_results = []
    for idx, r in enumerate(run.get("results", [])):
        res_defaults = {
            "unique_id": r.get("unique_id", f"Q{idx+1:03d}"),
            "question": r.get("question", ""),
            "ground_truth": r.get("ground_truth", ""),
            "answer": r.get("answer", ""),
            "category": r.get("category", "factual"),
            "difficulty": r.get("difficulty", "easy"),
            "expected_sources": r.get("expected_sources", "unknown"),
            "expected_citations": r.get("expected_citations", ""),
            "reasoning_type": r.get("reasoning_type", "direct_lookup"),
            "hit_rate": r.get("hit_rate", r.get("llm_relevancy", 1.0)),
            "mrr": r.get("mrr", 1.0),
            "context_precision": r.get("context_precision", 1.0),
            "context_recall": r.get("context_recall", 1.0),
            "latency_sec": r.get("latency_sec", 0.0),
            "semantic_similarity": r.get("semantic_similarity", 0.0),
            "correctness": r.get("correctness", r.get("semantic_similarity", 1.0)),
            "faithfulness": r.get("faithfulness", 1.0),
            "hallucination": r.get("hallucination", r.get("hallucination_rate", 0.0)),
            "confidence": r.get("confidence", 1.0),
            "judge_reasoning": r.get("judge_reasoning", r.get("relevancy_reasoning", "No detail.")),
            "status": r.get("status", "PASS"),
            "failure_category": r.get("failure_category", "N/A"),
            "prompt_tokens": r.get("prompt_tokens", 0),
            "completion_tokens": r.get("completion_tokens", 0),
            "cost_usd": r.get("cost_usd", 0.0),
            "retrieved_chunks": r.get("retrieved_chunks", []),
            "retrieved_sources": r.get("retrieved_sources", []),
            "retrieved_similarities": r.get("retrieved_similarities", [])
        }
        if res_defaults["hit_rate"] > 1.0:
            res_defaults["hit_rate"] /= 100.0
        normalized_results.append(res_defaults)

    run["results"] = normalized_results
    return run


# ── FastAPI Backend Connection ─────────────────────────────
API_BASE_URL = os.getenv("EVAL_API_URL", "http://127.0.0.1:8000")


def _api_get(path: str, timeout: int = 4):
    try:
        with _urllib_req.urlopen(f"{API_BASE_URL}{path}", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _api_post(path: str, payload: dict, timeout: int = 5):
    try:
        req = _urllib_req.Request(
            f"{API_BASE_URL}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _urllib_req.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as exc:
        return {"error": str(exc)}


def _api_connected() -> bool:
    return _api_get("/health") is not None


# ── Load Runs & History ────────────────────────────────────
@st.cache_data(ttl=5)
def load_all_runs():
    summaries = _api_get("/runs")
    if summaries is not None:
        runs = []
        for summary in reversed(summaries):
            detail = _api_get(f"/runs/{summary['run_id']}")
            if not detail:
                continue
            run_data = dict(detail.get("metadata") or {})
            run_data.update({
                "run_id": detail["run_id"],
                "status": detail["status"],
                "mode": detail["mode"],
                "commit_sha": detail.get("commit_sha"),
                "results": detail.get("results", []),
            })
            runs.append(normalize_run_data(run_data))
        return runs

    files = sorted(glob.glob(os.path.join("..", config.EVAL_RESULTS_DIR, "run_*.json")))
    if not files:
        files = sorted(glob.glob(os.path.join(config.EVAL_RESULTS_DIR, "run_*.json")))
    runs = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            runs.append(normalize_run_data(data))
        except Exception as e:
            st.warning(f"Could not load {f}: {e}")
    return runs


@st.cache_data(ttl=5)
def load_history_log():
    history_raw = _api_get("/history")
    if history_raw is not None:
        return [normalize_run_data(r) for r in history_raw]

    p = os.path.join("..", config.METRICS_HISTORY_PATH)
    if not os.path.exists(p):
        p = config.METRICS_HISTORY_PATH
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                history = json.load(f)
            return [normalize_run_data(r) for r in history]
        except Exception:
            return []
    return []


runs = [r for r in load_all_runs() if r.get("status") == "completed"]
history = load_history_log()

# ── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.markdown("<h2 style='text-align: center; color: #818cf8; font-weight: 800;'>🧪 LLM-Litmus</h2>", unsafe_allow_html=True)
st.sidebar.caption("Enterprise RAG Quality & Evaluation Suite")
st.sidebar.divider()

nav = st.sidebar.radio("Navigation Pages", [
    "Overview & KPI Matrix",
    "Metric Trends",
    "Regression Analysis",
    "Failure Explorer",
    "Retrieval Inspector",
    "Cost Analytics",
    "Latency Analytics",
    "Prompt Playground",
    "Dataset Explorer"
])

st.sidebar.divider()
st.sidebar.markdown(f"**Dataset Version:** `{config.VERSION_DATASET}`")
st.sidebar.markdown(f"**Retriever:** `{config.VERSION_RETRIEVER}`")
st.sidebar.markdown(f"**LLM Grader:** `{config.VERSION_LLM}`")

_connected = _api_connected()
if _connected:
    st.sidebar.markdown("<span style='color:#4ade80;font-size:12px;font-weight:600;'>● FastAPI Service Connected</span>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<span style='color:#f87171;font-size:12px;font-weight:600;'>● API Standalone File Mode</span>", unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.markdown("### 🚀 Trigger Evaluation")
if _connected:
    eval_mode = st.sidebar.selectbox("Mode", ["smoke", "full"], index=0)
    eval_provider = st.sidebar.selectbox("Provider", ["ollama", "groq", "openai", "anthropic"], index=0)
    no_judge_option = st.sidebar.checkbox("Skip LLM Judge (--no-judge)", value=True)
    if st.sidebar.button("Run Evaluation", type="primary", use_container_width=True):
        res = _api_post("/runs/enqueue", {"mode": eval_mode, "no_judge": no_judge_option, "provider": eval_provider})
        if res and "run_id" in res:
            st.sidebar.success(f"Enqueued `{res['run_id']}`!")
            time.sleep(1)
            st.rerun()
        else:
            st.sidebar.error(f"Failed: {res.get('error') if res else 'Unknown'}")
else:
    st.sidebar.info("Start API (`python api/app.py`) to launch live runs.")

if not runs:
    st.warning("No evaluation runs found. Run `python evaluate.py` first.")
    st.stop()

latest = runs[-1]
prev = runs[-2] if len(runs) > 1 else None


def make_metric_card(col, label, val, sub, delta=None, inverse=False):
    delta_html = ""
    if delta is not None:
        is_pos = delta > 0
        is_good = (not is_pos) if inverse else is_pos
        color = "#34d399" if is_good else "#f87171"
        sym = "+" if is_pos else ""
        delta_html = f" <span style='font-weight:700;color:{color}; font-size:11px;'>({sym}{delta:.3f})</span>"

    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-title">{label}</div>
        <div class="metric-card-val">{val}</div>
        <div class="metric-card-sub">{sub}{delta_html}</div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 1: Overview & KPI Matrix
# ══════════════════════════════════════════════════════════
if nav == "Overview & KPI Matrix":
    # ── Hero Banner ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">LLM Evaluation & RAG Quality Platform</div>
        <div class="hero-subtitle">Automated benchmark testing across 204 Indian income tax Q&A test cases · Real-time RAG accuracy, retrieval recall, and latency telemetry.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Explainer Accordion for 0-Knowledge Viewers ───────────────────────────
    with st.expander("❓ How Does This Platform Work? (Click to Expand Guide)", expanded=False):
        st.markdown("""
        ### 🎯 Core Mission
        This platform evaluates how accurately an **AI RAG (Retrieval-Augmented Generation)** assistant answers complex Indian income tax questions.
        
        ### 🔄 The 4-Step Evaluation Workflow:
        1. **📁 Tax Knowledge Base**: 8 official tax reference guides (Sec 80C, 80D, 24b, HRA, TDS, Capital Gains) indexed in **ChromaDB Vector Store**.
        2. **🔍 RAG Retrieval (Top-5 Chunks)**: When a query is asked, the system retrieves the top 5 most relevant document paragraphs.
        3. **🤖 LLM Answer Generation**: The LLM model (`Llama 3.3 70B` or `Llama 3.2 1B`) generates an official tax response.
        4. **⚖️ Quality Gate Audit**: Answers are benchmarked against 204 ground-truth solutions across 7 query categories:
           - 📌 **Factual**: Direct rule lookup (e.g. 80C limit = Rs 1.5 Lakh)
           - 🧠 **Reasoning**: Deductive tax calculations
           - 🔗 **Multi-Hop**: Multi-section calculations (Sec 80C + 80D + Senior Citizen)
           - ⚠️ **Adversarial**: Tricky questions testing false premises
           - 🛡️ **Out-of-Scope**: Queries outside tax domain expecting polite refusal
        """)

    # ── Evaluation Mode Indicator Banner ──────────────────────────────────────
    judge_active = latest.get("judge_enabled", True)
    if not judge_active:
        st.markdown("""
        <div class="mode-banner-fast">
            ⚡ <b>Fast Execution Mode (Semantic Similarity Scoring)</b> — This run used local semantic similarity for ultra-fast benchmarking. 
            <i>Faithfulness and Hallucination metrics are skipped in fast mode. To run a full LLM-as-a-judge evaluation, uncheck 'Skip LLM Judge' in the sidebar trigger!</i>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="mode-banner-judge">
            🟢 <b>Full LLM Judge Audit Mode</b> — Answers were evaluated across all dimensions (Correctness, Faithfulness, Completeness, Hallucination, and Confidence).
        </div>
        """, unsafe_allow_html=True)

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    make_metric_card(c1, "Pass Rate", f"{latest.get('pass_rate', 0.0)}%",
                     f"{latest.get('passed', 0)}/{latest.get('total_questions', 0)} queries passed",
                     latest.get('pass_rate', 0.0) - prev.get('pass_rate', 0.0) if prev else None)

    make_metric_card(c2, "Retrieval Hit Rate", f"{round(latest.get('avg_retrieval_hit_rate', 1.0)*100, 1)}%",
                     "Context Hit Rate @ K=5",
                     (latest.get('avg_retrieval_hit_rate', 1.0) - prev.get('avg_retrieval_hit_rate', 1.0))*100 if prev else None)

    faith_val = latest.get('avg_faithfulness', 1.0)
    faith_str = f"{faith_val:.3f}" if isinstance(faith_val, (int, float)) else "Fast Mode"
    make_metric_card(c3, "Avg Faithfulness", faith_str,
                     "1.00 = 100% grounded",
                     latest.get('avg_faithfulness', 1.0) - prev.get('avg_faithfulness', 1.0) if (prev and isinstance(faith_val, (int, float)) and isinstance(prev.get('avg_faithfulness'), (int, float))) else None)

    make_metric_card(c4, "p95 Latency", f"{latest.get('p95_latency_sec', 0.0):.2f}s",
                     f"Provider: {latest.get('provider', 'ollama').upper()}",
                     latest.get('p95_latency_sec', 0.0) - prev.get('p95_latency_sec', 0.0) if prev else None, inverse=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Visual Analytics Section ──────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📊 Category-Wise Pass Rate")
        df_q = pd.DataFrame(latest.get("results", []))
        if not df_q.empty and "category" in df_q:
            cat_perf = df_q.groupby("category").apply(
                lambda x: pd.Series({
                    "Total": len(x),
                    "Passed": (x["status"] == "PASS").sum(),
                    "Pass Rate (%)": round((x["status"] == "PASS").mean() * 100, 1)
                })
            ).reset_index()

            fig_bar = px.bar(
                cat_perf,
                x="category",
                y="Pass Rate (%)",
                text="Pass Rate (%)",
                color="Pass Rate (%)",
                color_continuous_scale=["#f87171", "#fbbf24", "#34d399"],
                range_y=[0, 105]
            )
            fig_bar.update_layout(
                height=350,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                xaxis_title="",
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.subheader("🎯 Pass vs Fail Distribution")
        if not df_q.empty and "status" in df_q:
            status_counts = df_q["status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]

            fig_donut = px.pie(
                status_counts,
                names="Status",
                values="Count",
                hole=0.6,
                color="Status",
                color_discrete_map={"PASS": "#34d399", "FAIL": "#f87171"}
            )
            fig_donut.update_layout(
                height=350,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    st.divider()

    # ── Run Metadata Details ──────────────────────────────────────────────────
    st.subheader("⚙️ Run Metadata Envelope")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.json({
            "run_timestamp": latest.get("timestamp"),
            "git_commit_hash": latest.get("git_commit_hash"),
            "branch": latest.get("branch"),
            "execution_mode": latest.get("mode"),
            "llm_provider": latest.get("provider", "ollama"),
            "llm_model": latest.get("llm_model"),
            "embedding_model": latest.get("embedding_model")
        })
    with col_m2:
        st.json({
            "total_questions": latest.get("total_questions"),
            "passed_count": latest.get("passed"),
            "failed_count": latest.get("total_questions", 0) - latest.get("passed", 0),
            "judge_enabled": latest.get("judge_enabled", False),
            "cache_hit_rate": latest.get("cache_hit_rate", 0.0),
            "total_cost_usd": f"${latest.get('avg_cost_usd', 0.0):.6f}"
        })


# ══════════════════════════════════════════════════════════
# PAGE 2: Metric Trends
# ══════════════════════════════════════════════════════════
elif nav == "Metric Trends":
    st.title("📈 Platform Metric Trends")
    st.caption("Tracking pass rate improvements, latency drift, and retrieval recall across execution runs.")
    st.divider()

    raw_list = history if history else runs
    trend_data = []
    for r in raw_list:
        hit_rate = r.get("avg_retrieval_hit_rate", 1.0)
        if hit_rate <= 1.0:
            hit_rate *= 100.0

        faithfulness = r.get("avg_faithfulness", 1.0)
        if isinstance(faithfulness, (int, float)):
            if faithfulness <= 1.0:
                faithfulness *= 100.0
        else:
            faithfulness = None

        trend_data.append({
            "timestamp": r.get("timestamp", "unknown")[:16],
            "pass_rate": r.get("pass_rate", 0.0),
            "avg_retrieval_hit_rate": hit_rate,
            "avg_faithfulness": faithfulness,
            "p95_latency_sec": r.get("p95_latency_sec", 0.0),
            "avg_cost_usd": r.get("avg_cost_usd", 0.0)
        })

    df_trend = pd.DataFrame(trend_data)
    if not df_trend.empty:
        t1, t2 = st.tabs(["Quality Metrics Trends", "Performance & Cost Trends"])
        with t1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_trend["timestamp"], y=df_trend["pass_rate"], name="Pass Rate (%)", line=dict(color="#818cf8", width=3)))
            df_faith = df_trend.dropna(subset=["avg_faithfulness"])
            if not df_faith.empty:
                fig.add_trace(go.Scatter(x=df_faith["timestamp"], y=df_faith["avg_faithfulness"], name="Faithfulness (%)", line=dict(color="#34d399", dash="dash")))
            fig.add_trace(go.Scatter(x=df_trend["timestamp"], y=df_trend["avg_retrieval_hit_rate"], name="Retrieval Hit Rate (%)", line=dict(color="#fbbf24", dash="dot")))
            fig.update_layout(height=420, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Run Timestamp", yaxis_title="Percentage (%)")
            st.plotly_chart(fig, use_container_width=True)
        with t2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_trend["timestamp"], y=df_trend["p95_latency_sec"], name="p95 Latency (s)", line=dict(color="#f87171", width=3)))
            fig2.add_trace(go.Scatter(x=df_trend["timestamp"], y=df_trend["avg_cost_usd"]*1000, name="Avg Cost x1000 ($)", line=dict(color="#60a5fa", dash="dot")))
            fig2.update_layout(height=420, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Run Timestamp")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Execute multiple evaluation runs to generate historical trend analytics.")


# ══════════════════════════════════════════════════════════
# PAGE 3: Regression Analysis
# ══════════════════════════════════════════════════════════
elif nav == "Regression Analysis":
    st.title("🔍 Comparative Regression & Drift Analysis")
    st.caption("Audit quality drift between baseline and candidate runs side-by-side.")
    st.divider()

    if len(runs) < 2:
        st.warning("At least two evaluation runs are required to perform comparative regression checks.")
        st.stop()

    run_opts = [f"{r.get('timestamp')} ({r.get('git_commit_hash', 'unknown')[:7]})" for r in runs]

    col_a, col_b = st.columns(2)
    idx_a = col_a.selectbox("Baseline Run (A)", range(len(run_opts)), index=max(0, len(run_opts)-2), format_func=lambda idx: run_opts[idx])
    idx_b = col_b.selectbox("Candidate Run (B)", range(len(run_opts)), index=len(run_opts)-1, format_func=lambda idx: run_opts[idx])

    run_a, run_b = runs[idx_a], runs[idx_b]

    faith_a = run_a.get('avg_faithfulness', 1.0)
    faith_b = run_b.get('avg_faithfulness', 1.0)
    faith_a_str = f"{faith_a:.3f}" if isinstance(faith_a, (int, float)) else "N/A"
    faith_b_str = f"{faith_b:.3f}" if isinstance(faith_b, (int, float)) else "N/A"
    faith_delta = (faith_b - faith_a) if (isinstance(faith_a, (int, float)) and isinstance(faith_b, (int, float))) else 0.0

    metrics = [
        ("Pass Rate", f"{run_a.get('pass_rate', 0.0)}%", f"{run_b.get('pass_rate', 0.0)}%", run_b.get('pass_rate', 0.0) - run_a.get('pass_rate', 0.0), False),
        ("Retrieval Hit Rate", f"{round(run_a.get('avg_retrieval_hit_rate', 1.0)*100, 1)}%", f"{round(run_b.get('avg_retrieval_hit_rate', 1.0)*100, 1)}%", (run_b.get('avg_retrieval_hit_rate', 1.0) - run_a.get('avg_retrieval_hit_rate', 1.0))*100, False),
        ("Avg Faithfulness", faith_a_str, faith_b_str, faith_delta, False),
        ("p95 Latency", f"{run_a.get('p95_latency_sec', 0.0):.2f}s", f"{run_b.get('p95_latency_sec', 0.0):.2f}s", run_b.get('p95_latency_sec', 0.0) - run_a.get('p95_latency_sec', 0.0), True),
        ("Avg Cost", f"${run_a.get('avg_cost_usd', 0.0):.6f}", f"${run_b.get('avg_cost_usd', 0.0):.6f}", run_b.get('avg_cost_usd', 0.0) - run_a.get('avg_cost_usd', 0.0), True),
    ]

    comp_rows = []
    for name, va, vb, diff, inverse in metrics:
        is_pos = diff > 0
        is_good = (not is_pos) if inverse else is_pos
        status = "Improved" if is_good else ("Regressed" if abs(diff) > 0.001 else "Unchanged")
        comp_rows.append({"Metric": name, "Baseline (A)": va, "Candidate (B)": vb, "Delta": f"{'+' if is_pos else ''}{diff:.4f}", "Status": status})

    df_comp = pd.DataFrame(comp_rows)
    st.dataframe(df_comp.style.map(lambda status: "color: #34d399; font-weight:700;" if status == "Improved" else ("color: #f87171; font-weight:700;" if status == "Regressed" else ""), subset=["Status"]), use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 4: Failure Explorer
# ══════════════════════════════════════════════════════════
elif nav == "Failure Explorer":
    st.title("🐞 Failure & Telemetry Explorer")
    st.caption("Inspect and debug failing test cases. Deep-dive into prompt inputs, ground truths, and counterfactual reports.")
    st.divider()

    df_q = pd.DataFrame(latest.get("results", []))
    fails = df_q[df_q["status"] == "FAIL"] if not df_q.empty else pd.DataFrame()

    if fails.empty:
        st.success("🎉 All queries passed in the latest execution run!")
    else:
        tab_summary, tab_trace = st.tabs([f"Active Failures Grid ({len(fails)})", "Failure Trace Inspector"])

        with tab_summary:
            cols_show = [c for c in ["unique_id", "question", "category", "failure_category", "attribution_reason", "correctness", "latency_sec"] if c in fails.columns]
            st.dataframe(fails[cols_show], use_container_width=True)

        with tab_trace:
            selected_id = st.selectbox("Select failing query ID to inspect", fails["unique_id"].tolist())
            row = fails[fails["unique_id"] == selected_id].iloc[0]

            st.markdown(f"### Question: `{row['question']}`")
            col_f1, col_f2 = st.columns(2)
            col_f1.error(f"**Failure Category:** {row.get('failure_category', 'unknown')}")
            col_f2.info(f"**Attribution:** {row.get('attribution_reason', 'No detail recorded.')}")

            st.markdown("#### Ground Truth vs Model Answer")
            col_gt, col_ans = st.columns(2)
            col_gt.success(f"**Ground Truth:**\n{row.get('ground_truth', '')}")
            col_ans.warning(f"**Model Answer:**\n{row.get('answer', '')}")


# ══════════════════════════════════════════════════════════
# OTHER PAGES (Fallback Views)
# ══════════════════════════════════════════════════════════
else:
    st.title(f"📌 {nav}")
    st.caption("Detailed breakdown and raw dataset telemetry.")
    df_all = pd.DataFrame(latest.get("results", []))
    if not df_all.empty:
        st.dataframe(df_all, use_container_width=True)
    else:
        st.info("No query results available for this view.")
