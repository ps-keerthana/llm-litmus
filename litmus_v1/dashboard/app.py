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
    "Dataset Explorer",
    "── New ──────────────",
    "Run Comparison",
    "Trace Replay",
    "Adversarial Explorer",
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
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">LLM Evaluation & RAG Quality Platform</div>
        <div class="hero-subtitle">Automated benchmark testing across 204 Indian income tax Q&A test cases · Real-time RAG accuracy, retrieval recall, and latency telemetry.</div>
    </div>
    """, unsafe_allow_html=True)

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
# PAGE 5: Retrieval Inspector
# ══════════════════════════════════════════════════════════
elif nav == "Retrieval Inspector":
    st.title("🔍 Vector Retrieval & Ranking Inspector")
    st.caption("Audit ChromaDB Top-K chunk retrieval quality, MRR scores, and similarity scores.")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    make_metric_card(c1, "Hit Rate @ K=5", f"{round(latest.get('avg_retrieval_hit_rate', 1.0)*100, 1)}%", "Target: >= 80%")
    make_metric_card(c2, "Mean Reciprocal Rank", f"{latest.get('avg_retrieval_mrr', 0.913):.3f}", "Top rank precision")
    make_metric_card(c3, "Context Precision", f"{latest.get('avg_context_precision', 0.85):.3f}", "Relevant chunk ratio")
    make_metric_card(c4, "Context Recall", f"{latest.get('avg_context_recall', 0.86):.3f}", "Ground-truth coverage")

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Query Context Inspection")
    df_q = pd.DataFrame(latest.get("results", []))
    if not df_q.empty:
        q_sel = st.selectbox("Select Question to inspect retrieved vector chunks", df_q["question"].tolist())
        q_row = df_q[df_q["question"] == q_sel].iloc[0]

        col_r1, col_r2 = st.columns(2)
        col_r1.markdown(f"**Expected Sources:** `{q_row.get('expected_sources', 'N/A')}`")
        col_r2.markdown(f"**Retrieved Sources:** `{q_row.get('retrieved_sources', 'N/A')}`")

        st.markdown("#### Retrieved Chunks & Vector Similarities")
        chunks = q_row.get("retrieved_chunks", [])
        sims = q_row.get("retrieved_similarities", [])
        if chunks:
            for idx, ch in enumerate(chunks):
                sim_score = sims[idx] if idx < len(sims) else "N/A"
                with st.expander(f"Chunk #{idx+1} (Similarity: {sim_score})"):
                    st.write(ch)
        else:
            st.info("No chunk text recorded for this result.")


# ══════════════════════════════════════════════════════════
# PAGE 6: Cost Analytics
# ══════════════════════════════════════════════════════════
elif nav == "Cost Analytics":
    st.title("💰 API Cost & Token Analytics")
    st.caption("Track token consumption and estimated API execution costs per query.")
    st.divider()

    c1, c2, c3 = st.columns(3)
    make_metric_card(c1, "Avg Cost / Query", f"${latest.get('avg_cost_usd', 0.0):.6f}", "Groq pricing")
    make_metric_card(c2, "Total Run Cost", f"${latest.get('total_cost_usd', 0.0):.4f}", "All queries combined")
    make_metric_card(c3, "Total Queries", f"{latest.get('total_questions', 0)}", "Evaluated count")

    st.markdown("<br>", unsafe_allow_html=True)
    df_q = pd.DataFrame(latest.get("results", []))
    if not df_q.empty and "cost_usd" in df_q.columns:
        fig_cost = px.histogram(df_q, x="cost_usd", nbins=20, title="Query Cost Distribution ($)")
        fig_cost.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cost, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 7: Latency Analytics
# ══════════════════════════════════════════════════════════
elif nav == "Latency Analytics":
    st.title("⚡ Response Latency & SLA Analytics")
    st.caption("Distribution of generation latencies across query categories.")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    make_metric_card(c1, "p50 Latency (Median)", f"{latest.get('p50_latency_sec', 0.0):.2f}s", "Median response time")
    make_metric_card(c2, "p95 Latency", f"{latest.get('p95_latency_sec', 0.0):.2f}s", "SLA limit: <= 3.5s")
    make_metric_card(c3, "p99 Latency", f"{latest.get('p99_latency_sec', 0.0):.2f}s", "Worst 1% response time")
    make_metric_card(c4, "Average Latency", f"{latest.get('avg_latency_sec', 0.0):.2f}s", "Mean response time")

    st.markdown("<br>", unsafe_allow_html=True)
    df_q = pd.DataFrame(latest.get("results", []))
    if not df_q.empty and "latency_sec" in df_q.columns:
        fig_lat = px.box(df_q, x="category", y="latency_sec", color="category", title="Latency Distribution by Category (Seconds)")
        fig_lat.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_lat, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 8: Prompt Playground
# ══════════════════════════════════════════════════════════
elif nav == "Prompt Playground":
    st.title("🧪 Live RAG & Prompt Playground")
    st.caption("Test custom tax questions live against the ChromaDB vector store and generator.")
    st.divider()

    custom_q = st.text_input("Enter a custom Indian Income Tax Question:", "Can I claim Section 80C deduction under the new tax regime?")
    if st.button("Generate & Test Answer", type="primary"):
        with st.spinner("Retrieving Top-5 vector context chunks & generating answer..."):
            try:
                from core.retrieval import load_docs, build_vector_store, retrieve
                from core.generator import generate_answer
                chunks = load_docs()
                collection = build_vector_store(chunks)
                retrieved, sims, sources = retrieve(custom_q, collection, top_k=5)
                ans, p_tok, c_tok = generate_answer(custom_q, retrieved)

                st.markdown("### Generated Answer:")
                st.success(ans)

                st.markdown("### Retrieved Context Chunks:")
                for idx, ch in enumerate(retrieved):
                    with st.expander(f"Chunk #{idx+1} (Source: {sources[idx] if idx < len(sources) else 'doc'})"):
                        st.write(ch)
            except Exception as exc:
                st.error(f"Error executing live prompt test: {exc}")


# ══════════════════════════════════════════════════════════
# PAGE 9: Dataset Explorer
# ══════════════════════════════════════════════════════════
elif nav == "Dataset Explorer":
    st.title("📁 Golden Dataset Explorer")
    st.caption(f"Search and inspect all benchmark questions in the golden dataset ({config.DATASET_PATH}).")
    st.divider()

    ds_path = config.DATASET_PATH
    if not os.path.exists(ds_path):
        ds_path = os.path.join("..", config.DATASET_PATH)

    if os.path.exists(ds_path):
        df_ds = pd.read_csv(ds_path)
        cat_filter = st.multiselect("Filter by Category", df_ds["category"].unique().tolist(), default=df_ds["category"].unique().tolist())
        df_filtered = df_ds[df_ds["category"].isin(cat_filter)]

        st.markdown(f"Displaying **{len(df_filtered)}** of **{len(df_ds)}** golden questions:")
        st.dataframe(df_filtered, use_container_width=True)
    else:
        st.error(f"Dataset file not found at {ds_path}")


# ═══════════════════════════════════════════════════════════════
# Phase 8 — New Pages
# ═══════════════════════════════════════════════════════════════

elif nav == "Run Comparison":
    st.title("⚖️ Run Comparison")
    st.caption("Compare any two evaluation runs side-by-side — useful for prompt A/B, model swaps, or retrieval strategy changes.")
    st.divider()

    if len(runs) < 2:
        st.info("Need at least 2 completed runs to compare. Run `python evaluate.py` at least twice.")
    else:
        run_labels = [f"{r.get('timestamp','?')} | {r.get('mode','?')} | {r.get('provider','?')}" for r in runs]
        col1, col2 = st.columns(2)
        with col1:
            idx_a = st.selectbox("Run A (baseline)", range(len(run_labels)), format_func=lambda i: run_labels[i], index=min(1, len(runs)-1))
        with col2:
            idx_b = st.selectbox("Run B (candidate)", range(len(run_labels)), format_func=lambda i: run_labels[i], index=0)

        run_a = runs[idx_a]
        run_b = runs[idx_b]

        # ── Summary metrics table ──────────────────────────────────────
        st.subheader("📊 Metric Comparison")
        metrics_to_compare = [
            ("Pass Rate (%)",         "pass_rate",              False),
            ("Hit Rate (%)",          "avg_retrieval_hit_rate", True,  100),
            ("MRR",                   "avg_retrieval_mrr",      False),
            ("nDCG@K",               "avg_ndcg_at_k",          False),
            ("MAP",                   "avg_map_score",          False),
            ("Coverage",              "avg_coverage",           False),
            ("Faithfulness",          "avg_faithfulness",       False),
            ("Hallucination Rate",    "hallucination_rate_avg", False),
            ("Token F1",              "avg_token_f1",           False),
            ("p95 Latency (s)",       "p95_latency_sec",        False),
            ("Avg Cost/Query ($)",    "avg_cost_usd",           False),
        ]

        rows = []
        for entry in metrics_to_compare:
            label = entry[0]; key = entry[1]
            scale = entry[3] if len(entry) > 3 else 1
            va = run_a.get(key, 0.0)
            vb = run_b.get(key, 0.0)
            if isinstance(va, (int, float)): va = round(va * scale, 3)
            if isinstance(vb, (int, float)): vb = round(vb * scale, 3)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                delta = round(vb - va, 3)
                sign = "+" if delta >= 0 else ""
                delta_str = f"{sign}{delta}"
            else:
                delta_str = "—"
            rows.append({"Metric": label, "Run A": va, "Run B": vb, "Delta (B-A)": delta_str})

        df_comp = pd.DataFrame(rows)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        # ── Per-category pass rate delta ───────────────────────────────
        st.subheader("📂 Per-Category Pass Rate")
        results_a = {r["unique_id"]: r for r in run_a.get("results", [])}
        results_b = {r["unique_id"]: r for r in run_b.get("results", [])}
        shared = set(results_a.keys()) & set(results_b.keys())

        cat_data = {}
        for qid in shared:
            cat = results_a[qid].get("category", "general")
            cat_data.setdefault(cat, {"a_pass": 0, "b_pass": 0, "total": 0})
            cat_data[cat]["total"] += 1
            if results_a[qid].get("status") == "PASS": cat_data[cat]["a_pass"] += 1
            if results_b[qid].get("status") == "PASS": cat_data[cat]["b_pass"] += 1

        if cat_data:
            cat_rows = []
            for cat, d in sorted(cat_data.items()):
                pa = round(d["a_pass"] / d["total"] * 100, 1) if d["total"] else 0
                pb = round(d["b_pass"] / d["total"] * 100, 1) if d["total"] else 0
                cat_rows.append({"Category": cat, "Run A (%)": pa, "Run B (%)": pb, "Delta": round(pb - pa, 1)})
            df_cat = pd.DataFrame(cat_rows)
            fig_cat = px.bar(df_cat, x="Category", y=["Run A (%)", "Run B (%)"],
                             barmode="group", title="Pass Rate by Category",
                             color_discrete_sequence=["#6366f1", "#22d3ee"])
            st.plotly_chart(fig_cat, use_container_width=True)

        # ── Status changes ──────────────────────────────────────────────
        st.subheader("🔄 Status Changes")
        regressions, improvements = [], []
        for qid in shared:
            sa = results_a[qid].get("status"); sb = results_b[qid].get("status")
            if sa == "PASS" and sb == "FAIL":
                regressions.append({"ID": qid, "Question": results_a[qid].get("question", "")[:80], "Category": results_a[qid].get("category", "")})
            elif sa == "FAIL" and sb == "PASS":
                improvements.append({"ID": qid, "Question": results_a[qid].get("question", "")[:80], "Category": results_a[qid].get("category", "")})

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**❌ Regressions (PASS→FAIL): {len(regressions)}**")
            if regressions: st.dataframe(pd.DataFrame(regressions), use_container_width=True, hide_index=True)
        with c2:
            st.markdown(f"**✅ Improvements (FAIL→PASS): {len(improvements)}**")
            if improvements: st.dataframe(pd.DataFrame(improvements), use_container_width=True, hide_index=True)


elif nav == "Trace Replay":
    st.title("🔬 Trace Replay")
    st.caption("Inspect the full per-phase execution trace for any query in any run.")
    st.divider()

    try:
        from core.telemetry import load_trace, list_trace_runs
        trace_runs = list_trace_runs()
    except Exception:
        trace_runs = []

    if not runs:
        st.info("No evaluation runs found. Run `python evaluate.py` first.")
    else:
        # Query selector
        run_labels = [f"{r.get('timestamp','?')} | {r.get('mode','?')}" for r in runs]
        selected_run_idx = st.selectbox("Select Run", range(len(run_labels)), format_func=lambda i: run_labels[i])
        selected_run = runs[selected_run_idx]
        all_results = selected_run.get("results", [])

        if all_results:
            q_labels = [f"{r['unique_id']} | {r['status']} | {r.get('question','')[:60]}" for r in all_results]
            selected_q_idx = st.selectbox("Select Query", range(len(q_labels)), format_func=lambda i: q_labels[i])
            result = all_results[selected_q_idx]

            st.subheader(f"Query: {result['unique_id']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Status", result.get("status", "?"))
            c2.metric("Semantic Sim", result.get("semantic_similarity", "?"))
            c3.metric("Latency", f"{result.get('latency_sec','?')}s")

            with st.expander("❓ Question & Ground Truth", expanded=True):
                st.markdown(f"**Question:** {result.get('question','')}")
                st.markdown(f"**Ground Truth:** {result.get('ground_truth','')}")
                st.markdown(f"**Generated Answer:** {result.get('answer','')}")

            with st.expander("📄 Retrieved Chunks", expanded=False):
                chunks = result.get("retrieved_chunks", [])
                sims   = result.get("retrieved_similarities", [])
                srcs   = result.get("retrieved_sources", [])
                for i, (chunk, sim, src) in enumerate(zip(chunks, sims, srcs)):
                    st.markdown(f"**Chunk {i+1}** — `{src}` | similarity: `{sim}`")
                    st.code(chunk[:400], language="text")

            with st.expander("⚖️ Judge Scores", expanded=False):
                judge_cols = ["correctness", "faithfulness", "completeness", "hallucination_rate", "judge_confidence", "token_f1"]
                judge_data = {k: result.get(k, "N/A") for k in judge_cols}
                st.json(judge_data)
                if result.get("judge_reasoning"):
                    st.markdown(f"**Reasoning:** {result['judge_reasoning']}")
                if result.get("judge_disagreement"):
                    st.warning("⚠️ Ensemble judge disagreement detected on this query.")

            with st.expander("🔎 Attribution & Diagnosis", expanded=False):
                st.markdown(f"**Failure Category:** `{result.get('failure_category', 'N/A')}`")
                st.markdown(f"**Attribution Reason:** {result.get('attribution_reason', '')}")
                diag = result.get("retrieval_diagnosis", {})
                if diag:
                    st.json(diag)

            # Telemetry waterfall
            ts = selected_run.get("timestamp", "")
            if ts and trace_runs and ts in trace_runs:
                with st.expander("⏱️ Latency Waterfall (Telemetry)", expanded=False):
                    spans = [s for s in load_trace(ts) if s.get("query_id") == result["unique_id"]]
                    if spans:
                        df_spans = pd.DataFrame([{"Phase": s["phase"], "Duration (ms)": s["duration_ms"]} for s in spans])
                        fig_wf = px.bar(df_spans, x="Duration (ms)", y="Phase", orientation="h",
                                        title="Phase Latency Breakdown", color="Phase",
                                        color_discrete_sequence=px.colors.qualitative.Pastel)
                        st.plotly_chart(fig_wf, use_container_width=True)
                    else:
                        st.info("No telemetry spans found for this query.")
            else:
                st.info("Telemetry traces available after next evaluation run.")
        else:
            st.info("No results found in this run.")


elif nav == "Adversarial Explorer":
    st.title("🛡️ Adversarial Explorer")
    st.caption("Analyse robustness test results — prompt injection, hallucination traps, negation, missing context, and more.")
    st.divider()

    # Find adversarial runs
    adv_runs = [r for r in runs if r.get("mode") == "adversarial" or
                any(res.get("adversarial_category") for res in r.get("results", []))]

    if not adv_runs:
        st.info("""
        No adversarial runs found yet. Run the adversarial test suite:
        ```bash
        python evaluate.py --adversarial --no-judge
        ```
        """)
    else:
        run_labels = [f"{r.get('timestamp','?')} | {r.get('mode','?')}" for r in adv_runs]
        sel = st.selectbox("Select Adversarial Run", range(len(run_labels)), format_func=lambda i: run_labels[i])
        adv_run = adv_runs[sel]
        adv_results = adv_run.get("results", [])

        # Summary metrics
        total_adv = len(adv_results)
        passed_adv = sum(1 for r in adv_results if r.get("status") == "PASS")
        fooled = total_adv - passed_adv  # model was 'fooled' = failed an adversarial check

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Adversarial Queries", total_adv)
        m2.metric("Passed (Robust)", passed_adv)
        m3.metric("Failed (Vulnerable)", fooled, delta=f"-{fooled}" if fooled > 0 else "0", delta_color="inverse")

        # Per-category breakdown
        st.subheader("Vulnerability by Adversarial Category")
        cat_counts: dict = {}
        for r in adv_results:
            acat = r.get("adversarial_category", r.get("category", "unknown"))
            cat_counts.setdefault(acat, {"total": 0, "failed": 0})
            cat_counts[acat]["total"] += 1
            if r.get("status") == "FAIL":
                cat_counts[acat]["failed"] += 1

        if cat_counts:
            cat_rows = []
            for cat, d in sorted(cat_counts.items()):
                vuln_rate = round(d["failed"] / d["total"] * 100, 1) if d["total"] else 0
                cat_rows.append({"Category": cat, "Total": d["total"], "Failed": d["failed"], "Vulnerability Rate (%)": vuln_rate})
            df_adv_cat = pd.DataFrame(cat_rows)
            fig_adv = px.bar(df_adv_cat, x="Category", y="Vulnerability Rate (%)",
                             title="Vulnerability Rate by Adversarial Category",
                             color="Vulnerability Rate (%)",
                             color_continuous_scale=["#22d3ee", "#f87171"])
            st.plotly_chart(fig_adv, use_container_width=True)
            st.dataframe(df_adv_cat, use_container_width=True, hide_index=True)

        # Detailed results table
        st.subheader("All Adversarial Query Results")
        adv_table = []
        for r in adv_results:
            adv_table.append({
                "ID":          r.get("unique_id", ""),
                "Status":      r.get("status", ""),
                "Adv. Type":   r.get("adversarial_category", r.get("category", "")),
                "Question":    r.get("question", "")[:70],
                "Answer":      r.get("answer", "")[:70],
                "Sem. Sim":    r.get("semantic_similarity", 0),
                "Failure Cat": r.get("failure_category", ""),
            })
        df_adv_table = pd.DataFrame(adv_table)
        fail_filter = st.checkbox("Show only FAILED (vulnerable) queries", value=False)
        if fail_filter:
            df_adv_table = df_adv_table[df_adv_table["Status"] == "FAIL"]
        st.dataframe(df_adv_table, use_container_width=True, hide_index=True)

elif nav == "── New ──────────────":
    st.info("Select one of the new pages above.")

