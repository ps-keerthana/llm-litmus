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
    page_title="LLM Eval Platform",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design Styles
st.markdown("""
<style>
    .metric-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        color: white;
    }
    .metric-label {
        font-size: 12px; color: #94a3b8; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;
    }
    .metric-val { font-size: 30px; font-weight: 700; margin-bottom: 2px; }
    .metric-sub { font-size: 11px; color: #64748b; }
    .badge-p {
        background-color: #312e81; color: #c7d2fe;
        padding: 4px 10px; border-radius: 999px;
        font-size: 11px; font-weight: 600;
        display: inline-block; margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Normalize Schema for Legacy/Mock Runs ──────────────────
def normalize_run_data(run: dict) -> dict:
    """
    Standardizes run output dictionaries (e.g., from old mock runs)
    to match the new modular RAG evaluation pipeline schema, preventing KeyErrors.
    """
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
        # In case some old items had hit_rate stored as percentage in the list
        if res_defaults["hit_rate"] > 1.0:
            res_defaults["hit_rate"] /= 100.0
        normalized_results.append(res_defaults)

    run["results"] = normalized_results
    return run


# ── FastAPI Backend Connection ─────────────────────────────
# Point to the running api/app.py service. Override via env var for staging.
API_BASE_URL = os.getenv("EVAL_API_URL", "http://127.0.0.1:8000")


def _api_get(path: str, timeout: int = 4):
    """Call the FastAPI backend. Returns parsed JSON dict/list or None on error."""
    try:
        with _urllib_req.urlopen(f"{API_BASE_URL}{path}", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _api_connected() -> bool:
    """Lightweight probe to check if the API is reachable."""
    return _api_get("/health") is not None


# ── Load Runs & History ────────────────────────────────────
@st.cache_data(ttl=5)
def load_all_runs():
    """
    Primary path: fetch run list from FastAPI (GET /runs then GET /runs/{run_id}).
    Fallback: read eval_results/*.json files directly (API not running).
    """
    summaries = _api_get("/runs")
    if summaries is not None:
        runs = []
        for summary in reversed(summaries):  # oldest first (API returns newest-first)
            detail = _api_get(f"/runs/{summary['run_id']}")
            if not detail:
                continue
            # Flatten stored metadata blob to top level so normalize_run_data() works
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

    # ── Filesystem fallback ──────────────────────────────────
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
    """
    Primary path: fetch history from FastAPI (GET /history).
    Fallback: read metrics_history.json directly.
    """
    history_raw = _api_get("/history")
    if history_raw is not None:
        return [normalize_run_data(r) for r in history_raw]

    # ── Filesystem fallback ──────────────────────────────────
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

# Sidebar Setup
st.sidebar.markdown("<h2 style='text-align: center; color: #818cf8;'>🧪 LLM Eval Platform</h2>", unsafe_allow_html=True)
st.sidebar.caption("Enterprise-grade evaluation diagnostics")
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

# API connection status badge
_connected = _api_connected()
if _connected:
    st.sidebar.markdown(
        "<span style='color:#4ade80;font-size:12px;'>● API connected</span>",
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        "<span style='color:#f87171;font-size:12px;'>● API offline — reading from files</span>",
        unsafe_allow_html=True,
    )

if not runs:
    st.warning("No evaluation runs found. Run `python evaluate.py` first.")
    st.stop()

latest = runs[-1]
prev = runs[-2] if len(runs) > 1 else None

# Helper card UI
def make_metric_card(col, label, val, sub, delta=None, inverse=False):
    delta_html = ""
    if delta is not None:
        is_pos = delta > 0
        is_good = (not is_pos) if inverse else is_pos
        color = "#10b981" if is_good else "#ef4444"
        sym = "+" if is_pos else ""
        delta_html = f"<span style='font-weight:600;color:{color};'> ({sym}{delta:.3f})</span>"
        
    col.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-val">{val}</div>
        <div class="metric-sub">{sub}{delta_html}</div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 1: Overview & KPI Matrix
# ══════════════════════════════════════════════════════════
if nav == "Overview & KPI Matrix":
    st.markdown("<div class='badge-p'>Platform Statistics</div>", unsafe_allow_html=True)
    st.title("System Overview")
    st.caption("Active platform status, baseline metadata, and core aggregate KPIs.")
    st.divider()

    # Honest Metrics warning banner
    if latest.get("judge_enabled") is False:
        st.warning("⚠️ LLM Judge Metrics (Faithfulness, Hallucination) are disabled for this run (bypassed in smoke test).")

    c1, c2, c3, c4 = st.columns(4)
    
    make_metric_card(c1, "Pass Rate", f"{latest.get('pass_rate', 0.0)}%",
                     f"{latest.get('passed', 0)}/{latest.get('total_questions', 0)} passed",
                     latest.get('pass_rate', 0.0) - prev.get('pass_rate', 0.0) if prev else None)
                     
    make_metric_card(c2, "Retrieval Hit Rate", f"{round(latest.get('avg_retrieval_hit_rate', 1.0)*100, 1)}%",
                     "Hit Rate @ K",
                     (latest.get('avg_retrieval_hit_rate', 1.0) - prev.get('avg_retrieval_hit_rate', 1.0))*100 if prev else None)
                     
    # Display Not Evaluated cleanly in card
    faith_val = latest.get('avg_faithfulness', 1.0)
    faith_str = f"{faith_val:.3f}" if isinstance(faith_val, (int, float)) else "N/A (Skipped)"
    make_metric_card(c3, "Avg Faithfulness", faith_str,
                     "1.00 = grounded context",
                     latest.get('avg_faithfulness', 1.0) - prev.get('avg_faithfulness', 1.0) if (prev and isinstance(faith_val, (int, float)) and isinstance(prev.get('avg_faithfulness'), (int, float))) else None)
                     
    make_metric_card(c4, "p95 Latency", f"{latest.get('p95_latency_sec', 0.0):.2f}s",
                     "SLA limit: < 3.5s",
                     latest.get('p95_latency_sec', 0.0) - prev.get('p95_latency_sec', 0.0) if prev else None, inverse=True)

    st.divider()
    
    # Metadata summary
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Evaluation Metadata Envelope")
        
        # Display rich metadata envelope if present, else fallback
        meta = latest.get("evaluation_metadata", {})
        if meta:
            st.json(meta)
        else:
            st.json({
                "timestamp": latest.get("timestamp"),
                "commit_sha": latest.get("git_commit_hash"),
                "branch": latest.get("branch"),
                "mode": latest.get("mode"),
                "embedding_model": latest.get("embedding_model"),
                "llm_model": latest.get("llm_model"),
                "judge_enabled": latest.get("judge_enabled", True),
                "cache_hit_rate": latest.get("cache_hit_rate", 0.0),
                "cached_queries_count": latest.get("cached_queries_count", 0)
            })
    with col_b:
        st.subheader("Failure Classification Breakdown")
        df_q = pd.DataFrame(latest.get("results", []))
        if "failure_category" in df_q and not df_q[df_q["status"] == "FAIL"].empty:
            fails = df_q[df_q["status"] == "FAIL"]
            fail_counts = fails["failure_category"].value_counts().reset_index()
            fail_counts.columns = ["Failure Reason", "Count"]
            fig = px.pie(fail_counts, names="Failure Reason", values="Count", color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, width="stretch")
        else:
            st.success("All queries passed successfully in this run!")


# ══════════════════════════════════════════════════════════
# PAGE 2: Metric Trends
# ══════════════════════════════════════════════════════════
elif nav == "Metric Trends":
    st.title("Platform Metric Trends")
    st.caption("Visualizing performance regressions and QA improvements across multiple commit SHAs.")
    st.divider()

    raw_list = history if history else runs
    trend_data = []
    for r in raw_list:
        # Normalize hit rate to percentage
        hit_rate = r.get("avg_retrieval_hit_rate", 1.0)
        if hit_rate <= 1.0:
            hit_rate *= 100.0
            
        # Normalize faithfulness to percentage
        faithfulness = r.get("avg_faithfulness", 1.0)
        if isinstance(faithfulness, (int, float)):
            if faithfulness <= 1.0:
                faithfulness *= 100.0
        else:
            # Skip strings from history graphing
            faithfulness = None

        trend_data.append({
            "timestamp": r.get("timestamp", ""),
            "pass_rate": r.get("pass_rate", 0.0),
            "avg_faithfulness": faithfulness,
            "avg_correctness": r.get("avg_correctness", 0.0),
            "p95_latency_sec": r.get("p95_latency_sec", 0.0),
            "avg_retrieval_hit_rate": hit_rate,
            "avg_cost_usd": r.get("avg_cost_usd", 0.0)
        })

    if len(trend_data) > 1:
        df_trend = pd.DataFrame(trend_data)
        
        t1, t2 = st.tabs(["Quality Metrics", "Performance Metrics"])
        with t1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_trend["timestamp"], y=df_trend["pass_rate"], name="Pass Rate (%)", line=dict(color="#6366f1", width=3)))
            # Drop None values before plotting
            df_faith = df_trend.dropna(subset=["avg_faithfulness"])
            if not df_faith.empty:
                fig.add_trace(go.Scatter(x=df_faith["timestamp"], y=df_faith["avg_faithfulness"], name="Faithfulness (%)", line=dict(color="#10b981", dash="dash")))
            fig.add_trace(go.Scatter(x=df_trend["timestamp"], y=df_trend["avg_retrieval_hit_rate"], name="Retrieval Hit Rate (%)", line=dict(color="#f59e0b", dash="dot")))
            fig.update_layout(height=400, template="plotly_dark", xaxis_title="Run Timestamp", yaxis_title="Percentage (%)")
            st.plotly_chart(fig, width="stretch")
        with t2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_trend["timestamp"], y=df_trend["p95_latency_sec"], name="p95 Latency (s)", line=dict(color="#ef4444", width=3)))
            fig2.add_trace(go.Scatter(x=df_trend["timestamp"], y=df_trend["avg_cost_usd"]*1000, name="Avg Cost x1000 (USD)", line=dict(color="#3b82f6", dash="dot")))
            fig2.update_layout(height=400, template="plotly_dark", xaxis_title="Run Timestamp")
            st.plotly_chart(fig2, width="stretch")
    else:
        st.info("Execute multiple evaluation runs to generate historical trend analytics.")


# ══════════════════════════════════════════════════════════
# PAGE 3: Regression Analysis
# ══════════════════════════════════════════════════════════
elif nav == "Regression Analysis":
    st.title("Regression & Drift Analysis")
    st.caption("Compare two execution runs side-by-side to identify semantic regressions.")
    st.divider()

    if len(runs) < 2:
        st.warning("At least two evaluation runs are required to run comparative regression checks.")
        st.stop()

    run_opts = [f"{r.get('timestamp')} ({r.get('git_commit_hash', 'unknown')})" for r in runs]
    
    col_a, col_b = st.columns(2)
    idx_a = col_a.selectbox("Baseline Run (A)", range(len(run_opts)), index=max(0, len(run_opts)-2), format_func=lambda idx: run_opts[idx])
    idx_b = col_b.selectbox("Candidate Run (B)", range(len(run_opts)), index=len(run_opts)-1, format_func=lambda idx: run_opts[idx])
    
    run_a, run_b = runs[idx_a], runs[idx_b]

    # Metrics matrix comparison
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
        status = "improved" if is_good else ("regressed" if abs(diff) > 0.001 else "unchanged")
        comp_rows.append({"Metric": name, "Baseline (A)": va, "Candidate (B)": vb, "Delta": f"{'+' if is_pos else ''}{diff:.4f}", "Status": status})

    df_comp = pd.DataFrame(comp_rows)
    st.dataframe(df_comp.style.map(lambda status: "color: #10b981;" if status == "improved" else ("color: #ef4444;" if status == "regressed" else ""), subset=["Status"]), width="stretch")

    # Group by category and compute metrics comparison
    st.subheader("Category-Level Metrics Comparison")
    df_a = pd.DataFrame(run_a.get("results", []))
    df_b = pd.DataFrame(run_b.get("results", []))

    if not df_a.empty and not df_b.empty:
        cat_stats_a = df_a.groupby("category").apply(lambda x: pd.Series({
            "Pass Rate (A)": f"{round((x['status'] == 'PASS').mean()*100, 1)}%",
            "Hit Rate (A)": f"{round(x['hit_rate'].mean()*100, 1)}%",
            "Latency (A)": f"{round(x['latency_sec'].mean(), 2)}s"
        })).reset_index()

        cat_stats_b = df_b.groupby("category").apply(lambda x: pd.Series({
            "Pass Rate (B)": f"{round((x['status'] == 'PASS').mean()*100, 1)}%",
            "Hit Rate (B)": f"{round(x['hit_rate'].mean()*100, 1)}%",
            "Latency (B)": f"{round(x['latency_sec'].mean(), 2)}s"
        })).reset_index()

        cat_comparison = pd.merge(cat_stats_a, cat_stats_b, on="category")
        st.dataframe(cat_comparison, width="stretch")

    # Detailed regressions list (Regression Explorer)
    st.subheader("Regression Explorer & Degradation Triage")
    q_a = {r["question"]: r for r in run_a.get("results", [])}
    q_b = {r["question"]: r for r in run_b.get("results", [])}
    
    regressions, improvements = [], []
    for q, rb in q_b.items():
        ra = q_a.get(q)
        if ra:
            sa, sb = ra.get("status", "PASS"), rb.get("status", "PASS")
            if sa == "PASS" and sb == "FAIL":
                regressions.append({
                    "Unique ID": rb.get("unique_id"),
                    "Question": q,
                    "Category": rb.get("category"),
                    "Failure Category": rb.get("failure_category", "unknown"),
                    "Attribution Reason": rb.get("attribution_reason", "No reason recorded."),
                    "Baseline sim": ra.get("semantic_similarity"),
                    "Candidate sim": rb.get("semantic_similarity")
                })
            elif sa == "FAIL" and sb == "PASS":
                improvements.append({
                    "Unique ID": rb.get("unique_id"),
                    "Question": q,
                    "Category": rb.get("category"),
                    "Baseline sim": ra.get("semantic_similarity"),
                    "Candidate sim": rb.get("semantic_similarity")
                })

    t_reg, t_imp = st.tabs([f"Regressions ({len(regressions)})", f"Improvements ({len(improvements)})"])
    with t_reg:
        if regressions:
            st.dataframe(pd.DataFrame(regressions), width="stretch")
        else:
            st.success("No code state regressions detected!")
    with t_imp:
        if improvements:
            st.dataframe(pd.DataFrame(improvements), width="stretch")
        else:
            st.info("No newly passing cases found.")


# ══════════════════════════════════════════════════════════
# PAGE 4: Failure Explorer
# ══════════════════════════════════════════════════════════
elif nav == "Failure Explorer":
    st.title("Failure & Debug Explorer")
    st.caption("Inspect and debug failures. Every record stores complete context parameters for full reproduction.")
    st.divider()

    # Honest metrics warning
    if latest.get("judge_enabled") is False:
        st.warning("⚠️ LLM Judge Metrics (Faithfulness, Hallucination) are disabled for this run (bypassed in smoke test).")

    df_q = pd.DataFrame(latest.get("results", []))
    fails = df_q[df_q["status"] == "FAIL"]
    
    if fails.empty:
        st.success("All queries passed in the latest execution run!")
    else:
        tab_summary, tab_trace, tab_charts = st.tabs(["Active Failures Grid", "Diagnostic Trace", "Failure Distributions"])
        
        with tab_summary:
            st.subheader("Failed Queries")
            df_fails_summary = fails[["unique_id", "question", "category", "failure_category", "attribution_reason", "correctness", "faithfulness", "latency_sec"]]
            st.dataframe(df_fails_summary, width="stretch")
            
        with tab_trace:
            st.subheader("Failure Trace Diagnostic")
            selected_id = st.selectbox("Select query to reproduce/inspect", fails["unique_id"].tolist())
            row = fails[fails["unique_id"] == selected_id].iloc[0]
            
            st.markdown(f"#### Question: {row['question']}")
            
            c1, c2 = st.columns(2)
            c1.warning(f"**Failure Category:** {row.get('failure_category', 'unknown')}")
            c2.info(f"**LLM Grader Reason:** {row.get('judge_reasoning', 'unknown')}")
            
            # Retrieval/Generation Diagnosis Flags
            diag = row.get("retrieval_diagnosis", {})
            if diag:
                st.markdown("### Telemetry Triage Indicators")
                cd1, cd2, cd3, cd4 = st.columns(4)
                cd1.markdown(f"**Correct Context Retrieved?** {'✅' if diag.get('context_retrieved', False) else '❌'}")
                cd2.markdown(f"**Retrieved Context Sufficient?** {'✅' if diag.get('context_sufficient', False) else '❌'}")
                cd3.markdown(f"**Model Utilized Context?** {'✅' if diag.get('model_used_context', False) else '❌'}")
                cd4.markdown(f"**Model Hallucinated?** {'❌' if not diag.get('model_hallucinated', False) else '✅'}")
            
            st.markdown("**Expected Ground Truth:**")
            st.code(row["ground_truth"])
            st.markdown("**Generated Answer:**")
            st.code(row["answer"])
            
            with st.expander("Show Retrieved Context Chunks"):
                for i, chunk in enumerate(row.get("retrieved_chunks", [])):
                    source = row.get("retrieved_sources", ["unknown"] * (i+1))[i]
                    sim = row.get("retrieved_similarities", [0.0] * (i+1))[i]
                    st.markdown(f"**Chunk {i+1} [Source: {source} | similarity: {sim}]:**")
                    st.write(chunk)
                    st.divider()

        with tab_charts:
            st.subheader("Aggregated Failure Classification")
            fail_counts = fails["failure_category"].value_counts().reset_index()
            fail_counts.columns = ["Failure Category", "Count"]
            fig_fail = px.bar(fail_counts, x="Failure Category", y="Count", color="Failure Category", template="plotly_dark")
            st.plotly_chart(fig_fail, width="stretch")


# ══════════════════════════════════════════════════════════
# PAGE 5: Retrieval Inspector
# ══════════════════════════════════════════════════════════
elif nav == "Retrieval Inspector":
    st.title("Retrieval Inspector & Diagnostics")
    st.caption("Trace document retrievals, document hit rates, and reciprocity scores.")
    st.divider()

    df_q = pd.DataFrame(latest.get("results", []))
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Hit Rate", f"{round(df_q['hit_rate'].mean()*100, 1)}%")
    c2.metric("Mean Reciprocal Rank (MRR)", f"{df_q['mrr'].mean():.3f}")
    c3.metric("Average Context Recall", f"{df_q['context_recall'].mean():.3f}")

    st.subheader("Retrieval Trace Visualizer")
    q_selected = st.selectbox("Select question to inspect context retrieval", df_q["question"].tolist())
    row = df_q[df_q["question"] == q_selected].iloc[0]
    
    st.markdown(f"**Question:** `{row['question']}`")
    st.markdown(f"**Expected Source Document:** `{row.get('expected_sources', 'unknown')}`")
    
    st.markdown("#### Retrieved Chunks Flow")
    for idx, chunk in enumerate(row.get("retrieved_chunks", [])):
        source = row.get("retrieved_sources", ["unknown"] * (idx+1))[idx]
        similarity = row.get("retrieved_similarities", [0.0] * (idx+1))[idx]
        
        st.markdown(f"""
        *   **Chunk {idx+1} [Similarity: {similarity} | File: {source}]**:
            > {chunk}
        """)
        st.divider()


# ══════════════════════════════════════════════════════════
# PAGE 6: Cost Analytics
# ══════════════════════════════════════════════════════════
elif nav == "Cost Analytics":
    st.title("Token Cost & Billing Analytics")
    st.caption("Detailed pricing metrics, costing categories, and commit costs.")
    st.divider()

    df_q = pd.DataFrame(latest.get("results", []))
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Execution Cost", f"${latest.get('total_cost_usd', 0.0):.6f}")
    c2.metric("Average Cost/Query", f"${latest.get('avg_cost_usd', 0.0):.6f}")
    c3.metric("Total Tokens Transferred", f"{df_q['prompt_tokens'].sum() + df_q['completion_tokens'].sum()}")

    st.subheader("Cost by Question Category")
    cat_cost = df_q.groupby("category")["cost_usd"].sum().reset_index()
    fig = px.bar(cat_cost, x="category", y="cost_usd", title="Total cost per category", color="category", template="plotly_dark")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Most Expensive Queries")
    st.dataframe(df_q.sort_values(by="cost_usd", ascending=False)[["unique_id", "question", "category", "prompt_tokens", "completion_tokens", "cost_usd"]].head(10), width="stretch")


# ══════════════════════════════════════════════════════════
# PAGE 7: Latency Analytics
# ══════════════════════════════════════════════════════════
elif nav == "Latency Analytics":
    st.title("Latency SLA Percentiles")
    st.caption("Analyzing box plots, median, p95/p99 execution speeds.")
    st.divider()

    df_q = pd.DataFrame(latest.get("results", []))
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average Latency", f"{df_q['latency_sec'].mean():.2f}s")
    c2.metric("Median (p50) Latency", f"{latest.get('p50_latency_sec', 0.0):.2f}s")
    c3.metric("p95 SLA Speed", f"{latest.get('p95_latency_sec', 0.0):.2f}s")
    c4.metric("p99 Outlier Speed", f"{latest.get('p99_latency_sec', 0.0):.2f}s")

    st.subheader("Latency SLA Exceedances")
    df_q["sla_violation"] = df_q["latency_sec"] > config.THRESHOLD_P95_LATENCY
    violators = df_q[df_q["sla_violation"]]
    if not violators.empty:
        st.error(f"⚠️ Detected {len(violators)} queries violating the {config.THRESHOLD_P95_LATENCY}s SLA.")
        st.dataframe(violators[["unique_id", "question", "category", "latency_sec"]], width="stretch")
    else:
        st.success("All queries satisfy the latency SLA constraints.")

    st.subheader("Latency Distribution Histogram")
    fig = px.histogram(df_q, x="latency_sec", nbins=20, title="Distribution of Query Processing Speed", template="plotly_dark", color_discrete_sequence=["#10b981"])
    st.plotly_chart(fig, width="stretch")

    st.subheader("Slowest Queries")
    st.dataframe(df_q.sort_values(by="latency_sec", ascending=False)[["unique_id", "question", "category", "latency_sec"]].head(10), width="stretch")


# ══════════════════════════════════════════════════════════
# PAGE 8: Prompt Playground
# ══════════════════════════════════════════════════════════
elif nav == "Prompt Playground":
    st.title("Interactive Prompt Sandbox")
    st.caption("Test parameters, customize retrieval settings, and compare outputs live.")
    st.divider()

    col_cfg, col_res = st.columns([2, 3])
    
    with col_cfg:
        st.subheader("Parameters")
        system_prompt = st.text_area("System Prompt Template", value=(
            "You are a helpful Indian income tax assistant. "
            "Answer the question using ONLY the context below.\n"
            "If the answer is not in the context, say \"I don't have information about that.\""
        ), height=140)
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
        top_k = st.slider("Retrieval top_k Chunks", 1, 5, 3)
        
        test_q = st.text_input("Test Question", value="What is the maximum deduction under Section 80C?")
        test_gt = st.text_area("Expected Answer (Ground Truth)", value="The maximum deduction under Section 80C is Rs 1.5 lakh per financial year.")
        
        execute_sandbox = st.button("Run Sandbox Execution", width="stretch")

    with col_res:
        st.subheader("Inference Diagnostic Report")
        if execute_sandbox and test_q:
            with st.spinner("Executing pipeline context lookup..."):
                from core.retrieval import load_docs, build_vector_store, retrieve
                from core.generator import generate_answer, calculate_cost
                from core.metrics import compute_semantic_similarity
                from core.judge import llm_judge_evaluate
                
                chunks = load_docs()
                # Run locally
                collection = build_vector_store(chunks, "playground_sandbox")
                
                start = time.time()
                retrieved_chunks, similarities, sources = retrieve(test_q, collection, top_k=top_k)
                answer, p_tok, c_tok = generate_answer(test_q, retrieved_chunks, system_prompt, temperature)
                latency = round(time.time() - start, 2)
                cost = calculate_cost(p_tok, c_tok)
                
            st.success("Generation Complete!")
            st.markdown("**Generated Answer:**")
            st.info(answer)
            
            st.markdown("**Metrics Evaluation:**")
            sim = compute_semantic_similarity(answer, test_gt)
            st.write(f"- Semantic Cosine Similarity: `{sim}`")
            st.write(f"- Prompt Tokens / Completion Tokens: `{p_tok} / {c_tok}`")
            st.write(f"- Latency: `{latency}s` | Cost: `${cost:.6f}`")
            
            with st.expander("Show Retrieved Context Chunks"):
                for idx, chunk in enumerate(retrieved_chunks):
                    st.markdown(f"**Chunk {idx+1} [Source: {sources[idx]} | Similarity: {similarities[idx]}]:**")
                    st.write(chunk)
        else:
            st.info("Input configurations and click 'Run Sandbox Execution' to test outputs.")


# ══════════════════════════════════════════════════════════
# PAGE 9: Dataset Explorer
# ══════════════════════════════════════════════════════════
elif nav == "Dataset Explorer":
    st.title("Dataset Explorer")
    st.caption("Benchmark suite questions containing structured tagging, versions, and citations metadata.")
    st.divider()

    if os.path.exists("golden_dataset.csv"):
        df_ds = pd.read_csv("golden_dataset.csv")
    elif os.path.exists("../golden_dataset.csv"):
        df_ds = pd.read_csv("../golden_dataset.csv")
    else:
        st.warning("Golden dataset not found.")
        st.stop()

    st.subheader("Filter Suite")
    f_cat = st.selectbox("Filter Category", ["All"] + df_ds["category"].unique().tolist())
    f_diff = st.selectbox("Filter Difficulty", ["All"] + df_ds["difficulty"].unique().tolist())
    
    filtered_ds = df_ds.copy()
    if f_cat != "All":
        filtered_ds = filtered_ds[filtered_ds["category"] == f_cat]
    if f_diff != "All":
        filtered_ds = filtered_ds[filtered_ds["difficulty"] == f_diff]

    st.dataframe(filtered_ds[["unique_id", "category", "difficulty", "tags", "expected_sources", "reasoning_type", "question", "ground_truth"]], width="stretch")

