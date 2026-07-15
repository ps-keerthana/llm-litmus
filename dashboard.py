import streamlit as st
import json
import glob
import os
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ── Page Configuration ─────────────────────────────────────
st.set_page_config(
    page_title="LLM Eval CI/CD Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        color: white;
    }
    .metric-container:hover {
        transform: translateY(-4px);
        border-color: #6366f1;
        box-shadow: 0 20px 25px -5px rgba(99, 102, 241, 0.15);
    }
    .metric-label {
        font-size: 13px; color: #94a3b8; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;
    }
    .metric-val { font-size: 32px; font-weight: 700; margin-bottom: 4px; }
    .metric-sub { font-size: 12px; color: #64748b; }
    .header-badge {
        background-color: #312e81; color: #c7d2fe;
        padding: 4px 12px; border-radius: 9999px;
        font-size: 12px; font-weight: 600;
        display: inline-block; margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load runs ──────────────────────────────────────────────
def load_all_runs():
    files = sorted(glob.glob("eval_results/run_*.json"))
    runs = []
    for f in files:
        try:
            with open(f, "r") as fp:
                data = json.load(fp)
            runs.append({
                "timestamp": data.get("run_timestamp", ""),
                "commit_sha": data.get("commit_sha", "unknown"),
                "pass_rate": data.get("pass_rate", 0.0),
                "hallucination": data.get("hallucination_rate_avg", 0.0),
                "faithfulness": data.get("avg_faithfulness", 0.0),
                "semantic_similarity": data.get("avg_semantic_similarity", 0.0),
                "relevancy": data.get("avg_relevancy", 0.0),
                "latency": data.get("avg_latency_sec", 0.0),
                "p50_latency": data.get("p50_latency_sec", 0.0),
                "p95_latency": data.get("p95_latency_sec", 0.0),
                "total_cost": data.get("total_cost_usd", 0.0),
                "avg_cost": data.get("avg_cost_usd", 0.0),
                "passed": data.get("passed", 0),
                "failed": data.get("failed", 0),
                "total": data.get("total_questions", 0),
                "results": data.get("results", []),
                "file": f
            })
        except Exception as e:
            st.warning(f"Could not load {f}: {e}")
    return runs

# Also load metrics_history.json for trend data
def load_metrics_history():
    if os.path.exists("metrics_history.json"):
        try:
            with open("metrics_history.json", "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

runs = load_all_runs()
history = load_metrics_history()

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.markdown("<h2 style='text-align: center;'>Indian Tax RAG Eval</h2>", unsafe_allow_html=True)
st.sidebar.divider()

nav = st.sidebar.radio("Navigation", [
    "Run History & Trends",
    "Run Comparison",
    "Prompt Playground"
])

st.sidebar.divider()
dataset_count = len(pd.read_csv("golden_dataset.csv")) if os.path.exists("golden_dataset.csv") else 0
st.sidebar.markdown(f"**Domain:** Indian Income Tax")
st.sidebar.markdown(f"**Golden Dataset:** `{dataset_count} pairs`")
st.sidebar.markdown(f"**Total Runs:** `{len(runs)}`")
if runs:
    st.sidebar.caption(f"Latest: `{runs[-1]['timestamp']}` (`{runs[-1]['commit_sha']}`)")

if not runs:
    st.warning("No evaluation runs found. Run `python evaluate_dataset.py` first.")
    st.stop()

latest = runs[-1]

# ── Helper ────────────────────────────────────────────────
def metric_card(col, label, val, sub, delta=None, inverse=False):
    delta_html = ""
    if delta is not None:
        is_pos = delta > 0
        is_good = (not is_pos) if inverse else is_pos
        color = "#10b981" if is_good else "#ef4444"
        sym = "+" if is_pos else ""
        delta_html = f"<span style='font-weight:600;color:{color};'> ({sym}{delta:.2f})</span>"

    col.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-val">{val}</div>
        <div class="metric-sub">{sub}{delta_html}</div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 1: Run History & Trends
# ══════════════════════════════════════════════════════════
if nav == "Run History & Trends":
    st.markdown("<div class='header-badge'>Active Run Summary</div>", unsafe_allow_html=True)
    st.title("Run History & Trends")
    st.caption("Tracking semantic quality, hallucination rates, costs, and latency percentiles across commits.")
    st.divider()

    prev = runs[-2] if len(runs) > 1 else None
    c1, c2, c3, c4 = st.columns(4)

    metric_card(c1, "Pass Rate", f"{latest['pass_rate']}%",
                f"{latest['passed']}/{latest['total']} passed",
                latest['pass_rate'] - prev['pass_rate'] if prev else None)
    metric_card(c2, "Avg Faithfulness", f"{latest['faithfulness']:.3f}",
                "1.0 = fully grounded",
                latest['faithfulness'] - prev['faithfulness'] if prev else None)
    metric_card(c3, "p95 Latency", f"{latest['p95_latency']:.2f}s",
                "SLA target: < 3.5s",
                latest['p95_latency'] - prev['p95_latency'] if prev else None, inverse=True)
    metric_card(c4, "Avg Query Cost", f"${latest['avg_cost']:.5f}",
                f"Total: ${latest['total_cost']:.4f}",
                latest['avg_cost'] - prev['avg_cost'] if prev else None, inverse=True)

    st.divider()

    # Trend charts from metrics_history.json or runs
    trend_data = history if history else [{
        "timestamp": r["timestamp"], "pass_rate": r["pass_rate"],
        "avg_faithfulness": r["faithfulness"],
        "avg_semantic_similarity": r["semantic_similarity"],
        "p50_latency_sec": r["p50_latency"], "p95_latency_sec": r["p95_latency"],
        "avg_cost_usd": r["avg_cost"]
    } for r in runs]

    if len(trend_data) > 1:
        st.subheader("Historical Trends")
        df_trend = pd.DataFrame(trend_data)

        t1, t2 = st.tabs(["Quality Metrics", "Performance & Cost"])

        with t1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_trend["timestamp"], y=df_trend["pass_rate"],
                                     name="Pass Rate (%)", line=dict(color="#6366f1", width=3), marker=dict(size=8)))
            if "avg_faithfulness" in df_trend:
                fig.add_trace(go.Scatter(x=df_trend["timestamp"], y=df_trend["avg_faithfulness"].apply(lambda x: x*100),
                                         name="Faithfulness (%)", line=dict(color="#10b981", dash="dash")))
            if "avg_semantic_similarity" in df_trend:
                fig.add_trace(go.Scatter(x=df_trend["timestamp"], y=df_trend["avg_semantic_similarity"].apply(lambda x: x*100),
                                         name="Semantic Similarity (%)", line=dict(color="#f59e0b", dash="dot")))
            fig.update_layout(title="Quality Metrics Over Runs", height=380, template="plotly_dark",
                              xaxis_title="Run", yaxis_title="Percentage (%)")
            st.plotly_chart(fig, use_container_width=True)

        with t2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_trend["timestamp"], y=df_trend["p95_latency_sec"],
                                      name="p95 Latency", line=dict(color="#ef4444", width=3)))
            fig2.add_trace(go.Scatter(x=df_trend["timestamp"], y=df_trend["p50_latency_sec"],
                                      name="p50 Latency", line=dict(color="#3b82f6", width=2)))
            fig2.add_hline(y=3.5, line_dash="dash", line_color="red", annotation_text="SLA (3.5s)")
            fig2.update_layout(title="Latency SLA Tracking", height=380, template="plotly_dark",
                              xaxis_title="Run", yaxis_title="Seconds")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Run the eval pipeline multiple times to see historical trends.")

    st.divider()

    # Question breakdown
    st.subheader("Question Breakdown (Latest Run)")

    rows = [{
        "Status": r.get("status", ""), "Question": r["question"],
        "Category": r["category"], "Difficulty": r["difficulty"],
        "Semantic Sim": r.get("semantic_similarity", 0.0),
        "Faithfulness": r.get("faithfulness", 1.0),
        "Latency (s)": r.get("latency_sec", 0.0),
        "Cost ($)": r.get("cost_usd", 0.0),
        "Answer": r["answer"], "Ground Truth": r["ground_truth"],
        "Faithfulness Reasoning": r.get("faithfulness_reasoning", "N/A")
    } for r in latest["results"]]
    df_q = pd.DataFrame(rows)

    f1, f2, f3 = st.columns(3)
    sf = f1.selectbox("Status", ["All", "PASS", "FAIL"])
    cf = f2.selectbox("Category", ["All"] + sorted(df_q["Category"].unique().tolist()))
    df_filt = f3.selectbox("Difficulty", ["All"] + sorted(df_q["Difficulty"].unique().tolist()))

    filtered = df_q.copy()
    if sf != "All": filtered = filtered[filtered["Status"] == sf]
    if cf != "All": filtered = filtered[filtered["Category"] == cf]
    if df_filt != "All": filtered = filtered[filtered["Difficulty"] == df_filt]

    def color_status(val):
        if val == "PASS": return "background-color: #065f46; color: #a7f3d0"
        if val == "FAIL": return "background-color: #7f1d1d; color: #fca5a5"
        return ""

    st.dataframe(
        filtered[["Status", "Question", "Category", "Difficulty", "Semantic Sim", "Faithfulness", "Latency (s)", "Cost ($)"]].style.map(color_status, subset=["Status"]),
        use_container_width=True, height=350
    )

    # Failure analysis
    st.subheader("Failure Analysis")
    fails = filtered[filtered["Status"] == "FAIL"]
    if fails.empty:
        st.success("No failures for the selected filters!")
    else:
        # Category breakdown
        col1, col2 = st.columns(2)
        with col1:
            cat_counts = fails["Category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Failures"]
            fig3 = px.bar(cat_counts, x="Category", y="Failures", title="Failures by Category",
                          color_discrete_sequence=["#D85A30"])
            fig3.update_layout(height=280)
            st.plotly_chart(fig3, use_container_width=True)
        with col2:
            diff_counts = fails["Difficulty"].value_counts().reset_index()
            diff_counts.columns = ["Difficulty", "Failures"]
            fig4 = px.pie(diff_counts, names="Difficulty", values="Failures", title="Failures by Difficulty",
                          color_discrete_sequence=["#534AB7", "#D85A30", "#1D9E75"])
            fig4.update_layout(height=280)
            st.plotly_chart(fig4, use_container_width=True)

        for _, row in fails.iterrows():
            with st.expander(f"FAIL: {row['Question']}"):
                st.markdown(f"**Expected:** {row['Ground Truth']}")
                st.markdown(f"**Got:** {row['Answer']}")
                st.markdown(f"**Sim:** {row['Semantic Sim']} | **Faithfulness:** {row['Faithfulness']}")
                st.markdown(f"**Judge reasoning:** {row['Faithfulness Reasoning']}")


# ══════════════════════════════════════════════════════════
# PAGE 2: Run Comparison
# ══════════════════════════════════════════════════════════
elif nav == "Run Comparison":
    st.markdown("<div class='header-badge'>Regression Checker</div>", unsafe_allow_html=True)
    st.title("Run Comparison Matrix")
    st.caption("Compare two runs side-by-side to catch drift, cost increases, and latency regressions.")
    st.divider()

    if len(runs) < 2:
        st.warning("Need at least 2 runs for comparison.")
        st.stop()

    run_opts = [f"{r['timestamp']} ({r['commit_sha']})" for r in runs]
    c1, c2 = st.columns(2)
    idx_a = c1.selectbox("Baseline (A)", range(len(run_opts)), index=max(0, len(run_opts)-2), format_func=lambda i: run_opts[i])
    idx_b = c2.selectbox("Candidate (B)", range(len(run_opts)), index=len(run_opts)-1, format_func=lambda i: run_opts[i])
    run_a, run_b = runs[idx_a], runs[idx_b]

    # Comparison table
    metrics = [
        ("Pass Rate", f"{run_a['pass_rate']}%", f"{run_b['pass_rate']}%", run_b['pass_rate'] - run_a['pass_rate'], False),
        ("Avg Faithfulness", f"{run_a['faithfulness']:.3f}", f"{run_b['faithfulness']:.3f}", run_b['faithfulness'] - run_a['faithfulness'], False),
        ("Hallucination Rate", f"{run_a['hallucination']:.3f}", f"{run_b['hallucination']:.3f}", run_b['hallucination'] - run_a['hallucination'], True),
        ("p95 Latency", f"{run_a['p95_latency']:.2f}s", f"{run_b['p95_latency']:.2f}s", run_b['p95_latency'] - run_a['p95_latency'], True),
        ("Avg Cost", f"${run_a['avg_cost']:.6f}", f"${run_b['avg_cost']:.6f}", run_b['avg_cost'] - run_a['avg_cost'], True),
    ]

    comp_rows = []
    for name, va, vb, diff, inverse in metrics:
        is_pos = diff > 0
        is_good = (not is_pos) if inverse else is_pos
        indicator = "improved" if is_good else ("regressed" if abs(diff) > 0.001 else "unchanged")
        comp_rows.append({"Metric": name, "Baseline (A)": va, "Candidate (B)": vb, "Change": f"{'+' if is_pos else ''}{diff:.4f}", "Status": indicator})

    df_comp = pd.DataFrame(comp_rows)

    def color_change(val):
        if val == "improved": return "color: #10b981; font-weight: 600;"
        if val == "regressed": return "color: #ef4444; font-weight: 600;"
        return ""

    st.dataframe(df_comp.style.map(color_change, subset=["Status"]), use_container_width=True)
    st.divider()

    # Question-level regression analysis
    st.subheader("Question-Level Diffs")
    q_a = {r["question"]: r for r in run_a["results"]}
    q_b = {r["question"]: r for r in run_b["results"]}

    regressions, improvements = [], []
    for q, rb in q_b.items():
        ra = q_a.get(q)
        if ra:
            sa, sb = ra.get("status", "PASS"), rb.get("status", "PASS")
            if sa == "PASS" and sb == "FAIL":
                regressions.append({"Question": q, "A Answer": ra["answer"][:80], "B Answer": rb["answer"][:80],
                                    "A Sim": ra.get("semantic_similarity", 0), "B Sim": rb.get("semantic_similarity", 0)})
            elif sa == "FAIL" and sb == "PASS":
                improvements.append({"Question": q, "A Answer": ra["answer"][:80], "B Answer": rb["answer"][:80],
                                     "A Sim": ra.get("semantic_similarity", 0), "B Sim": rb.get("semantic_similarity", 0)})

    tab1, tab2 = st.tabs([f"Regressions ({len(regressions)})", f"Improvements ({len(improvements)})"])
    with tab1:
        if not regressions:
            st.success("No regressions! Candidate B didn't break any passing baselines.")
        else:
            st.dataframe(pd.DataFrame(regressions), use_container_width=True)
    with tab2:
        if not improvements:
            st.info("No newly passing cases compared to baseline.")
        else:
            st.dataframe(pd.DataFrame(improvements), use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 3: Prompt Playground
# ══════════════════════════════════════════════════════════
elif nav == "Prompt Playground":
    st.markdown("<div class='header-badge'>Interactive Sandbox</div>", unsafe_allow_html=True)
    st.title("RAG Prompt Playground")
    st.caption("Test prompt changes, adjust retrieval parameters, and evaluate answers in real-time.")
    st.divider()

    col_cfg, col_out = st.columns([2, 3])

    with col_cfg:
        st.subheader("Configuration")
        system_prompt = st.text_area("System Prompt", value=(
            'You are a helpful Indian income tax assistant. '
            'Answer the question using ONLY the context below.\n'
            'If the answer is not in the context, say "I don\'t have information about that."'
        ), height=120)

        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
        top_k = st.slider("Context Chunks (K)", 1, 5, 3)

        st.divider()
        st.subheader("Test Query")
        test_q = st.text_input("Question", value="Can I claim both 80C and NPS deduction?")
        test_gt = st.text_area("Expected Answer (optional)", value="Yes, you can claim Rs 1.5 lakh under 80C and Rs 50,000 under 80CCD(1B) for NPS.", height=80)
        run_btn = st.button("Run Test Query", use_container_width=True)

    with col_out:
        st.subheader("Results")

        if run_btn and test_q:
            with st.spinner("Loading RAG pipeline..."):
                from rag_pipeline import (
                    load_docs, build_vector_store, retrieve, generate_answer,
                    call_groq_with_retry, calculate_cost, embedder
                )

                chunks = load_docs()
                collection = build_vector_store(chunks, "playground")

            with st.spinner("Generating answer..."):
                start = time.time()
                ctx = retrieve(test_q, collection, top_k=top_k)
                answer, p_tok, c_tok = generate_answer(test_q, ctx, system_prompt, temperature)
                latency = round(time.time() - start, 2)
                cost = calculate_cost(p_tok, c_tok)

            st.success("Done!")
            st.markdown("**Generated Answer:**")
            st.info(answer)

            with st.expander("Retrieved Context Chunks"):
                for i, c in enumerate(ctx):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(c)
                    st.divider()

            # Real-time evaluation
            st.markdown("**Real-Time Evaluation**")
            with st.spinner("LLM judge evaluating..."):
                sim = 0.0
                if test_gt:
                    embs = embedder.encode([answer, test_gt])
                    n1, n2 = np.linalg.norm(embs[0]), np.linalg.norm(embs[1])
                    sim = round(float(np.dot(embs[0], embs[1]) / (n1 * n2)), 3) if n1 > 0 and n2 > 0 else 0.0

                import json
                context_str = "\n\n".join(ctx)
                judge_prompt = f"""You are an expert evaluator. Rate relevancy and faithfulness.
Question: {test_q}
Ground Truth: {test_gt if test_gt else "N/A"}
Context: {context_str}
Answer: {answer}

Return JSON with relevancy_score (0-1), faithfulness_score (0-1), relevancy_reasoning, faithfulness_reasoning."""

                try:
                    jr = call_groq_with_retry(
                        messages=[{"role": "user", "content": judge_prompt}],
                        response_format={"type": "json_object"}, temperature=0.0
                    )
                    jd = json.loads(jr.choices[0].message.content.strip())
                    rel_score = float(jd.get("relevancy_score", 0))
                    faith_score = float(jd.get("faithfulness_score", 1))
                    rel_reason = jd.get("relevancy_reasoning", "")
                    faith_reason = jd.get("faithfulness_reasoning", "")
                except Exception:
                    rel_score, faith_score = 0.0, 1.0
                    rel_reason, faith_reason = "Error", "Error"

            m1, m2, m3 = st.columns(3)
            m1.metric("Faithfulness", f"{faith_score:.2f}")
            m2.metric("Relevancy", f"{rel_score:.2f}")
            m3.metric("Semantic Sim", f"{sim:.3f}" if test_gt else "N/A")

            m4, m5, m6 = st.columns(3)
            m4.metric("Latency", f"{latency}s")
            m5.metric("Cost", f"${cost:.6f}")
            m6.metric("Tokens", f"{p_tok + c_tok}")

            st.markdown(f"**Faithfulness:** {faith_reason}")
            if test_gt:
                st.markdown(f"**Relevancy:** {rel_reason}")
        elif not run_btn:
            st.info("Configure parameters and click 'Run Test Query' to see results.")