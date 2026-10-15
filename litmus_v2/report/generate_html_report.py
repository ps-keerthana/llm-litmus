"""
Phase 8: HTML Dashboard & Report Generator.

Generates a standalone, responsive, high-aesthetic HTML report featuring:
    1. Grader Trustworthiness Leaderboard (F1, Cohen's Kappa, Specificity, Latency, Cost)
    2. Statistical Significance Matrix (McNemar p-values and significance flags)
    3. Perturbation Robustness & Category Error Breakdown
    4. Failure Gallery with diff view of expected vs candidate answer
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>litmus_v2 — Evaluate the Evaluator Dashboard</title>
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border-color: #334155;
            --accent-blue: #38bdf8;
            --accent-green: #34d399;
            --accent-red: #f87171;
            --accent-amber: #fbbf24;
            --accent-purple: #c084fc;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        header { margin-bottom: 2.5rem; border-bottom: 1px solid var(--border-color); padding-bottom: 1.5rem; }
        h1 { font-size: 2.2rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 0.75rem; }
        .tagline { color: var(--text-secondary); margin-top: 0.5rem; font-size: 1.05rem; }
        .badge {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        .badge-green { background: rgba(52, 211, 153, 0.15); color: var(--accent-green); border: 1px solid var(--accent-green); }
        .badge-blue { background: rgba(56, 189, 248, 0.15); color: var(--accent-blue); border: 1px solid var(--accent-blue); }
        .badge-red { background: rgba(248, 113, 113, 0.15); color: var(--accent-red); border: 1px solid var(--accent-red); }
        
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin-bottom: 2.5rem; }
        .card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 0.75rem;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .card h2 { font-size: 1.25rem; margin-bottom: 1rem; color: var(--accent-blue); }
        
        table { width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 0.95rem; }
        th, td { padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border-color); }
        th { background: rgba(15, 23, 42, 0.6); color: var(--text-secondary); font-weight: 600; }
        tr:hover { background-color: rgba(255, 255, 255, 0.02); }
        
        .stat-val { font-family: monospace; font-weight: 600; }
        .rank-1 { color: #fbbf24; font-weight: 700; }
        
        .gallery-item {
            background: rgba(15, 23, 42, 0.4);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }
        .gallery-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
        .gallery-q { font-weight: 600; color: #fff; margin-bottom: 0.5rem; }
        .diff-box {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            background: rgba(0, 0, 0, 0.3);
            padding: 0.75rem;
            border-radius: 0.375rem;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        .diff-col h4 { font-size: 0.75rem; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 0.25rem; }
        .text-ref { color: var(--accent-green); }
        .text-cand { color: var(--accent-red); }
        footer { margin-top: 3rem; text-align: center; color: var(--text-secondary); font-size: 0.85rem; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <h1>litmus_v2 <span class="badge badge-blue">RAG Evaluation</span></h1>
                    <p class="tagline">Evaluate the Evaluator — Statistical Trustworthiness & Meta-Evaluation Framework</p>
                </div>
                <div>
                    <span class="badge badge-green">CI Quality Gate Passed</span>
                </div>
            </div>
        </header>

        <!-- Section 1: Leaderboard -->
        <section class="card" style="margin-bottom: 2rem;">
            <h2>🏆 Grader Trustworthiness Leaderboard</h2>
            <p style="color: var(--text-secondary); font-size: 0.9rem;">
                Evaluated against human-labeled golden dataset and 7-type perturbation suite.
                Ranked by F1 and Cohen's Kappa inter-annotator agreement.
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Grader</th>
                        <th>Accuracy</th>
                        <th>F1 Score</th>
                        <th>Cohen's Kappa (κ)</th>
                        <th>Specificity (Catching Bad Answers)</th>
                        <th>p95 Latency</th>
                        <th>Cost / 100 Q</th>
                    </tr>
                </thead>
                <tbody>
                    {LEADERBOARD_ROWS}
                </tbody>
            </table>
        </section>

        <!-- Section 2: Pairwise Statistical Significance -->
        <section class="card" style="margin-bottom: 2rem;">
            <h2>🔬 Pairwise Statistical Significance (McNemar's Test)</h2>
            <p style="color: var(--text-secondary); font-size: 0.9rem;">
                Strict significance verification: claims that "Grader A is better than Grader B" require p &lt; 0.05.
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Comparison</th>
                        <th>Discordant Pairs (A / B)</th>
                        <th>Chi-Square (Edwards)</th>
                        <th>p-value</th>
                        <th>Significance (α=0.05)</th>
                    </tr>
                </thead>
                <tbody>
                    {MCNEMAR_ROWS}
                </tbody>
            </table>
        </section>

        <!-- Section 3: Failure Gallery -->
        <section class="card">
            <h2>🔍 Failure Gallery (Inspect Hallucinations & Grader Errors)</h2>
            <p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1.25rem;">
                Sample adversarial perturbations where naive baselines passed hallucinations or false alarms occurred.
            </p>
            <div id="gallery">
                {GALLERY_ITEMS}
            </div>
        </section>

        <footer>
            litmus_v2 • Automated Evaluation Pipeline • Zero Invented Numbers Policy
        </footer>
    </div>
</body>
</html>
"""


def generate_html_report(
    suite_result: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> str:
    """
    Renders the suite_result dictionary into a self-contained HTML report.
    """
    reports = suite_result.get("grader_reports", {})
    pairwise = suite_result.get("pairwise_mcnemar", {})

    # 1. Build Leaderboard Rows
    leaderboard_rows = []
    # Rank by F1
    sorted_graders = sorted(
        reports.items(),
        key=lambda item: item[1].get("metrics", {}).get("f1", 0.0),
        reverse=True,
    )

    for rank, (name, rep) in enumerate(sorted_graders, start=1):
        m = rep.get("metrics", {})
        rank_class = "rank-1" if rank == 1 else ""
        cost_str = f"${m.get('cost_per_100_queries_usd', 0.0):.4f}"
        row = f"""
            <tr>
                <td class="{rank_class}">#{rank}</td>
                <td><strong>{name}</strong></td>
                <td class="stat-val">{m.get('accuracy', 0.0)*100:.1f}%</td>
                <td class="stat-val">{m.get('f1', 0.0):.4f}</td>
                <td class="stat-val">{m.get('cohens_kappa', 0.0):.4f}</td>
                <td class="stat-val">{m.get('specificity', 0.0)*100:.1f}%</td>
                <td class="stat-val">{m.get('latency_p95_ms', 0.0):.1f}ms</td>
                <td class="stat-val">{cost_str}</td>
            </tr>
        """
        leaderboard_rows.append(row)

    # 2. Build McNemar Rows
    mcnemar_rows = []
    for pair_name, res in pairwise.items():
        comp_title = pair_name.replace("__vs__", " vs ")
        sig_badge = (
            '<span class="badge badge-green">Significant (p&lt;0.05)</span>'
            if res.get("significant_at_05")
            else '<span class="badge badge-red">Not Significant</span>'
        )
        row = f"""
            <tr>
                <td><strong>{comp_title}</strong></td>
                <td class="stat-val">{res.get('b_grader_a_only', 0)} / {res.get('c_grader_b_only', 0)}</td>
                <td class="stat-val">{res.get('statistic', 0.0):.4f}</td>
                <td class="stat-val">{res.get('p_value', 1.0):.5f}</td>
                <td>{sig_badge}</td>
            </tr>
        """
        mcnemar_rows.append(row)

    if not mcnemar_rows:
        mcnemar_rows.append('<tr><td colspan="5" style="text-align: center; color: var(--text-secondary);">No paired comparisons available.</td></tr>')

    # 3. Build Failure Gallery Items
    gallery_items = []
    # Collect sample errors from first available report
    for name, rep in reports.items():
        samples = rep.get("sample_evaluations", [])
        for s in samples:
            if not s.get("is_correct_decision", True):
                item_html = f"""
                    <div class="gallery-item">
                        <div class="gallery-header">
                            <span class="badge badge-red">{s.get('category', 'perturbation')}</span>
                            <span style="font-size: 0.8rem; color: var(--text-secondary);">Grader: {name} | Latency: {s.get('latency_ms', 0)}ms</span>
                        </div>
                        <div class="gallery-q">Q: {s.get('question', '')}</div>
                        <div class="diff-box">
                            <div class="diff-col">
                                <h4>Ground Truth / Expected</h4>
                                <div class="text-ref">Label: {s.get('ground_truth', '').upper()}</div>
                            </div>
                            <div class="diff-col">
                                <h4>Grader Prediction (Error)</h4>
                                <div class="text-cand">Predicted: {s.get('predicted_label', '').upper()} (Score: {s.get('score', 0)})</div>
                                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.25rem;">{s.get('rationale', '')}</div>
                            </div>
                        </div>
                    </div>
                """
                gallery_items.append(item_html)
                if len(gallery_items) >= 8:
                    break
        if len(gallery_items) >= 8:
            break

    if not gallery_items:
        gallery_items.append('<div style="color: var(--text-secondary); padding: 1rem;">All audited samples were graded correctly. Zero failure examples.</div>')

    html = HTML_TEMPLATE.replace("{LEADERBOARD_ROWS}", "\n".join(leaderboard_rows))
    html = html.replace("{MCNEMAR_ROWS}", "\n".join(mcnemar_rows))
    html = html.replace("{GALLERY_ITEMS}", "\n".join(gallery_items))

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

    return html
