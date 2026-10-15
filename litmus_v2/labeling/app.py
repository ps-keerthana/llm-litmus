"""
Streamlit Human Labeling App — litmus_v2 Phase 2.

Usage:
    cd litmus_v2
    streamlit run labeling/app.py -- --csv data/labeling_template.csv

Hard rules enforced by this UI:
- No label is ever auto-filled. Every field starts blank.
- The app only reads/writes the CSV — it never modifies results/.
- Session state tracks which rows have been saved.
- Saving writes ONLY the currently viewed row; it does not touch other rows.
"""

import csv
import sys
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

# ── Bootstrap path so app can be run from litmus_v2/ root ────────────────────
_HERE = Path(__file__).parent
_LITMUS_V2_ROOT = _HERE.parent
if str(_LITMUS_V2_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_LITMUS_V2_ROOT.parent))

try:
    import streamlit as st
except ImportError:
    print("ERROR: streamlit is not installed. Run: pip install streamlit")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_csv_path() -> str:
    """Parses --csv argument from sys.argv (Streamlit passes through after --)."""
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg in ("--csv", "-c") and i + 1 < len(args):
            return args[i + 1]
    # Default fallback
    default = _LITMUS_V2_ROOT / "data" / "labeling_template.csv"
    return str(default)


def _load_csv(path: str) -> List[Dict]:
    """Load all rows from CSV into a list of dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _save_row(path: str, row_index: int, updated_row: Dict) -> None:
    """Overwrites a single row in the CSV by row index. Atomic write."""
    rows = _load_csv(path)
    if row_index >= len(rows):
        st.error(f"Row index {row_index} out of range ({len(rows)} rows total).")
        return
    rows[row_index].update(updated_row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _completion_stats(rows: List[Dict]) -> Dict:
    total = len(rows)
    labeled = sum(
        1 for r in rows
        if r.get("reference_correct", "").strip()
        and r.get("supported_by_chunk", "").strip()
        and r.get("labeled_by", "").strip()
    )
    return {"total": total, "labeled": labeled, "remaining": total - labeled}


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="litmus_v2 — Human Gold Set Labeler",
        page_icon="🏷️",
        layout="wide",
    )

    # ── Header ───────────────────────────────────────────────────────────────
    st.title("🏷️ litmus_v2 — Human Gold Set Labeler")
    st.caption(
        "Label each Q/A pair. Every field must be filled by a human — "
        "no auto-labeling. Your labels form the gold standard for evaluating the evaluators."
    )

    csv_path = _get_csv_path()
    csv_file = Path(csv_path)

    if not csv_file.exists():
        st.error(f"CSV not found: `{csv_path}`")
        st.info(
            "Run `python -m litmus_v2.labeling.export_csv` first to generate the template, "
            "or pass `--csv <path>` on the command line."
        )
        st.stop()

    # ── Load data ─────────────────────────────────────────────────────────────
    rows = _load_csv(csv_path)
    if not rows:
        st.warning("CSV is empty.")
        st.stop()

    stats = _completion_stats(rows)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Items", stats["total"])
    col2.metric("Labeled", stats["labeled"])
    col3.metric("Remaining", stats["remaining"])

    progress = stats["labeled"] / stats["total"] if stats["total"] > 0 else 0
    st.progress(progress, text=f"{stats['labeled']}/{stats['total']} labeled ({progress:.0%})")

    st.divider()

    # ── Row navigation ────────────────────────────────────────────────────────
    if "row_idx" not in st.session_state:
        # Start at the first unlabeled row
        first_unlabeled = next(
            (
                i for i, r in enumerate(rows)
                if not r.get("reference_correct", "").strip()
                or not r.get("labeled_by", "").strip()
            ),
            0,
        )
        st.session_state["row_idx"] = first_unlabeled

    ids = [r["id"] for r in rows]
    # Show completion marker in sidebar list
    with st.sidebar:
        st.subheader("📋 All Items")
        for i, r in enumerate(rows):
            done = (
                r.get("reference_correct", "").strip()
                and r.get("supported_by_chunk", "").strip()
                and r.get("labeled_by", "").strip()
            )
            label = f"{'✅' if done else '⬜'} [{r['question_type']}] {r['id']}"
            if st.button(label, key=f"nav_{i}", use_container_width=True):
                st.session_state["row_idx"] = i
                st.rerun()

    current_idx = st.session_state["row_idx"]
    # Reload rows each render to reflect latest saves
    rows = _load_csv(csv_path)
    row = rows[current_idx]

    # ── Navigation buttons ────────────────────────────────────────────────────
    nav_prev, nav_info, nav_next = st.columns([1, 4, 1])
    with nav_prev:
        if st.button("⬅ Previous", disabled=(current_idx == 0)):
            st.session_state["row_idx"] = current_idx - 1
            st.rerun()
    with nav_info:
        st.markdown(f"**Item {current_idx + 1} of {len(rows)}** — ID: `{row['id']}`")
    with nav_next:
        if st.button("Next ➡", disabled=(current_idx == len(rows) - 1)):
            st.session_state["row_idx"] = current_idx + 1
            st.rerun()

    st.divider()

    # ── Item display ──────────────────────────────────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        st.subheader(f"🔍 {row['id']} — `{row['question_type']}`")
        st.markdown("**Question:**")
        st.info(row["question"])

        st.markdown("**Reference Answer (candidate):**")
        st.warning(row["reference_answer"])

        if row.get("source_chunk_text"):
            with st.expander("📄 Source Chunk Text", expanded=True):
                st.text(row["source_chunk_text"])
        else:
            st.caption("*(No source chunk — unanswerable item)*")

    with right:
        st.subheader("✏️ Your Label")

        # Reviewer name
        reviewer = st.text_input(
            "Your name / reviewer ID *",
            value=row.get("labeled_by", ""),
            key=f"reviewer_{current_idx}",
            placeholder="e.g. keerthana",
        )

        # reference_correct
        ref_correct_raw = row.get("reference_correct", "").strip()
        ref_correct_options = ["(not labeled)", "True", "False"]
        if ref_correct_raw.lower() in ("true", "yes"):
            ref_correct_default = 1
        elif ref_correct_raw.lower() in ("false", "no"):
            ref_correct_default = 2
        else:
            ref_correct_default = 0

        ref_correct_sel = st.selectbox(
            "Is the reference answer factually correct? *",
            options=ref_correct_options,
            index=ref_correct_default,
            key=f"ref_correct_{current_idx}",
            help="True = the reference answer is factually accurate per the source chunk.",
        )

        # supported_by_chunk
        supp_raw = row.get("supported_by_chunk", "").strip()
        supp_options = ["(not labeled)", "True", "False"]
        if supp_raw.lower() in ("true", "yes"):
            supp_default = 1
        elif supp_raw.lower() in ("false", "no"):
            supp_default = 2
        else:
            supp_default = 0

        supp_sel = st.selectbox(
            "Is the answer directly supported by the source chunk? *",
            options=supp_options,
            index=supp_default,
            key=f"supp_{current_idx}",
            help="True = you can point to a specific sentence in the chunk that supports it.",
        )

        # edited_reference (optional)
        edited_ref = st.text_area(
            "Corrected reference answer (optional — fill if reference_correct=False):",
            value=row.get("edited_reference", ""),
            height=100,
            key=f"edited_{current_idx}",
            placeholder="Leave blank if reference answer is correct.",
        )

        notes = st.text_area(
            "Notes / flags (optional):",
            value=row.get("notes", ""),
            height=80,
            key=f"notes_{current_idx}",
            placeholder="e.g. 'ambiguous question', 'chunk missing context'",
        )

        # ── Save button ───────────────────────────────────────────────────────
        if st.button("💾 Save Label", type="primary", use_container_width=True):
            if ref_correct_sel == "(not labeled)" or supp_sel == "(not labeled)":
                st.error("You must select True or False for both label fields before saving.")
            elif not reviewer.strip():
                st.error("Please enter your reviewer name / ID.")
            else:
                updated = {
                    "reference_correct": ref_correct_sel,
                    "supported_by_chunk": supp_sel,
                    "edited_reference": edited_ref.strip(),
                    "notes": notes.strip(),
                    "labeled_by": reviewer.strip(),
                    "labeled_at": datetime.now(timezone.utc).isoformat(),
                }
                _save_row(csv_path, current_idx, updated)
                st.success(f"✅ Label saved for {row['id']}")
                # Auto-advance to next unlabeled item
                rows_refreshed = _load_csv(csv_path)
                next_unlabeled = next(
                    (
                        i for i in range(current_idx + 1, len(rows_refreshed))
                        if not rows_refreshed[i].get("reference_correct", "").strip()
                    ),
                    None,
                )
                if next_unlabeled is not None:
                    st.session_state["row_idx"] = next_unlabeled
                st.rerun()

    st.divider()
    st.caption(
        "**⚠️ REMINDER**: Do not start evaluation experiments until all items are labeled. "
        f"CSV path: `{csv_path}`"
    )


if __name__ == "__main__":
    main()
