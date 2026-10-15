"""
CLI script to export the labeling CSV template from the candidate dataset.

Usage:
    python -m litmus_v2.labeling.export_csv \
        --output litmus_v2/data/labeling_template.csv

This script reads the curated candidate Q/A items and writes a CSV with all
human-label columns BLANK. It does NOT write to results/.
"""

import argparse
import sys
from pathlib import Path

# Ensure package root is on path when run as a script
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from litmus_v2.data.corpus import load_corpus
from litmus_v2.data.generator import build_candidate_dataset
from litmus_v2.labeling.labeling import export_labeling_csv


def main():
    parser = argparse.ArgumentParser(
        description="Export blank labeling CSV template for litmus_v2 gold set."
    )
    parser.add_argument(
        "--docs",
        default="docs",
        help="Path to the knowledge corpus folder (default: docs/)",
    )
    parser.add_argument(
        "--output",
        default="litmus_v2/data/labeling_template.csv",
        help="Output CSV path (default: litmus_v2/data/labeling_template.csv)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV (USE WITH CARE — you will lose previous labels)",
    )
    args = parser.parse_args()

    print(f"[litmus_v2] Loading corpus from: {args.docs}")
    try:
        chunks, chunk_lookup = load_corpus(args.docs)
        print(f"[litmus_v2] Loaded {len(chunks)} chunks from {len(set(c.doc_name for c in chunks))} documents.")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Tip: Run from the repo root, or use --docs to specify the corpus folder.")
        sys.exit(1)

    print("[litmus_v2] Generating candidate Q/A items...")
    candidates = build_candidate_dataset(chunks)
    print(f"[litmus_v2] Generated {len(candidates)} candidate items.")

    from collections import Counter
    type_dist = Counter(qa.question_type.value for qa in candidates)
    print(f"[litmus_v2] Type distribution: {dict(type_dist)}")

    print(f"[litmus_v2] Writing labeling template → {args.output}")
    export_labeling_csv(
        candidates=candidates,
        chunk_lookup=chunk_lookup,
        output_path=args.output,
        overwrite=args.overwrite,
    )
    print("\n[litmus_v2] ✅ Done. Next steps:")
    print("  1. Open the CSV in a spreadsheet or run: streamlit run litmus_v2/labeling/app.py")
    print("  2. Fill in all label columns for each row.")
    print("  3. STOP — do not proceed to Phase 3 until all rows are labeled.")


if __name__ == "__main__":
    main()
