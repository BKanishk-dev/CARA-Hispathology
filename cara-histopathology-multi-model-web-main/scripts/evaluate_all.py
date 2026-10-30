#!/usr/bin/env python
"""Evaluate all models with the CARA framework and write the dashboard results.

    python scripts/evaluate_all.py                  # real mode: needs trained checkpoints + PanNuke
    python scripts/evaluate_all.py --demo           # Demo Mode: simulated probabilities, synthetic data
    python scripts/evaluate_all.py --demo --update-readme

Procedure (identical in both modes)
  1. obtain softmax probabilities for the validation and test splits
  2. compute the image quality score of every image
  3. fit CARA (class reliability + class-specific thresholds) on the VALIDATION split only
  4. apply the frozen CARA statistics to the test split and compute
     accuracy, precision, recall, F1, coverage, rejection rate, selective accuracy
Outputs: backend/results/calibration_<mode>.json and backend/results/test_results_<mode>.json
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.cara import CARAConfig  # noqa: E402
from app.inference import evaluate_models, results_markdown_table  # noqa: E402
from app.utils import MODEL_KEYS  # noqa: E402

START, END = "<!-- RESULTS_TABLE_START -->", "<!-- RESULTS_TABLE_END -->"


def update_readme(results: dict) -> None:
    readme = ROOT / "README.md"
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    label = ("**Demo Mode numbers (simulated probabilities on a synthetic dataset - NOT PanNuke results):**"
             if results["mode"] == "demo" else "**PanNuke test-set results (1,581 images):**")
    block = f"{START}\n{label}\n\n{results_markdown_table(results)}\n{END}"
    if START in text and END in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
        readme.write_text(text, encoding="utf-8")
        print("README table updated")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="Demo Mode (simulated probabilities, synthetic data)")
    ap.add_argument("--models", nargs="+", default=MODEL_KEYS, choices=MODEL_KEYS)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--update-readme", action="store_true")
    args = ap.parse_args()

    cfg = CARAConfig(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    mode = "demo" if args.demo else "real"
    results = evaluate_models(mode, args.models, cfg, args.batch_size, args.workers)
    print("\n" + results_markdown_table(results))
    if args.update_readme:
        update_readme(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
