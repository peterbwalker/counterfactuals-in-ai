"""
run_all.py
==========
Orchestrates the full four-stage CRF benchmark pipeline.

Stages:
  I   — Baseline fallacy profiling   (stage1_baseline.py)
  II  — λ_CF fine-tuning ablation    (stage2_finetune.py)  [stub — see note]
  III — CRF inference evaluation     (stage3_crf_eval.py)
  IV  — Adversarial stress test      (stage4_adversarial.py)

Usage:
  # Full pipeline, dry run (no API calls)
  python run_all.py --dry-run

  # Single model, real API calls, 3 statements per domain
  python run_all.py --model gpt-4o --n-per-domain 3

  # Specific stages only
  python run_all.py --stages 1 3 4 --dry-run

Note on Stage II:
  Fine-tuning ablation requires access to model weights and a training
  loop — not feasible via inference APIs. stage2_finetune.py provides
  the experimental design and a simulation scaffold. Run it separately
  against a fine-tunable model (e.g. Llama-3-8B via HuggingFace).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

sys.path.insert(0, os.path.dirname(__file__))
import config
from stage1_baseline   import run_stage1,  save_results as save1
from stage3_crf_eval   import run_stage3,  save_results as save3
from stage4_adversarial import run_stage4, save_results as save4


def parse_args():
    parser = argparse.ArgumentParser(
        description="CRF Benchmark — full four-stage pipeline"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Run only this model (name from config.MODELS). Default: all.",
    )
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        default=[1, 3, 4],
        choices=[1, 3, 4],
        help="Which stages to run (2 requires fine-tuning access). Default: 1 3 4.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate model responses — no API calls or charges.",
    )
    parser.add_argument(
        "--n-per-domain",
        type=int,
        default=3,
        help="Statements per domain (default 3 for quick runs; use 20+ for full eval).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds between API calls.",
    )
    return parser.parse_args()


def print_banner(text: str) -> None:
    width = 64
    print(f"\n{'═'*width}")
    print(f"  {text}")
    print(f"{'═'*width}")


def run_pipeline(args) -> None:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Select models
    models = config.MODELS
    if args.model:
        models = [m for m in models if m["name"] == args.model]
        if not models:
            print(f"Model '{args.model}' not found in config.MODELS.")
            sys.exit(1)

    mode = "DRY RUN (simulated)" if args.dry_run else "LIVE (API calls)"
    print_banner(
        f"CRF Benchmark Pipeline\n"
        f"  Models   : {[m['name'] for m in models]}\n"
        f"  Stages   : {args.stages}\n"
        f"  Domains  : {list(config.DOMAINS.keys())}\n"
        f"  N/domain : {args.n_per_domain}\n"
        f"  Mode     : {mode}"
    )

    start_total = time.time()
    summary: dict = {}

    # ── Stage I ───────────────────────────────────────────────────────────────
    if 1 in args.stages:
        print_banner("Stage I — Baseline Fallacy Profiling")
        s1_results = []
        for model_cfg in models:
            result = run_stage1(
                model_cfg    = model_cfg,
                dry_run      = args.dry_run,
                n_per_domain = args.n_per_domain,
                delay        = args.delay,
            )
            s1_results.append(result)
        save1(s1_results)

        summary["stage1"] = {
            res["model"]: {"cri_0": res["cri_0"]}
            for res in s1_results
        }

    # ── Stage II (informational stub) ────────────────────────────────────────
    if 2 in args.stages:
        print_banner("Stage II — Fine-Tuning Ablation (stub)")
        print("  Stage II requires model weight access.")
        print("  Run stage2_finetune.py separately with a local model.")
        print("  See config.LAMBDA_CF_VALUES for the sweep design.")

    # ── Stage III ─────────────────────────────────────────────────────────────
    if 3 in args.stages:
        print_banner("Stage III — CRF Inference Evaluation")
        s3_results = []
        for model_cfg in models:
            result = run_stage3(
                model_cfg    = model_cfg,
                dry_run      = args.dry_run,
                n_per_domain = args.n_per_domain,
                delay        = args.delay,
            )
            s3_results.append(result)
        save3(s3_results)

        summary["stage3"] = {
            res["model"]: {
                "delta_u": res["delta_u"],
                "cri":     res["cri"],
            }
            for res in s3_results
        }

    # ── Stage IV ─────────────────────────────────────────────────────────────
    if 4 in args.stages:
        print_banner("Stage IV — Adversarial Stress Test")
        s4_results = []
        for model_cfg in models:
            result = run_stage4(
                model_cfg    = model_cfg,
                dry_run      = args.dry_run,
                n_per_domain = args.n_per_domain,
                delay        = args.delay,
            )
            s4_results.append(result)
        save4(s4_results)

        summary["stage4"] = {
            res["model"]: {
                lvl: vals["cri_crf"]
                for lvl, vals in res["cri_curve"].items()
            }
            for res in s4_results
        }

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - start_total
    print_banner(f"Pipeline Complete — {elapsed:.1f}s")

    for model in models:
        name = model["name"]
        print(f"\n  {name}")
        if "stage1" in summary and name in summary["stage1"]:
            print(f"    CRI_0          = {summary['stage1'][name]['cri_0']:.4f}")
        if "stage3" in summary and name in summary["stage3"]:
            print(f"    ΔU (CRF vs AR) = {summary['stage3'][name]['delta_u']:+.4f}")
            print(f"    CRI (Stage III)= {summary['stage3'][name]['cri']:.4f}")
        if "stage4" in summary and name in summary["stage4"]:
            curve = summary["stage4"][name]
            print(f"    CRI curve      = " +
                  " → ".join(f"{lvl}:{v:.3f}" for lvl, v in curve.items()))
            certified = curve.get("far", 0) >= config.CRI_DEPLOYMENT_THRESHOLD
            print(f"    Deployment cert= {'YES ✓' if certified else 'NO ✗'}")

    # Save summary JSON
    summary_path = os.path.join(config.RESULTS_DIR, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Full summary → {summary_path}")


if __name__ == "__main__":
    run_pipeline(parse_args())
