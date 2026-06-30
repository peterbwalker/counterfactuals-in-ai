"""
stage1_baseline.py
==================
Stage I — Baseline Fallacy Profiling

Goal    : Establish CRI_0 and the per-pattern error matrix E ∈ R^{8 x K}
          before any counterfactual training.
Output  : results/stage1_fallacy_profile.csv
          results/stage1_cri0.json

Protocol (Slide 10):
  1. Query model on all 8 logical variants of N canonical P=>Q statements.
  2. Record fraction judged TRUE per pattern.
  3. Compute CRI_0 over the full statement set.
  4. Stratify by application domain K.

Run:
    python stage1_baseline.py [--model gpt-4o] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import config
from metrics import (
    compute_cri,
    compute_pattern_error,
    build_error_matrix,
    summarize_stage1,
)
from data.statement_bundles import (
    StatementVariant,
    load_corpus,
    bundles_by_pattern,
    bundles_by_domain,
)


# ─────────────────────────────────────────────────────────────────────────────
# Model query layer — add new providers here
# ─────────────────────────────────────────────────────────────────────────────

def query_model(
    statement: str,
    model_cfg: Dict,
    dry_run: bool = False,
) -> Tuple[bool, float]:
    """
    Query a model with a single statement and return (judgement, probability).

    Returns
    -------
    judgement : True if model says statement is TRUE, else False.
    prob      : estimated probability of TRUE in [0, 1].
                For APIs that don't return logprobs, this is 1.0 / 0.0.
    """
    if dry_run:
        # Simulate a biased model: mostly says TRUE (mimics affirmation bias)
        import random
        prob = random.uniform(0.55, 0.95)
        return prob > 0.5, prob

    prompt = config.EVALUATION_PROMPT_TEMPLATE.format(statement=statement)
    provider = model_cfg["provider"]

    if provider == "openai":
        return _query_openai(prompt, model_cfg)
    elif provider == "anthropic":
        return _query_anthropic(prompt, model_cfg)
    elif provider == "hf":
        return _query_hf(prompt, model_cfg)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _query_openai(prompt: str, model_cfg: Dict) -> Tuple[bool, float]:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_cfg["model_id"],
        messages=[
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=5,
        temperature=0,
    )
    text = response.choices[0].message.content.strip().upper()
    judgement = text.startswith("TRUE")
    prob = 1.0 if judgement else 0.0
    return judgement, prob


def _query_anthropic(prompt: str, model_cfg: Dict) -> Tuple[bool, float]:
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model_cfg["model_id"],
        max_tokens=5,
        system=config.SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip().upper()
    judgement = text.startswith("TRUE")
    prob = 1.0 if judgement else 0.0
    return judgement, prob


def _query_hf(prompt: str, model_cfg: Dict) -> Tuple[bool, float]:
    """
    Query a HuggingFace model locally.
    Requires transformers + torch. The model is loaded once and cached.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = model_cfg["model_id"]

    # Lazy load — cache on function attribute
    if not hasattr(_query_hf, "_loaded") or _query_hf._loaded != model_id:
        _query_hf.tokenizer = AutoTokenizer.from_pretrained(model_id)
        _query_hf.model     = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
        _query_hf._loaded = model_id

    tok   = _query_hf.tokenizer
    model = _query_hf.model

    full_prompt = f"{config.SYSTEM_PROMPT}\n\n{prompt}"
    inputs = tok(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id
        )
    generated = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    judgement = generated.strip().upper().startswith("TRUE")
    return judgement, float(judgement)


# ─────────────────────────────────────────────────────────────────────────────
# Stage I core logic
# ─────────────────────────────────────────────────────────────────────────────

def run_stage1(
    model_cfg: Dict,
    dry_run: bool = False,
    n_per_domain: Optional[int] = None,
    delay: float = 0.5,
) -> Dict:
    """
    Run Stage I — Baseline Fallacy Profiling for one model.

    Returns a results dict with:
      - error_matrix : {pattern: {domain: error_rate}}
      - cri_0        : scalar CRI_0
      - raw_records  : list of dicts for CSV export
    """
    corpus  = load_corpus(n_per_domain=n_per_domain or config.N_STATEMENTS_PER_DOMAIN)
    by_pat  = bundles_by_pattern(corpus)

    model_name = model_cfg["name"]
    print(f"\nStage I — {model_name}  ({len(corpus)} bundles × 8 variants)\n")

    # ── Query every variant ───────────────────────────────────────────────────
    raw_records: List[Dict] = []
    results_by_pat_dom: Dict[str, Dict[str, List[bool]]] = {}

    all_responses:   List[bool]  = []
    all_probs:       List[float] = []
    all_neg_probs:   List[float] = []  # p(TRUE | ~P) for CRI_0 approximation

    pattern_validity = {p: v for p, _, v in config.LOGICAL_PATTERNS}

    for pattern_name, variants in tqdm(by_pat.items(), desc="Patterns"):
        results_by_pat_dom[pattern_name] = {}

        for v in tqdm(variants, desc=f"  {pattern_name}", leave=False):
            judgement, prob = query_model(v.statement, model_cfg, dry_run)

            all_responses.append(judgement)
            all_probs.append(prob)

            # For CRI_0 we pair P=>Q responses with ~P=>~Q responses
            # as a proxy for f(x) vs f(x') under negation
            if pattern_name == "~P=>~Q":
                all_neg_probs.append(prob)

            domain = v.domain
            results_by_pat_dom[pattern_name].setdefault(domain, []).append(judgement)

            raw_records.append({
                "model":       model_name,
                "domain":      v.domain,
                "domain_label": v.domain_label,
                "pattern":     v.pattern,
                "is_valid":    v.is_valid,
                "p":           v.p,
                "q":           v.q,
                "statement":   v.statement,
                "judgement":   judgement,
                "prob":        prob,
            })

            if not dry_run:
                time.sleep(delay)

    # ── Compute error matrix ──────────────────────────────────────────────────
    error_matrix = build_error_matrix(results_by_pat_dom, pattern_validity)

    # ── Compute CRI_0 ─────────────────────────────────────────────────────────
    # Approximate: compare p(TRUE | P=>Q) vs p(TRUE | ~P=>~Q).
    # Distance proxy: uniform 0.3 (mid-level semantic shift under negation).
    # In production, compute actual embedding distances.
    p_pos_probs  = [r["prob"] for r in raw_records if r["pattern"] == "P=>Q"]
    p_neg_probs  = [r["prob"] for r in raw_records if r["pattern"] == "~P=>~Q"]
    n_pairs      = min(len(p_pos_probs), len(p_neg_probs))
    distances    = [0.3] * n_pairs   # placeholder — replace with embed distances

    cri_0 = compute_cri(
        outputs           = p_pos_probs[:n_pairs],
        outputs_perturbed = p_neg_probs[:n_pairs],
        distances         = distances,
    ) if n_pairs > 0 else float("nan")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(summarize_stage1(cri_0, error_matrix))

    return {
        "model":        model_name,
        "error_matrix": error_matrix,
        "cri_0":        cri_0,
        "raw_records":  raw_records,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Export helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results_list: List[Dict]) -> None:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Flat CSV of all raw records
    all_records = [r for res in results_list for r in res["raw_records"]]
    df = pd.DataFrame(all_records)
    df.to_csv(config.STAGE1_OUTPUT, index=False)
    print(f"Raw records saved to {config.STAGE1_OUTPUT}")

    # CRI_0 and error matrices as JSON
    summary = {
        res["model"]: {
            "cri_0":        res["cri_0"],
            "error_matrix": res["error_matrix"],
        }
        for res in results_list
    }
    with open(config.STAGE1_CRI, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"CRI_0 and error matrices saved to {config.STAGE1_CRI}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage I — Baseline Fallacy Profiling"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name from config.MODELS (default: run all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate model responses without calling any API.",
    )
    parser.add_argument(
        "--n-per-domain",
        type=int,
        default=None,
        help="Limit statements per domain (for quick test runs).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between API calls (default: 0.5).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    models = config.MODELS
    if args.model:
        models = [m for m in models if m["name"] == args.model]
        if not models:
            print(f"Model '{args.model}' not found in config.MODELS.")
            sys.exit(1)

    all_results = []
    for model_cfg in models:
        result = run_stage1(
            model_cfg    = model_cfg,
            dry_run      = args.dry_run,
            n_per_domain = args.n_per_domain,
            delay        = args.delay,
        )
        all_results.append(result)

    save_results(all_results)
    print("\nStage I complete.")
