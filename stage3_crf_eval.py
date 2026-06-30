"""
stage3_crf_eval.py
==================
Stage III — CRF Inference Evaluation

Goal    : Measure assumption-level improvement from the CRF pipeline
          vs. autoregressive, chain-of-thought, and self-consistency baselines.
Output  : results/stage3_crf_eval.csv

Protocol (Slide 11):
  1. Hold-out premise set with known validity labels V(A_i) ∈ {0, 1}.
  2. Compare: autoregressive | CoT | self-consistency | CRF.
  3. Compute ARA, FAPR, ΔU per method per domain.

Use-case mapping (Slide 11):
  - Policy & alignment testing : ΔU over decision task suite.
  - Model comparison & selection: ARA/FAPR leaderboard across runs.

Run:
    python stage3_crf_eval.py [--model gpt-4o] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import config
from metrics import (
    validate_assumption,
    gibbs_weights,
    best_hypothesis,
    compute_ara,
    compute_fapr,
    compute_delta_u,
    compute_cri,
    summarize_stage3,
)
from data.statement_bundles import load_corpus, StatementBundle, StatementVariant
from stage1_baseline import query_model


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1 — Autoregressive (standard forward inference)
# ─────────────────────────────────────────────────────────────────────────────

def baseline_autoregressive(
    variant: StatementVariant,
    model_cfg: Dict,
    dry_run: bool = False,
) -> Tuple[bool, float]:
    """
    Standard next-token prediction. No special prompting.
    This is the current state of all major LLMs (Slide 12, baseline 1).
    """
    judgement, prob = query_model(variant.statement, model_cfg, dry_run)
    return judgement, prob


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2 — Chain-of-thought prompting
# ─────────────────────────────────────────────────────────────────────────────

COT_PROMPT_TEMPLATE = """Evaluate the following logical statement step by step.

Statement: "{statement}"

Step 1: Identify the antecedent (if-clause) and consequent (then-clause).
Step 2: Consider whether the logical relationship is valid.
Step 3: State your conclusion: TRUE or FALSE.

Answer (TRUE or FALSE only):"""


def baseline_chain_of_thought(
    variant: StatementVariant,
    model_cfg: Dict,
    dry_run: bool = False,
) -> Tuple[bool, float]:
    """
    Chain-of-thought prompting — reasoning transparency, no falsification.
    Slide 12, baseline 2.
    """
    if dry_run:
        prob = random.uniform(0.50, 0.90)
        return prob > 0.5, prob

    prompt = COT_PROMPT_TEMPLATE.format(statement=variant.statement)
    provider = model_cfg["provider"]

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=model_cfg["model_id"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model_cfg["model_id"],
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
    else:
        text = "TRUE" if random.random() > 0.4 else "FALSE"

    # Extract final TRUE/FALSE from CoT response
    lines = [l.strip().upper() for l in text.split("\n") if l.strip()]
    judgement = False
    for line in reversed(lines):
        if line.startswith("TRUE"):
            judgement = True
            break
        if line.startswith("FALSE"):
            judgement = False
            break

    return judgement, float(judgement)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 3 — Self-consistency sampling
# ─────────────────────────────────────────────────────────────────────────────

def baseline_self_consistency(
    variant: StatementVariant,
    model_cfg: Dict,
    k: int = None,
    dry_run: bool = False,
) -> Tuple[bool, float]:
    """
    Majority vote over k sampled responses.
    Slide 12, baseline 3.
    """
    k = k or config.SELF_CONSISTENCY_K

    if dry_run:
        votes = [random.random() > 0.45 for _ in range(k)]
    else:
        votes = []
        for _ in range(k):
            j, _ = query_model(variant.statement, model_cfg, dry_run=False)
            votes.append(j)
            time.sleep(0.2)

    true_count = sum(votes)
    judgement  = true_count > k / 2
    prob       = true_count / k
    return judgement, prob


# ─────────────────────────────────────────────────────────────────────────────
# CRF Pipeline (ours)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_counterfactual_assumptions(
    variant: StatementVariant,
    bundle,
) -> List[Tuple[StatementVariant, float]]:
    """
    Construct A^cf — the set of alternative assumptions for the CRF.

    Strategy: use the other 7 logical variants of the same (P, Q) pair
    as the candidate assumption set. This is the logical negation /
    perturbation path described in Slide 6.

    Returns list of (variant, placeholder_distance).
    """
    candidates = [
        (v, 0.1 + i * 0.05)   # placeholder distances — replace with embeddings
        for i, v in enumerate(bundle.variants)
        if v.pattern != variant.pattern
    ]
    return candidates


def crf_pipeline(
    variant: StatementVariant,
    bundle,
    model_cfg: Dict,
    observed_outcome: float,
    epsilon: float = None,
    tau: float = None,
    dry_run: bool = False,
) -> Tuple[bool, float, Dict]:
    """
    Full CRF inference pipeline (Slides 6–7).

    Stages:
      1. Assumption extraction  — use variant as current assumption A_i.
      2. Query model            — get predicted outcome B_j = f_theta(A_i).
      3. Assumption validation  — V(A_i) = 1 if D(B_j, B̂_j) ≤ ε.
      4. If V(A_i) = 0:
           a. Generate A^cf from bundle variants.
           b. Score each candidate.
           c. Select A* = argmin D(f_theta(A'_i), B̂_j).
           d. Compute Gibbs distribution P(A'_i).
      5. Return final judgement, probability, and diagnostic dict.
    """
    epsilon = epsilon or config.CRF_VALIDATION_EPSILON
    tau     = tau     or config.CRF_TEMPERATURE

    # Step 1–2: query model on current assumption
    judgement, prob = query_model(variant.statement, model_cfg, dry_run)
    predicted_outcome = prob

    # Step 3: validate assumption
    v_ai = validate_assumption(predicted_outcome, observed_outcome, epsilon)

    diagnostics = {
        "pattern":         variant.pattern,
        "v_ai":            v_ai,
        "initial_prob":    prob,
        "initial_judge":   judgement,
        "crf_triggered":   False,
        "a_star_pattern":  None,
        "gibbs_weights":   None,
    }

    if v_ai == 1:
        # Assumption consistent with evidence — accept
        return judgement, prob, diagnostics

    # Step 4: CRF triggered — premise rejected, find replacement
    diagnostics["crf_triggered"] = True

    candidates = _generate_counterfactual_assumptions(variant, bundle)
    if not candidates:
        return judgement, prob, diagnostics

    # Score each candidate
    cand_variants  = [c[0] for c in candidates]
    cand_distances = []

    for cand_v, _ in candidates:
        cand_j, cand_p = query_model(cand_v.statement, model_cfg, dry_run)
        d = abs(cand_p - observed_outcome)
        cand_distances.append(d)
        if not dry_run:
            time.sleep(0.2)

    # Select A* and compute Gibbs distribution
    best_v, best_d, weights = best_hypothesis(
        [v.pattern for v in cand_variants],
        cand_distances,
        tau=tau,
    )
    diagnostics["a_star_pattern"] = best_v
    diagnostics["gibbs_weights"]  = dict(zip(
        [v.pattern for v in cand_variants], weights
    ))

    # Re-query with best assumption
    best_variant = cand_variants[int(np.argmin(cand_distances))]
    final_j, final_p = query_model(best_variant.statement, model_cfg, dry_run)

    return final_j, final_p, diagnostics


# ─────────────────────────────────────────────────────────────────────────────
# Stage III core logic
# ─────────────────────────────────────────────────────────────────────────────

def run_stage3(
    model_cfg: Dict,
    dry_run: bool = False,
    n_per_domain: Optional[int] = None,
    delay: float = 0.5,
) -> Dict:
    """
    Run Stage III for one model across all four inference methods.
    """
    corpus     = load_corpus(n_per_domain=n_per_domain or 3)
    model_name = model_cfg["name"]
    methods    = ["autoregressive", "cot", "self_consistency", "crf"]

    print(f"\nStage III — {model_name}  ({len(corpus)} bundles)\n")

    raw_records: List[Dict] = []

    for bundle in tqdm(corpus, desc="Bundles"):
        # Ground truth: P=>Q variant should be TRUE, all others FALSE
        for variant in bundle.variants:
            ground_truth_valid = variant.is_valid
            # observed_outcome: 1.0 for valid patterns, 0.0 for fallacies
            observed_outcome   = 1.0 if ground_truth_valid else 0.0

            record = {
                "model":             model_name,
                "domain":            variant.domain,
                "pattern":           variant.pattern,
                "is_valid":          variant.is_valid,
                "statement":         variant.statement,
                "ground_truth":      ground_truth_valid,
                "observed_outcome":  observed_outcome,
            }

            # ── Autoregressive ────────────────────────────────────────────
            j_ar, p_ar = baseline_autoregressive(variant, model_cfg, dry_run)
            # Treat as "rejected" if model says FALSE (i.e. prob < 0.5)
            record["ar_judgement"] = j_ar
            record["ar_prob"]      = p_ar
            record["ar_rejected"]  = not j_ar
            if not dry_run: time.sleep(delay)

            # ── Chain-of-thought ──────────────────────────────────────────
            j_cot, p_cot = baseline_chain_of_thought(variant, model_cfg, dry_run)
            record["cot_judgement"] = j_cot
            record["cot_prob"]      = p_cot
            record["cot_rejected"]  = not j_cot
            if not dry_run: time.sleep(delay)

            # ── Self-consistency ──────────────────────────────────────────
            j_sc, p_sc = baseline_self_consistency(
                variant, model_cfg, dry_run=dry_run
            )
            record["sc_judgement"] = j_sc
            record["sc_prob"]      = p_sc
            record["sc_rejected"]  = not j_sc

            # ── CRF ───────────────────────────────────────────────────────
            j_crf, p_crf, diag = crf_pipeline(
                variant, bundle, model_cfg, observed_outcome, dry_run=dry_run
            )
            record["crf_judgement"]    = j_crf
            record["crf_prob"]         = p_crf
            record["crf_rejected"]     = not j_crf
            record["crf_triggered"]    = diag["crf_triggered"]
            record["crf_a_star"]       = diag["a_star_pattern"]

            raw_records.append(record)
            if not dry_run: time.sleep(delay)

    # ── Compute ARA, FAPR, ΔU per method ─────────────────────────────────────
    df = pd.DataFrame(raw_records)

    # Ground truth invalid = fallacy patterns (not is_valid)
    gt_invalid = (~df["ground_truth"]).tolist()

    metrics_by_method: Dict[str, Dict] = {}
    for method in methods:
        rejected_col = f"{method[:2] if method != 'autoregressive' else 'ar'}_rejected"
        if method == "autoregressive":
            rejected = df["ar_rejected"].tolist()
        elif method == "cot":
            rejected = df["cot_rejected"].tolist()
        elif method == "self_consistency":
            rejected = df["sc_rejected"].tolist()
        else:
            rejected = df["crf_rejected"].tolist()

        ara  = compute_ara(gt_invalid, rejected)
        fapr = compute_fapr(ara)

        # ΔU: fraction of decisions that are correct (accuracy as proxy utility)
        if method == "autoregressive":
            correct = [j == g for j, g in zip(df["ar_judgement"], df["ground_truth"])]
        elif method == "cot":
            correct = [j == g for j, g in zip(df["cot_judgement"], df["ground_truth"])]
        elif method == "self_consistency":
            correct = [j == g for j, g in zip(df["sc_judgement"], df["ground_truth"])]
        else:
            correct = [j == g for j, g in zip(df["crf_judgement"], df["ground_truth"])]

        utility = float(np.mean(correct))
        metrics_by_method[method] = {"ara": ara, "fapr": fapr, "utility": utility}

    # ΔU = U_CRF - U_baseline (autoregressive as baseline)
    u_baseline = metrics_by_method["autoregressive"]["utility"]
    u_crf      = metrics_by_method["crf"]["utility"]
    delta_u    = compute_delta_u(u_crf, u_baseline)

    # CRI for CRF vs autoregressive
    cri = compute_cri(
        outputs           = df["ar_prob"].tolist(),
        outputs_perturbed = df["crf_prob"].tolist(),
        distances         = [0.2] * len(df),  # placeholder
    )

    ara_crf  = metrics_by_method["crf"]["ara"]
    fapr_crf = metrics_by_method["crf"]["fapr"]

    print(summarize_stage3(ara_crf, fapr_crf, delta_u, cri))
    print("\n  Method leaderboard (ARA / FAPR / Utility):")
    for m, met in metrics_by_method.items():
        print(f"    {m:<20} ARA={met['ara']:.3f}  FAPR={met['fapr']:.3f}  U={met['utility']:.3f}")

    return {
        "model":          model_name,
        "raw_records":    raw_records,
        "metrics":        metrics_by_method,
        "delta_u":        delta_u,
        "cri":            cri,
    }


def save_results(results_list: List[Dict]) -> None:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    all_records = [r for res in results_list for r in res["raw_records"]]
    df = pd.DataFrame(all_records)
    df.to_csv(config.STAGE3_OUTPUT, index=False)
    print(f"\nStage III results saved to {config.STAGE3_OUTPUT}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Stage III — CRF Inference Evaluation")
    parser.add_argument("--model",       type=str, default=None)
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--n-per-domain",type=int, default=None)
    parser.add_argument("--delay",       type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    models  = config.MODELS
    if args.model:
        models = [m for m in models if m["name"] == args.model]

    all_results = []
    for model_cfg in models:
        result = run_stage3(
            model_cfg    = model_cfg,
            dry_run      = args.dry_run,
            n_per_domain = args.n_per_domain,
            delay        = args.delay,
        )
        all_results.append(result)

    save_results(all_results)
    print("\nStage III complete.")
