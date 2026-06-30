"""
metrics.py
==========
Implements all quantitative metrics from the CRF framework.

Equations map directly to the slides:
  - CRI      : Slide 4 / Slide 9
  - ARA      : Slide 7
  - FAPR     : Slide 7
  - delta_U  : Slide 7
  - L_neg    : Slide 5
  - V(A_i)   : Slide 6
  - P(A'_i)  : Slide 7 (Gibbs weighting)
"""

from __future__ import annotations

import math
import numpy as np
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# CRI  —  Counterfactual Robustness Index
# Slide 4:  CRI = 1 - (1/N) * sum( |f(x_i) - f(x'_i)| / Delta(x_i, x'_i) )
# ─────────────────────────────────────────────────────────────────────────────

def compute_cri(
    outputs: List[float],
    outputs_perturbed: List[float],
    distances: List[float],
    epsilon: float = 1e-8,
) -> float:
    """
    Compute the Counterfactual Robustness Index.

    Parameters
    ----------
    outputs           : f(x_i)  — model output scores for original inputs.
                        Float in [0, 1] (e.g. P(TRUE)).
    outputs_perturbed : f(x'_i) — scores for perturbed inputs.
    distances         : Delta(x_i, x'_i) — semantic distances in [0, 1].
                        Use 1 - cosine_similarity(embed(x), embed(x')).
    epsilon           : small constant to avoid division by zero.

    Returns
    -------
    CRI in [0, 1]. Higher = more robust.
    """
    assert len(outputs) == len(outputs_perturbed) == len(distances), (
        "outputs, outputs_perturbed, and distances must have the same length."
    )
    if len(outputs) == 0:
        return float("nan")

    ratios = [
        abs(f - fp) / max(d, epsilon)
        for f, fp, d in zip(outputs, outputs_perturbed, distances)
    ]
    return 1.0 - float(np.mean(ratios))


def cri_from_bool_responses(
    responses: List[bool],
    responses_perturbed: List[bool],
    distances: List[float],
) -> float:
    """
    Convenience wrapper when model outputs are boolean (TRUE/FALSE).
    Maps True -> 1.0, False -> 0.0 before computing CRI.
    """
    return compute_cri(
        outputs=[float(r) for r in responses],
        outputs_perturbed=[float(r) for r in responses_perturbed],
        distances=distances,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fallacy Error Matrix  —  Stage I output
# E[pattern][domain] = fraction of statements judged TRUE
# ─────────────────────────────────────────────────────────────────────────────

def compute_pattern_error(
    responses: List[bool],
    is_valid_pattern: bool,
) -> float:
    """
    Error rate for a single logical pattern.

    For the valid pattern (P=>Q), error = fraction judged FALSE (missed).
    For fallacy patterns,      error = fraction judged TRUE  (accepted).

    Parameters
    ----------
    responses        : list of model TRUE/FALSE judgements.
    is_valid_pattern : True only for P=>Q.

    Returns
    -------
    Error rate in [0, 1].
    """
    if not responses:
        return float("nan")
    if is_valid_pattern:
        # error = missed valid inferences
        return sum(not r for r in responses) / len(responses)
    else:
        # error = accepted fallacies
        return sum(responses) / len(responses)


def build_error_matrix(
    results: Dict[str, Dict[str, List[bool]]],
    pattern_validity: Dict[str, bool],
) -> Dict[str, Dict[str, float]]:
    """
    Build the E ∈ R^{8 x K} fallacy profile matrix (Slide 10).

    Parameters
    ----------
    results          : {pattern_name: {domain_name: [bool responses]}}
    pattern_validity : {pattern_name: bool} — from config.LOGICAL_PATTERNS

    Returns
    -------
    {pattern_name: {domain_name: error_rate}}
    """
    matrix: Dict[str, Dict[str, float]] = {}
    for pattern, domain_results in results.items():
        matrix[pattern] = {}
        valid = pattern_validity.get(pattern, False)
        for domain, responses in domain_results.items():
            matrix[pattern][domain] = compute_pattern_error(responses, valid)
    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# ARA / FAPR  —  Assumption Rejection Accuracy & False Assumption Persistence
# Slide 7:  ARA  = |A_invalid ∩ A_rejected| / |A_invalid|
#           FAPR = 1 - ARA
# ─────────────────────────────────────────────────────────────────────────────

def compute_ara(
    ground_truth_invalid: List[bool],
    model_rejected: List[bool],
) -> float:
    """
    Assumption Rejection Accuracy.

    Parameters
    ----------
    ground_truth_invalid : True  if assumption is actually invalid.
    model_rejected       : True  if the CRF/model rejected the assumption.

    Returns
    -------
    ARA in [0, 1]. Higher = better at catching invalid premises.
    """
    assert len(ground_truth_invalid) == len(model_rejected)
    n_invalid = sum(ground_truth_invalid)
    if n_invalid == 0:
        return float("nan")

    correctly_rejected = sum(
        gt and rej
        for gt, rej in zip(ground_truth_invalid, model_rejected)
    )
    return correctly_rejected / n_invalid


def compute_fapr(ara: float) -> float:
    """False Assumption Persistence Rate = 1 - ARA."""
    return 1.0 - ara


# ─────────────────────────────────────────────────────────────────────────────
# Delta U  —  Decision Utility Improvement
# Slide 7:  ΔU = U_CRF - U_baseline
# ─────────────────────────────────────────────────────────────────────────────

def compute_delta_u(
    utility_crf: float,
    utility_baseline: float,
) -> float:
    """
    Decision utility improvement from the CRF pipeline.

    Utility is task-specific (e.g. accuracy, F1, or a domain score).
    ΔU > 0 is the primary empirical criterion for deployment (Slide 7).
    """
    return utility_crf - utility_baseline


# ─────────────────────────────────────────────────────────────────────────────
# L_neg  —  Counterfactual Denial Loss
# Slide 5:  L_neg = - sum_i log(1 - p_theta(Q_i | ~P_i))
# ─────────────────────────────────────────────────────────────────────────────

def compute_l_neg(
    probs_q_given_neg_p: List[float],
    epsilon: float = 1e-8,
) -> float:
    """
    Compute the counterfactual denial loss L_neg.

    Parameters
    ----------
    probs_q_given_neg_p : p_theta(Q_i | ¬P_i) for each statement i.
                          Probability model assigns TRUE to Q given negated P.
    epsilon             : numerical stability clip.

    Returns
    -------
    L_neg (positive scalar; lower = better at denying invalid premises).
    """
    losses = [
        -math.log(max(1.0 - p, epsilon))
        for p in probs_q_given_neg_p
    ]
    return float(np.mean(losses))


def compute_l_pos(
    probs_q_given_p: List[float],
    epsilon: float = 1e-8,
) -> float:
    """
    Compute the affirmative loss L_pos.

    Slide 2:  L_pos = - sum_i log p_theta(Q_i | P_i)
    """
    losses = [
        -math.log(max(p, epsilon))
        for p in probs_q_given_p
    ]
    return float(np.mean(losses))


def compute_fluency_degradation(
    l_task_lambda: float,
    l_task_baseline: float,
) -> float:
    """
    Fluency degradation δ_L = L_Task^λ - L_Task^0 (Slide 10).
    Should remain < FLUENCY_DEGRADATION_THRESHOLD at optimal λ_CF*.
    """
    return l_task_lambda - l_task_baseline


# ─────────────────────────────────────────────────────────────────────────────
# V(A_i)  —  Assumption Validation Function
# Slide 6:  V(A_i) = 1 if D(B_j, B̂_j) ≤ ε, else 0
# ─────────────────────────────────────────────────────────────────────────────

def validate_assumption(
    predicted_outcome: float,
    observed_outcome: float,
    epsilon: float,
    distance_fn=None,
) -> int:
    """
    CRF assumption validation: V(A_i) ∈ {0, 1}.

    Parameters
    ----------
    predicted_outcome : B_j  = f_theta(A_i)
    observed_outcome  : B̂_j = ground truth / evidence
    epsilon           : acceptance threshold
    distance_fn       : D(B_j, B̂_j). Defaults to abs difference.

    Returns
    -------
    1 if premise consistent with evidence, 0 if rejected.
    """
    if distance_fn is None:
        distance_fn = lambda a, b: abs(a - b)
    return int(distance_fn(predicted_outcome, observed_outcome) <= epsilon)


# ─────────────────────────────────────────────────────────────────────────────
# P(A'_i)  —  Gibbs / Boltzmann weighting over hypothesis space
# Slide 7:  P(A'_i) ∝ exp( -D(f_theta(A'_i), B̂_j) / tau )
# ─────────────────────────────────────────────────────────────────────────────

def gibbs_weights(
    distances: List[float],
    tau: float = 0.5,
) -> List[float]:
    """
    Compute Gibbs (Boltzmann) distribution over candidate assumptions.

    Parameters
    ----------
    distances : D(f_theta(A'_i), B̂_j) for each candidate i.
    tau       : temperature. Low τ → sharp peak (one dominant hypothesis).
                High τ → diffuse (genuine analytic ambiguity).

    Returns
    -------
    Normalized probability distribution over candidates.
    """
    if not distances:
        return []
    log_weights = [-d / tau for d in distances]
    # numerically stable softmax
    max_lw = max(log_weights)
    weights = [math.exp(lw - max_lw) for lw in log_weights]
    total = sum(weights)
    return [w / total for w in weights]


def best_hypothesis(
    candidates: List[str],
    distances: List[float],
    tau: float = 0.5,
) -> Tuple[str, float, List[float]]:
    """
    Select A* = argmin D(f_theta(A'_i), B̂_j) and return the full distribution.

    Returns
    -------
    (best_candidate, min_distance, gibbs_distribution)
    """
    assert len(candidates) == len(distances)
    weights = gibbs_weights(distances, tau)
    best_idx = int(np.argmin(distances))
    return candidates[best_idx], distances[best_idx], weights


# ─────────────────────────────────────────────────────────────────────────────
# Summary reporter
# ─────────────────────────────────────────────────────────────────────────────

def summarize_stage1(
    cri_0: float,
    error_matrix: Dict[str, Dict[str, float]],
) -> str:
    """Pretty-print Stage I results."""
    lines = [
        f"\n{'='*60}",
        f"  STAGE I — Baseline Fallacy Profile",
        f"{'='*60}",
        f"  CRI_0 = {cri_0:.4f}",
        f"",
        f"  {'Pattern':<14} " + "  ".join(f"{d[:10]:<12}" for d in next(iter(error_matrix.values()))),
    ]
    for pattern, domain_errs in error_matrix.items():
        row = f"  {pattern:<14} " + "  ".join(
            f"{err:<12.3f}" for err in domain_errs.values()
        )
        lines.append(row)
    lines.append(f"{'='*60}\n")
    return "\n".join(lines)


def summarize_stage3(
    ara: float,
    fapr: float,
    delta_u: float,
    cri: float,
) -> str:
    """Pretty-print Stage III results."""
    lines = [
        f"\n{'='*60}",
        f"  STAGE III — CRF Inference Evaluation",
        f"{'='*60}",
        f"  ARA    = {ara:.4f}   (Assumption Rejection Accuracy)",
        f"  FAPR   = {fapr:.4f}   (False Assumption Persistence Rate)",
        f"  ΔU     = {delta_u:+.4f}  (Decision utility vs. baseline)",
        f"  CRI    = {cri:.4f}   (Robustness Index)",
        f"{'='*60}\n",
    ]
    return "\n".join(lines)
