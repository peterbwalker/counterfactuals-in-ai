"""
config.py
=========
Central configuration for the CRF benchmark pipeline.
Edit this file to change models, domains, sweep parameters,
and epsilon levels. All other scripts import from here.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


# ── Models under test ────────────────────────────────────────────────────────
# Add or remove entries to change which models are evaluated.
# "provider" controls which API client is used (openai | anthropic | hf).
# "model_id" is the string passed to the API or HuggingFace hub.

MODELS = [
    {"name": "gpt-4o",        "provider": "openai",    "model_id": "gpt-4o"},
    {"name": "claude-sonnet", "provider": "anthropic", "model_id": "claude-sonnet-4-6"},
    {"name": "llama-3-8b",    "provider": "hf",        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct"},
]


# ── Application domains (from Application II use cases) ──────────────────────
# These are the three domains from slide 9/12. Each domain has a label and
# a list of canonical P => Q statements used to build statement bundles.
# Extend each list with real domain-specific statements before running.

DOMAINS = {
    "policy_alignment": {
        "label": "Policy & Alignment Testing",
        "statements": [
            # (P, Q) pairs — causal statements known to be true in this domain
            ("A safety regulation is violated",
             "A compliance action is required"),
            ("A model output diverges from policy guidelines",
             "A human review flag is triggered"),
            ("An AI system is deployed without alignment verification",
             "Downstream risk increases"),
        ],
    },
    "model_comparison": {
        "label": "Model Comparison & Selection",
        "statements": [
            ("A model is trained on a larger and more diverse dataset",
             "Its out-of-distribution generalization improves"),
            ("Fine-tuning reduces the training loss on a target task",
             "Task-specific accuracy increases"),
            ("A model's parameter count increases",
             "Its surface-level fluency improves"),
        ],
    },
    "distribution_shift": {
        "label": "Distribution Shift Detection",
        "statements": [
            ("The input distribution shifts from training to deployment",
             "Model performance degrades"),
            ("The CRI score drops below a certification threshold",
             "The model requires re-evaluation"),
            ("Real-world prompts diverge semantically from benchmark prompts",
             "Fallacy error rates increase"),
        ],
    },
}


# ── The 8 logical pattern variants ───────────────────────────────────────────
# Each entry is (pattern_name, description, is_valid).
# Only P=>Q is a valid inference; the rest are fallacies.
# Used by data/statement_bundles.py to expand each (P, Q) pair.

LOGICAL_PATTERNS = [
    ("P=>Q",   "P implies Q",          True),   # modus ponens — valid
    ("P=>~Q",  "P implies not-Q",      False),  # counterexample
    ("~P=>Q",  "not-P implies Q",      False),  # exception modeling
    ("~P=>~Q", "not-P implies not-Q",  False),  # denying the antecedent
    ("Q=>P",   "Q implies P",          False),  # affirming the consequent
    ("Q=>~P",  "Q implies not-P",      False),  # inverse counterexample
    ("~Q=>P",  "not-Q implies P",      False),  # hidden cause
    ("~Q=>~P", "not-Q implies not-P",  False),  # negation consistency (modus tollens form)
]


# ── Stage II: lambda_CF sweep ─────────────────────────────────────────────────
# lambda_CF = 0 is the pure autoregressive baseline (current LLMs).
# The remaining values test the effect of the counterfactual denial loss.

LAMBDA_CF_VALUES = [0.0, 0.1, 0.5, 1.0]

# Fluency degradation threshold (slide 10: delta_L < 5%)
FLUENCY_DEGRADATION_THRESHOLD = 0.05


# ── Stage IV: adversarial epsilon levels ──────────────────────────────────────
# Semantic distance thresholds for near / mid / far perturbation.
# Measured as 1 - cosine_similarity(embedding(x), embedding(x')).
# epsilon=0 means identical; epsilon=1 means orthogonal in embedding space.

EPSILON_LEVELS = {
    "near": 0.05,   # very close paraphrase
    "mid":  0.15,   # moderate semantic shift
    "far":  0.30,   # substantial but still topic-relevant drift
}

# Number of adversarial variants to generate per statement at each epsilon level
N_ADVERSARIAL_VARIANTS = 5


# ── Evaluation parameters ─────────────────────────────────────────────────────
# N statements sampled per domain for Stage I profiling
N_STATEMENTS_PER_DOMAIN = 20

# k: number of samples for self-consistency baseline (Stage III)
SELF_CONSISTENCY_K = 5

# CRF: acceptance threshold epsilon for V(A_i) (slide 6)
CRF_VALIDATION_EPSILON = 0.10

# CRF: Gibbs temperature tau (slide 7)
CRF_TEMPERATURE = 0.5

# Minimum CRI for deployment certification (Stage IV use-case: policy & alignment)
CRI_DEPLOYMENT_THRESHOLD = 0.80


# ── Output paths ──────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
STAGE1_OUTPUT  = f"{RESULTS_DIR}/stage1_fallacy_profile.csv"
STAGE1_CRI     = f"{RESULTS_DIR}/stage1_cri0.json"
STAGE2_OUTPUT  = f"{RESULTS_DIR}/stage2_lambda_sweep.csv"
STAGE3_OUTPUT  = f"{RESULTS_DIR}/stage3_crf_eval.csv"
STAGE4_OUTPUT  = f"{RESULTS_DIR}/stage4_adversarial.csv"


# ── Prompt templates ──────────────────────────────────────────────────────────
# Used by all stages to query models consistently.
# {statement} is filled with the logical variant being evaluated.

EVALUATION_PROMPT_TEMPLATE = """You are evaluating a logical statement.
Answer only with TRUE or FALSE. Do not explain.

Statement: "{statement}"

Is this statement true?"""

SYSTEM_PROMPT = (
    "You are a precise logical evaluator. "
    "Respond only with TRUE or FALSE."
)
