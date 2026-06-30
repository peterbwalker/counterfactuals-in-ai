# counterfactuals-in-ai
Counterfactual Blog Series

This repository hosts the LaTeX source and PDFs for the Counterfactuals in AI blog series by Peter Walker, Ph.D.

crf_benchmark/
├── config.py              ← All parameters in one place — edit this first
├── metrics.py             ← Every equation from the slides (CRI, ARA, FAPR, ΔU, L_neg, V(A_i), Gibbs)
├── requirements.txt
├── data/
│   └── statement_bundles.py   ← Corpus loader + 8-variant expander
├── stage1_baseline.py     ← Fallacy profiling → E matrix, CRI_0
├── stage3_crf_eval.py     ← CRF vs 3 baselines → ARA, FAPR, ΔU leaderboard
├── stage4_adversarial.py  ← ε stress test → CRI degradation curve
└── run_all.py             ← Orchestrates everything, writes results/
