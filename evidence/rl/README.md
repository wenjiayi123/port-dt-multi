# Portable RL evidence

This directory contains generated, integrity-checked benchmark summaries that
are safe to review after a fresh clone. Every bundle records the dataset hash,
algorithm implementation, seed, training steps, no-render count, chronological
holdout protocol, evaluation metrics, uncertainty, and model artifact hash.

The full model files and runtime directories remain excluded from Git to keep
the repository small. Reproduce them with:

```bash
source .venv/bin/activate
python -m scripts.rl_benchmark --dataset public_us_la_6min_v1 --steps 10000
python -m scripts.export_rl_evidence --dataset public_us_la_6min_v1
```

`RL_SMOKE_WIRING_ONLY` proves that optimizer, persistence, evaluation and
inference wiring ran. It is not comparative performance evidence.

Historical runs whose recorded training hash no longer matches the checked-in
dataset remain in the JSON bundle for provenance, with
`current_artifact_matches_training=false`. They are excluded from the Markdown
performance table until the exact historical artifact is restored or the run
is reproduced on the current dataset.
