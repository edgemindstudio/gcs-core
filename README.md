# gcs-core/README.md

# gcs-core

Shared evaluation utilities for GenCyberSynth repos (Diffusion, DCGAN, VAE, AR, MAF, RBM, GMM).

## What it provides
- `compute_all_metrics(...)`: FID/cFID/JS/KL/Diversity + evaluator-CNN utility (REAL vs REAL+SYNTH)
- `evaluate_model_suite(...)`: standard Phase-2 summary dict (generative + utility + deltas + images)
- `write_summary_with_gcs_core(...)`: writes a clean console block and appends one-line JSONL
- `resolve_synth_dir(...)`, `load_synth_any(...)`: robust synthetic loading across repos

## Install (from git, pinned tag)
```bash
pip install "gcs-core @ git+https://github.com/YOURORG/gcs-core@v0.1.0"

