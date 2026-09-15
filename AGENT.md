# AGENT Execution Guidelines for SynAdapt-AD

This document guides AI Agents (e.g. Gemini Notebook, Claude, GPT-4o) on executing tasks within this repository.

## Execution Rules
1. **Low VRAM Constraint (Quadro P1000 - 4GB VRAM):**
   - Always load CLAP in FP16 precision.
   - Cache extracted embeddings to disk (`/results/cached_features/`) before training adapters.
   - Do NOT fine-tune full backbone weights; train only `peft_adapter` (d_mid=64).

2. **GitHub Auto-Sync Protocol:**
   - Execute `python scripts/github_sync.py --message "Stage X checkpoint"` after completing each notebook.

3. **Paper Table Formatting:**
   - Output all benchmark results directly to `/results/tables/` in LaTeX (`.tex`) and Markdown (`.md`) format using `scripts/paper_exporter.py`.
