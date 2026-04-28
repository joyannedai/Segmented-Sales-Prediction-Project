---
name: feedback_engineering
description: User requires all code to follow strict engineering standards: config-driven, logging, reproducible, portable, clean function boundaries, and no orphaned modules.
type: feedback
originSessionId: c949e9f2-8959-40ef-b750-25d5e4d966ee
---
**Rule:** All code integrations must follow these engineering standards:
1. Clear directory structure (src/, logs/, output/, config.yaml)
2. Dedicated log files and config parameter files
3. Standardized naming; each function does exactly one thing
4. Reproducible results — random seeds must be set for all stochastic operations
5. Portable Python dependencies — requirements.txt with pinned versions
6. Keep it concise and simple; avoid premature abstractions
7. **Written code must actually be called** — do not leave modules orphaned in the source tree
8. **Default parameters must have a documented source** — either extracted from original code, or explicitly labeled as empirical placeholders

**Why:** The user explicitly stated these requirements for their capstone code submission. They want code that can run manually with input data and produce predictions, and work on any machine. Additionally, the user discovered that `src/tuning.py` was written but never invoked by `main.py`, and that default hyperparameters in `tree_models.py` were arbitrary empirical values rather than extracted from the original notebooks.

**How to apply:** When integrating or refactoring code, always create/update config.yaml, requirements.txt, and a logger. Split notebook cells into single-purpose functions. Set random seeds via a centralized utility. Never leave hardcoded absolute paths in source code. After writing a new module, trace the call chain from `main.py` to confirm it is actually invoked. If you introduce default values (especially model hyperparameters), note their source in comments or docstrings — do not silently use untuned hand-picked values.
