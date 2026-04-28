---
name: feedback_design_closure
description: Code design must form closed loops: tuned metrics feed into downstream fusion, and feature switches belong in config with CLI overrides.
type: feedback
originSessionId: c949e9f2-8959-40ef-b750-25d5e4d966ee
---
**Rule:** When designing pipelines with tuning + ensemble/fusion:
1. Tuning must produce metrics that are passed downstream to fusion/weighting logic, not discarded
2. Feature switches (e.g., enable tuning) should live in `config.yaml` as the primary control, with command-line flags as optional overrides
3. After writing a module, verify end-to-end data flow — ensure outputs from one stage are consumed by the next

**Why:** The user had to explicitly ask why `run_fusion` was not receiving `val_wapes`, revealing that the weighted ensemble was silently degrading to simple average. They also asked where the tuning switch should live, validating that config-first + CLI-override is the preferred pattern.

**How to apply:** After refactoring notebook code into modules, trace every variable that crosses stage boundaries (e.g., validation metrics → ensemble weights). Place behavioral toggles in config.yaml so the pipeline is reproducible without memorizing CLI flags. Use CLI flags only for one-off overrides (e.g., `--skip-tuning` for quick debugging).
