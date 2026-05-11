# Decisions

A running record of decisions made while building this pipeline. New entries go at the **top**.
Each entry follows the same shape: **What / Why / Alternatives / Outcome**. Keep entries short — half a page is plenty. Write them as you make decisions, not at the end of the week.

See `README.md` → "Technical Report & Decision Log" for what we score on.

---

## YYYY-MM-DD — (replace with your first real entry)

Use the example below as a template. Delete this placeholder once you have your own entries. The first 3–5 entries usually cover: initial pipeline scaffolding, first end-to-end run, first prompt iteration, first model swap, first cost-cutting change.

---

## 2026-05-03 — Switched cheap-path model from GPT-4o-mini to Qwen2-VL-7B *(example entry — delete me)*

**What.** Replaced the per-frame "is anything happening?" classifier from `gpt-4o-mini` to `qwen-2-vl-7b-instruct` via OpenRouter.

**Why.** On clip `R042-circuit-breaker`, GPT-4o-mini missed 3/5 wrench-vs-screwdriver disambiguations and triggered two false `error_detected` events on the dashboard. Qwen2-VL-7B got all 5 right in a quick offline check on the same frames, at ~1/4 the per-call cost. Cheap-path quality was the bottleneck, not the expensive-path verifier.

**Alternatives considered.**
- **Keep GPT-4o-mini, add a second-look pass with GPT-4o on low-confidence frames** — rejected: doubles latency on exactly the frames that matter, and the failures weren't low-confidence, they were confidently wrong.
- **Drop to per-2-second sampling instead of per-1-second** — rejected: would have hidden the symptom (fewer chances to be wrong) without fixing the underlying tool-recognition gap.
- **Fine-tune a small classifier on the training clips** — rejected for scope: 7-day window, no labelled per-frame data, and the procedure JSONs change per task type so a clip-specific model wouldn't generalize to the held-out test clips.

**Outcome.**
- Step F1 on the 15 training clips: **0.71 → 0.78** (aggregate).
- Error F1: **0.52 → 0.64**.
- Mean latency unchanged (Qwen response time was within 50ms of GPT-4o-mini in my tests).
- Cost per clip: **$0.41 → $0.13**.
- One regression: clip `R009-belt-tension` lost 1 step detection. Acceptable trade vs. the across-the-board gain; flagged for follow-up if I have time.

---
