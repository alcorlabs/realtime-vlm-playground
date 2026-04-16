# Session 1 — VLM Orchestrator scaffolding + C1 baseline

**Date:** 2026-04-15
**Branch:** master
**Starting commit:** `1de3608 VLM Orchestrator — take-home assignment`

## Goal

Alcor Labs take-home: build a real-time procedural assistant that streams
frames + audio from a video of a technician and emits `step_completion` /
`error_detected` events via VLM calls. Scoring: `0.40·step_f1 +
0.40·error_f1 + 0.20·latency_score`. This session was the first real attempt:
stand up a C1 baseline end-to-end and make it observable, then learn from
running it.

## What we built

### 1. Docs + plan

- `CLAUDE.md` at repo root: project context, layout, output schema, timing
  conventions, hard gotchas (synchronous harness callbacks, speed-scaled
  detection delay, pitch-shifted audio), run commands, current status.
- Plan file: `~/.claude/plans/nested-crunching-storm.md` — checkpoints
  **C0–C10** and hypotheses **H1–H11**. Treat this as the running roadmap
  across sessions.

### 2. YAML-driven config

`configs/R066.yaml` replaces long CLI commands. Sections: `clip`
(procedure + video + ground_truth), `output`, `harness` (speed, frame_fps,
audio_chunk_sec), `pipeline` (model, call_interval_sec, max_workers),
`evaluation` (tolerance_sec).

Run is now just: `python src/run.py --config configs/R066.yaml`.

### 3. C1 baseline pipeline (`src/run.py`, ~670 lines)

- `Pipeline.on_frame` rate-limits on **video time** at `call_interval_sec`
  and dispatches to a `ThreadPoolExecutor` — never blocks the harness thread.
- Prompt asks for strict JSON: `{step_complete, completed_step_id, error,
  reasoning}`. State machine tracks `current_expected_step` and
  `completed_steps`; emits `step_completion` events when VLM confirms, and
  `error_detected` events when `error` is a non-empty string.
- **Sidecar JSONL log** (`output/<clip>.jsonl`): every VLM call's full prompt
  metadata, latency, token counts, cost, and truncated response — audit trail
  independent of what the harness emits.
- Env loading: reads `OPENROUTER_API_KEY` from env or `.env`.

### 4. Rich live dashboard

`RunDisplay` + `RunTracker`:
- Header (clip / model / speed)
- Video-progress bar (current_video_time / video_duration)
- Recent-prompts panel (last 5 dispatched, + last response preview)
- Stats panel: API calls, in-flight, tokens in/out, spend USD (live pricing
  from `/api/v1/models`), mean latency, events emitted, step K/N

Auto-evaluation fires when `evaluation.ground_truth` is set: prints step P/R/F1,
error P/R/F1, detection delay (mean/p50/p90), latency score, combined score,
and $/F1-point.

### 5. Environment

- `make setup` blocked on Ubuntu 24.04 (PEP 668). Worked around with a
  project-local venv: `python3 -m venv venv && ./venv/bin/pip install -r
  requirements.txt`. Added `rich>=13.7` and `PyYAML>=6.0` to requirements.

## Bugs found (in order)

### Bug #1 — Rich "live stats don't update" (first misdiagnosis: race)

- **Symptom:** stats panel redrew but always read `API calls: 0, In flight: 3,
  Tokens in: 0` while workers were clearly completing.
- **First theory:** worker threads mutate `state.prompts` deque and
  `call_latencies` list without the lock while Rich's refresh thread reads.
- **Fix attempted:** added `_DisplaySnapshot` dataclass; `RunDisplay._snapshot()`
  acquires `tracker.lock` and copies state. Kept this fix — it *is* the
  correct threading discipline, but it wasn't what broke the display.

### Bug #1 — Rich display (second misdiagnosis: VLM hang)

- When the user pasted a stats snapshot, I misread "In flight: 3, API calls: 0"
  as the VLM calls hanging. Ran latency probes on five models; all 120–180 s.
- User corrected: "Its the UI. The model works fine. so does open router. I
  was showing that despite working fine the ui still shows 0 tokens."

### Bug #1 — Rich display (actual root cause)

- Wrote a counter wrapping `__rich__`. **Rich's `Live(renderable)` calls
  `__rich__` exactly once at init**, caches the returned `Layout` tree, and
  redraws the same object on every refresh tick. Since our panels (`_header`,
  `_stats`, etc.) returned *new* Panel instances via `__rich__`, but `Live`
  never re-invoked `__rich__`, the snapshot was frozen at t=0.
- **Fix:** pass `get_renderable=display.__rich__` (a callable) instead of
  `display` (the object) to `Live`. Rich re-invokes the callable on every
  refresh tick. Verified: 9 renders in 1.8 s ≈ the configured
  `refresh_per_second=4` (plus some extras from state changes).
- **Lesson for memory / CLAUDE.md:** with Rich `Live`, if your renderable must
  re-read mutable state on each tick, pass it as `get_renderable=callable`,
  not as the object itself.

### Bug #2 — `output/R066.json` has `events: []` even when workers emit

- Harness snapshots `self._emitted_events` into `HarnessResults.events` the
  moment the delivery loop ends (`src/harness.py:394-399`).
- At speed=10 on 176 s video, the delivery loop ends at wall ≈ 17.6 s. Every
  VLM call takes ~72 s wall. So when the loop ends, **zero** VLM calls have
  returned; workers' `harness.emit_event()` calls land on
  `harness._emitted_events` *after* `results` is already frozen.
- **Proof from `output/R066.jsonl`:** 9 `vlm_call` records present, including
  one with `step_complete: true, completed_step_id: 1` at video-time t=14s
  and one with a non-null `error` at t=6s. The responses exist — they just
  never made it into `results.events`.
- **Not yet applied.** The fix is either:
  (a) after `pipeline.shutdown(wait=True)`, rebuild `results.events` from
  `harness._emitted_events` before `harness.save_results(...)`; or
  (b) block the last `on_frame` callback until the executor has drained, so
  the loop end waits for stragglers.
  (a) is cheaper and doesn't need harness modification.

### Bug #3 — VLM latency is 60–72 s per call

- Even after fix (b) to Bug #2, latency_score → 0 because detection delay =
  `wall_elapsed × speed - timestamp_sec`, and 72 s × 10 = 720 s delay per
  event. The evaluator gives latency_score = max(0, 1 - delay/10).
- Didn't narrow the cause this session. Possible culprits: OpenRouter
  provider routing to a slow backend, image payload size, no `provider.order`
  pin, or just Gemini 2.5 Flash behind a congested gateway.

## Why the current run is hanging (also: why only 9 JSONL records)

Same root cause, two symptoms.

- During delivery (wall 0–17.6 s), `Pipeline.on_frame` rate-limited on video
  time at `call_interval_sec=2.0` → submitted **~88 tasks** to a
  `ThreadPoolExecutor(max_workers=3)`.
- Each VLM call takes ~72 s. So the executor drains 3 calls per ~72 s.
- 9 completed calls = 3 batches × 3 workers ≈ 216 s since first dispatch.
  That's exactly what the JSONL shows.
- `pipeline.shutdown(wait=True)` in the `finally` block waits for *all
  queued tasks* (not just running). Remaining ≈ 88 − 9 = **79 more calls**
  → (79 / 3) × 72 s ≈ **32 min** to drain.
- Ctrl+C does not interrupt a blocking `requests.post` inside a worker, so
  the shutdown won't exit cleanly. `kill -9 <pid>` is the way out right now.
- Longer-term fixes:
  - Switch to `executor.shutdown(wait=False, cancel_futures=True)` (3.9+) on
    abnormal exit — abandons queued tasks.
  - Cap submissions: if the executor has more than N in-flight / queued,
    drop the frame instead of submitting.
  - Reduce latency (Bug #3): get the 72 s → <5 s first, then queue depth
    stops mattering.

## Numbers from the run in `output/R066.json[l]`

- Video: 176.0 s, speed 10.0, delivered 353 frames, 0 audio chunks (C1
  ignores audio).
- Wall duration (harness loop only): 46.3 s (this is suspicious — should be
  ~17.6 s at speed 10; the extra time is probably the harness waiting on
  something, worth investigating).
- Events in `results`: 0 (Bug #2).
- Events actually emitted by workers (JSONL evidence): at least 2 so far (one
  step-complete, one error), more pending in the queue.
- API calls completed: 9. Mean call latency: ~65 s. Token usage: two prompt
  sizes observed — 504 tokens early, 2051 tokens later (procedure block is
  being re-sent each call; cacheable context would help).

## Open items for the next session

1. **Kill/accept the running hang.** `kill -9` the process.
2. **Apply the event-rebuild fix (Bug #2)** — rebuild `results.events` from
   `harness._emitted_events` after `pipeline.shutdown()`.
3. **Cap executor backlog.** Submit only if `in_flight < max_workers`; drop
   the frame otherwise. This also removes the shutdown hang.
4. **Diagnose VLM latency (Bug #3).**
   - Probe with `provider: {"order": ["Google"]}` or other provider pins.
   - Try smaller images (downscale before base64 encode).
   - Try `google/gemini-2.5-flash-lite` or direct Gemini API.
   - Baseline the actual round-trip with a curl probe to isolate OpenRouter
     vs model vs our payload.
5. **Investigate harness wall-duration = 46 s (vs expected 17.6 s).**
6. **Once latency is acceptable, resume checkpoint plan at C2** (procedure-
   aware state machine has already been partially implemented — revisit
   once we can see a real score).
