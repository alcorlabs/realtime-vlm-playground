# CLAUDE.md — VLM Orchestrator (Alcor Labs take-home)

## Project

Real-time procedural assistant take-home for **Alcor Labs** (wearable VLM startup). Submission deadline: 7 days from assignment receipt. Questions / submission: **elior@alcor-labs.com**.

## The task

Given a per-clip procedure JSON (ordered step list, no timestamps) + a video with audio, stream frames and audio through VLMs (OpenRouter API) and emit timestamped `step_completion` and `error_detected` events. The harness paces delivery to wall-clock time and measures how late each emission is.

Automated score (85pts) + manual (15pts):

```
combined = 0.40 * step_f1 + 0.40 * error_f1 + 0.20 * latency_score
latency_score = max(0, 1 - mean_detection_delay_sec / 10)
```

- Matching tolerance: **±5s**, optimal closest-first bipartite matching.
- `step_completion` matches require **same `step_id`** AND timestamp within tolerance.
- `error_detected` matches by timestamp only.
- Pipeline is re-run on held-out clips at `--speed 1.0` — do not overfit to the 15 training clips.

## Layout

| Path | Role |
|---|---|
| `src/run.py` | **The only file we implement.** `Pipeline.on_frame` / `on_audio` are empty stubs. Contains a working `call_vlm(api_key, frame_base64, prompt, model, stream)` helper. |
| `src/harness.py` | **Do not modify.** Streaming harness. Sync callbacks on one thread — blocking VLM calls inside `on_frame` stall the timeline. `emit_event` is thread-safe. Audio extracted upfront via ffmpeg → 16kHz mono 16-bit PCM, 5s chunks. |
| `src/evaluator.py` | Scorer: precision/recall/F1 + latency. CLI: `python -m src.evaluator --predicted ... --ground-truth ... --tolerance 5` |
| `src/dashboard.py` | HTML timeline (GT vs predictions). Single-clip and multi-clip tabs. |
| `src/data_loader.py` | Procedure/video helpers. |
| `data/clip_procedures/*.json` | 15 per-clip procedures — **pipeline input** (step list, no timestamps). |
| `data/ground_truth_sample/*.json` | Matching GT with step end-times, error start-times, idle periods — for self-testing only. |
| `data/procedures/*.json` | 6 canonical SOPs per task type (reference). |
| `data/schema/event_log.schema.json` | Authoritative output schema. |
| `data/videos_full/` | **Not in repo.** Download from the GDrive link in `README.md`. |
| `thoughts/sessions/` | Scratch space. |

## Output schema crib

Required: `timestamp_sec`, `type` (`step_completion` \| `error_detected` \| `idle_detected`). For `step_completion`: `step_id` is required. For `error_detected`: `spoken_response` is expected; `error_type` ∈ {`wrong_action`, `wrong_sequence`, `safety_violation`, `improper_technique`, `other`}; `severity` ∈ {`info`, `warning`, `critical`}. `source` ∈ {`video`, `audio`, `both`}. `detection_delay_sec` is added by the harness.

## Timing conventions

- **Step completions** are timestamped at the **end** of the step. Emit as soon as possible after the step finishes.
- **Errors** are timestamped at the **start** of the mistake. Instructor's verbal correction typically follows 2–5s later (strong audio signal).
- **Idle** is optional/bonus — unscored.

## Hard constraints / gotchas

- **Harness callbacks are synchronous on one thread.** Any blocking VLM call inside `on_frame` / `on_audio` directly pushes the timeline back. Dispatch VLM work to a thread pool and return immediately from the callback.
- `harness.emit_event` is thread-safe (internal `self._lock`) — workers may call it directly.
- Detection delay is measured from wall-clock emission time; late emission hurts latency score even if the event's `timestamp_sec` field is correct.
- At `--speed 10`, API $ cost is the same as 1× but wall-clock delays scale ×10 in detection delay math. **Validate final numbers at `--speed 1.0`** before submission.
- Audio is **pitch-shifted for privacy** — STT must tolerate that. Run a quick probe before committing to a provider.
- The 15pts for code quality / cost / architecture are manual — keep commits atomic and informative, keep OpenRouter spend tracked.

## Run commands

```bash
make setup                          # create venv + install deps
make dry-run                        # validate env + one procedure

python src/run.py \
  --procedure data/clip_procedures/CLIP.json \
  --video data/videos_full/CLIP.mp4 \
  --output output/CLIP.json \
  --speed 10.0                      # dev: 10x playback

python -m src.evaluator \
  --predicted output/CLIP.json \
  --ground-truth data/ground_truth_sample/CLIP.json \
  --tolerance 5

python -m src.dashboard \
  --predicted output/CLIP.json \
  --ground-truth data/ground_truth_sample/CLIP.json \
  --output output/CLIP.html
```

Final sweep: re-run every clip at `--speed 1.0` and check `mean_detection_delay_sec`.

## Environment

Python 3.11+, ffmpeg (system), `OPENROUTER_API_KEY` env var. Deps in `requirements.txt` (opencv-headless, numpy, requests, Pillow, av, python-dotenv, pytest).

## Current status

- Scaffolding only. `Pipeline.on_frame` / `Pipeline.on_audio` are empty.
- No videos downloaded yet (`data/videos_full/` missing).
- No `output/` yet.

## Approach (plan + hypotheses)

Full plan: `~/.claude/plans/nested-crunching-storm.md`. Summary:

1. **C0** — bootstrap env, download videos, pick 2 iteration clips.
2. **C1** — naive baseline: one prompt/frame, ThreadPoolExecutor, measure.
3. **C2** — procedure-aware state machine (`is step K done?` yes/no).
4. **C3** — activity-gated frame sampling.
5. **C4** — audio STT + instructor-correction keyword detection → error events.
6. **C5** — vision error cross-check, audio+video fusion with timestamp-proximity dedupe.
7. **C6** — streaming outputs + early verdict parsing, capped in-flight calls.
8. **C7** — model tiering (Flash default, stronger model on low-confidence / ambiguous frames).
9. **C8** — generalization sweep across all 15 clips.
10. **C9** — final 1× run, cost audit, git history cleanup.
11. **C10** — 1-page report (architecture, sampling, audio, tiering, cost, latency, bidirectional-streaming redesign) + optional demo.

Key hypotheses to validate along the way:

- **H2**: step-anchored yes/no prompts beat open-ended classification.
- **H4**: audio instructor-correction signal dominates error recall.
- **H6**: streaming + early-termination shaves 0.5–1.5s off mean detection delay.
- **H7**: async dispatch is mandatory — confirmed by reading `harness.py` (single-thread sync callbacks).
- **H11**: sub-5s detection delay is "free" accuracy (within tolerance); after that, each second costs ~2 combined points.
