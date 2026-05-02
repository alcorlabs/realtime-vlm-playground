<claude-mem-context>
# Memory Context

# [realtime-vlm-playground-1] recent context, 2026-05-02 5:45pm EDT

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 19 obs (6,712t read) | 170,326t work | 96% savings

### Apr 30, 2026
154 4:55p 🔵 Challenge Project: Real-Time Streaming Step Detection
156 " 🔵 realtime-vlm-playground: Full Architecture and Data Format Traced
158 7:42p ⚖️ realtime-vlm-playground: Rolling Frame Buffer Architecture for Step/Error Detection
160 8:00p ⚖️ realtime-vlm-playground: Minimal First Baseline Architecture Decided
162 8:01p 🔵 realtime-vlm-playground: src/run.py Pipeline Template Structure
164 " 🟣 realtime-vlm-playground: Rolling Buffer Pipeline Implemented in src/run.py
166 " 🔵 realtime-vlm-playground: No virtualenv present; system python3 missing requests module
170 8:08p ✅ realtime-vlm-playground: VLM Prompt Near-Future Steps Lookahead Reduced from 4 to 2
171 8:14p 🟣 realtime-vlm-playground: VLM Response Now Includes Structured Status Field
172 8:24p 🔵 realtime-vlm-playground: Primary Experiment Target and Run Command
174 8:25p 🔵 realtime-vlm-playground: R066 Pipeline Run Output — Full Event Sequence
176 8:26p 🔵 R066 Ground Truth Format: Step Timings, Errors, and Idle Periods
177 " 🔴 src/run.py: Step Completion Timestamp Snapped to Window End + Prompt Clarified
180 " 🔴 src/run.py: VLM Error Prompt Enhanced for Wrong Toolbox/Part Detection
182 8:34p 🔵 R066 Pipeline Run 2: Step Ordering Regression and Spurious Errors After Prompt Fix
183 8:41p 🟣 src/run.py: Ordered Step Emission Replaces Per-Event Step Emission
185 8:49p 🟣 src/run.py: Debug JSONL Logging for Event-Bearing VLM Calls
186 8:50p 🔵 src/run.py: File State Reverted — Prior Refactor Changes Lost
187 8:51p 🟣 src/run.py: Console Debug Printing for VLM Event Proposals vs Emissions

Access 170k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>

## Experiment Notes

### 2026-05-01: R066 Current Pipeline Baseline

Command:

```bash
.venv/bin/python src/run.py \
  --procedure data/clip_procedures/R066-15July-Circuit-Breaker-part2.json \
  --video data/videos_full/R066-15July-Circuit-Breaker-part2/Export_py/Video_pitchshift.mp4 \
  --output output/R066-events-current.json \
  --speed 10.0
```

Evaluation:

```text
STEP COMPLETION
Precision: 45.5%
Recall:    45.5%
F1:        0.455
5/11 matched, 6 FP, 6 FN

ERROR DETECTION
Precision: 0.0%
Recall:    0.0%
F1:        0.000
0/6 matched
```

Step timing detail:

```text
1:  -2.153s   matched
2:  +2.151s   matched
3:  -6.157s   early
4: -27.216s   early
5: -35.845s   early
6: -34.491s   early
7:  -1.349s   matched
8:  -5.265s   just outside tolerance
9: -23.451s   early
10: -4.883s   matched
11: -1.216s   matched
```

Conclusions:

- The VLM can recognize many target actions, but it often emits visually similar switch/door steps much too early.
- Step timing detail is more useful than aggregate F1 alone; it shows that steps 4, 5, 6, and 9 are the major timing failures in this run.
- Error detection remains at zero without audio; R066 errors are mostly wrong-toolbox/wrong-part moments that likely need instructor correction transcripts or stronger multimodal context.
- The current direct event-emission approach is partially viable for step recognition, but needs temporal confirmation/state modeling before it can reliably match completion timestamps.

### 2026-05-01: Hybrid OpenRouter STT + Audio Error Path

Implementation:

- All API calls now use `OPENROUTER_API_KEY`: chat/VLM calls use OpenRouter chat completions, and STT uses OpenRouter `/api/v1/audio/transcriptions`.
- Audio chunks are transcribed asynchronously, then paired with frames from only the exact same 5s audio window for error-only VLM calls.
- The harness now supports `on_complete()` so pending async audio work can finish before results are written.
- Adding recent transcript history to the audio-error prompt was important. Without it, chunks like "to the right" lacked the prior instruction needed to understand the correction.

Best run so far:

```bash
.venv/bin/python src/run.py \
  --procedure data/clip_procedures/R066-15July-Circuit-Breaker-part2.json \
  --video data/videos_full/R066-15July-Circuit-Breaker-part2/Export_py/Video_pitchshift.mp4 \
  --output output/R066-events-audio-context.json \
  --speed 10.0
```

Evaluation:

```text
STEP COMPLETION
Precision: 40.0%
Recall:    36.4%
F1:        0.381
4/11 matched, 6 FP, 7 FN

ERROR DETECTION
Precision: 20.0%
Recall:    16.7%
F1:        0.182
1/6 matched, 4 FP, 5 FN
```

Step timing detail:

```text
1:  -5.153s   just outside tolerance
2: +20.151s   late
3: +20.843s   late
4:  -3.216s   matched
5:  -5.845s   just outside tolerance
6:  -1.491s   matched
7:  -1.349s   matched
8:  -5.265s   just outside tolerance
9:  -5.451s   just outside tolerance
10: -4.883s   matched
11:    n/a    missing prediction
```

Conclusions:

- Audio integration is viable for surfacing the early wrong-toolbox cluster: the matched error came from the 10-15s chunk after adding recent transcript continuity.
- Raw audio-triggered error calls are still noisy. They need stricter correction detection, not just "instruction changed" detection.
- Step completion did not materially improve with the audio path and remains a separate visual timing/state problem.
- Running `google/gemini-2.5-pro` for every visual and audio VLM call was too slow for this loop. A stronger model should be used selectively, e.g. only for uncertain/error-candidate windows.

### 2026-05-01: Visual Context Prompting Findings

Current direction:

- Runtime is back to a visual-only baseline. Instructor audio/STT is kept out of `src/run.py` because it can leak correction/error labels.
- The visual VLM now receives 5s windows, previous step-wise visual context, and recent window descriptions.
- The prompt asks the VLM to describe the first-person POV action, object appearance, object motion, uncertainty, and relevance before emitting events.

Latest short-window check:

```bash
.venv/bin/python src/run.py \
  --procedure data/clip_procedures/R066-15July-Circuit-Breaker-part2.json \
  --video output/R066-first35s.mp4 \
  --output output/R066-events-visual-context-first35-current.json \
  --speed 10.0 \
  --vlm-log output/R066-visual-context-first35-current.jsonl \
  --vlm-log-start 0 \
  --vlm-log-end 35
```

Observed behavior:

- The VLM no longer treats early red-toolbox handling as step 1 completion.
- It correctly describes the first-person view: hands, camera motion, and manipulated objects are treated as the student's action.
- It can now describe useful temporal changes across consecutive windows, e.g. the small red toolbox is lifted/repositioned in one window, then lowered/resting on the floor in the next.
- Prompting reduced one major hallucination: the model now says the small red toolbox lid does not visibly open, instead of stating that it opened.
- The model distinguishes the small red floor toolbox from the larger red `PRO STEEL` drawer toolbox when the view moves to the larger toolbox.

Current conclusion:

- We are able to prompt the VLM into producing the right kind of visual observations: object identity, state changes, motion, uncertainty, and step relevance are much better than the initial raw event prompt.
- The remaining hard part is not visual description; it is deciding when those observations should become `error_detected`.
- Error detection is still too brittle because many wrong-action cases look like normal searching/preparation without audio or stronger task-level reasoning.
- The next useful architecture is likely a two-layer approach: first collect grounded visual context, then run a stricter event decision layer over that context and current frames. The decision layer should only emit errors when the current frames show a clear contradiction or wrong object/action, not merely because prior context speculated one path and the student moved elsewhere.

### 2026-05-01: R066 Visual Step Definition

Core visual-only rubric:

- A step is completed only when the required final visual state is visible.
- `start_sec` includes searching, preparation, mistakes, corrections, and partial manipulation. `end_sec` is the detection target for completion.
- Prior visual context can explain continuity, but it cannot prove completion by itself.
- Touching, searching, approaching, handling a container, or beginning manipulation should remain `step_in_progress`.
- Object acquisition steps require the target object itself to be visible in the student's hand or clearly controlled by the student.
- Door, switch, and panel steps require the final changed state to be visible after manipulation.
- Errors can occur inside a step without ending that step. For example, wrong-toolbox actions during step 1 are errors, but step 1 continues until the actual circuit breaker is grabbed.

Working mental model:

```text
A step completion is the first frame where the current expected step's required
final state is visually true. Prior context explains how the student got there,
but current-frame evidence must prove the completion.
```

### 2026-05-01: R066 Full Visual Context Run Failure Analysis

Run:

```bash
.venv/bin/python src/run.py \
  --procedure data/clip_procedures/R066-15July-Circuit-Breaker-part2.json \
  --video data/videos_full/R066-15July-Circuit-Breaker-part2/Export_py/Video_pitchshift.mp4 \
  --output output/R066-events-visual-context-full.json \
  --speed 10.0 \
  --vlm-log output/R066-visual-context-full.jsonl \
  --vlm-log-start 0 \
  --vlm-log-end 176
```

Evaluation:

```text
STEP COMPLETION
Precision: 0.0%
Recall:    0.0%
F1:        0.000
0/11 matched, 2 FP, 11 FN

Step timing:
1:  -5.153s   just outside tolerance
2: +112.651s  late
3-11: missing prediction
```

Main diagnosis:

- The VLM produced useful visual descriptions, but emitted only two step events.
- Step 1 was visually described and emitted at 44.5s, but GT completion is 49.653s. This is just outside tolerance because the model used first clear object acquisition, while the annotation appears slightly later after moving away with the breaker.
- The 45.0-49.5s VLM response failed JSON parsing because the model put literal newlines inside string values. That call said step 1 was complete and step 2 was being prepared, but it emitted no event.
- Step 2 was never correctly detected near 57.349s. The VLM described internal component manipulation as relevant preparation, but did not know what final visual state completes "turns on the circuit breaker box."
- Because ordered step gating requires step 2 before steps 3+, the pipeline stayed stuck on current expected step 2 for the rest of the video.
- Later windows often described the correct visual facts for later steps, e.g. door closed, main disconnect moved, breaker removed, breaker returned to toolbox, second breaker inserted, but the VLM framed them as still related to step 2 instead of emitting later step completions.

Conclusion:

- The prompt is good enough to produce visual context, but not good enough to map visual context to granular step completion events.
- The step section needs an explicit "active step reasoning" instruction: compare the current frames against the expected step's required final state, decide whether that state is now visually true, and explain why it does or does not complete the step.
- The procedure descriptions are too semantically overlapping for the model without per-step completion criteria. In R066, "turns on the circuit breaker box," "turns on the main power," and "turns on the circuit_breaker" are visually distinct but linguistically easy to collapse.

### 2026-05-02: R142 Strict Rubric Sampling Sweep

Code change:

- Added `--temperature` and `--top-p` to `src/run.py`.
- These values are passed through to the OpenRouter chat-completions payload for multi-frame VLM calls.
- Experiments below used strict/hard step rubrics on `data/clip_procedures/R142-31Aug-RAM.json`.

Experiment command shape:

```bash
.venv/bin/python src/run.py \
  --procedure data/clip_procedures/R142-31Aug-RAM.json \
  --video data/videos_full/R142-31Aug-RAM/Export_py/Video_pitchshift.mp4 \
  --output output/R142-events-strict-<config>-run<N>.json \
  --speed 10.0 \
  --step-rubric output/step_rubrics/R142-31Aug-RAM.json \
  --step-rubric-mode strict \
  --temperature <temp> \
  --top-p <top_p>
```

Results:

```text
temperature=0.0, top_p=1.0
run 1: F1 0.143, 1/13 matched
run 2: F1 0.143, 1/13 matched
run 3: F1 0.143, 1/13 matched
average F1: 0.143

temperature=0.2, top_p=0.5
run 1: F1 0.143, 1/13 matched
run 2: F1 0.538, 7/13 matched
run 3: F1 0.143, 1/13 matched
average F1: 0.275

temperature=0.7, top_p=0.9
run 1: F1 0.500, 5/13 matched
run 2: F1 0.143, 1/13 matched
run 3: F1 0.381, 4/13 matched
average F1: 0.341

temperature=1.0, top_p=1.0
run 1: F1 0.462, 6/13 matched
run 2: F1 0.143, 1/13 matched
run 3: F1 0.538, 7/13 matched
average F1: 0.381
```

Main observations:

- `temperature=0.0, top_p=1.0` is reproducible, but reproducibly bad. All three runs emitted step 1 and then missed step 2, so ordered gating blocked the rest of the sequence.
- Moderate/high sampling sometimes helps the model escape the step-2 miss, but it is not reliable. The same config can produce either a full-sequence run or a blocked run.
- `temperature=1.0, top_p=1.0` had the best average in this sweep and resembles the earlier default/provider behavior, but it still collapsed in one of three runs.
- Higher sampling also increases output instability risk, including occasional JSON parse failures seen in the high-sampling runs.
- The dominant failure is not the rubric quality on R142. The VLM often proposes correct later steps even when an earlier step is pending. The state manager discards those proposals because strict ordered gating cannot recover from one missed current step.

Conclusion:

- Sampling controls affect reproducibility, but they do not solve the real bottleneck.
- Deterministic decoding can lock the model into a bad interpretation.
- The next architecture improvement should be pending/catch-up step handling: store blocked future-step proposals instead of discarding them, then replay/commit them when the missing earlier step is recovered or when enough evidence accumulates.

### 2026-05-02: R142 Lifecycle Rubric + Phase-Gate Experiments

Recent changes:

- Rubric generation now asks for lifecycle fields: `state_start_visual`, `state_during_visual`, `state_end_visual`, `not_completion`, `timestamp_target`, and `ambiguities`.
- Runtime prompts now ask every `step_completion` to include `matched_phase`, `rubric_reference`, and `completion_reasoning` so we can inspect why the VLM believed a step completed.
- `matched_phase` is now intentionally limited to `state_end_visual` or `catch_up` for emitted completions. The earlier `mechanical_completion` escape hatch was removed because models misused it for non-mechanical sustained-contact steps.
- Visual wrong-sequence errors were removed from the prompt/runtime framing. Apparent later-step progress should be treated as possible missed prior completion/state drift, not as an error event.

Key findings:

- R142 step 4, "touches the metal of the computer tower," exposed a rubric ambiguity. Models naturally treat first/sustained metal contact as completion because the procedure says "touches"; the benchmark timestamp appears to mark the end/release of sustained contact.
- The manual lifecycle mock should avoid broad end-state wording such as "transitions from touching the metal chassis to the next action or a waiting/silence interval" because waiting while still touching can be misread as completion. Better wording is "after sustained contact, the hand clearly breaks contact and moves away from the metal touch point."
- Adding explanation fields improves debuggability but does not by itself constrain emissions. Models can still cite the wrong rubric field or invent a post-hoc reason unless the prompt forces phase classification before emission.
- The `mechanical_completion` phase improved some RAM insertion matches, but it also gave the VLM a broad escape hatch. In one R142 run it labeled step 4 as `mechanical_completion`, even though sustained-contact steps were explicitly excluded. This confirms that phase values should remain simple unless code-side validation enforces them.
- Gemini 3.1 Flash Image Preview produced mixed results: it sometimes matched more RAM insertion steps than Gemini 2.5 Flash, but it was slower, occasionally emitted invalid event types, and still mishandled step 4 without tighter phase/rubric enforcement.

Current working hypothesis:

```text
Keep the rubric lifecycle strict and auditable first. If RAM insertion becomes
too conservative again, add a narrow, code-validated mechanism for hard-to-see
mechanical steps rather than a broad prompt-only phase.
```
