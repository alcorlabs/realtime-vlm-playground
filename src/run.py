"""
VLM Orchestrator — C1 baseline pipeline with Rich live observability.

Usage:
    python src/run.py --config configs/R066.yaml
"""

import json
import os
import sys
import argparse
import time
import threading
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import requests
import numpy as np
import yaml

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import StreamingHarness
from src.data_loader import load_procedure_json, validate_procedure_format
from src.evaluator import evaluate as evaluate_against_gt


# ==========================================================================
# PRICING — OpenRouter returns $/token in its models endpoint.
# ==========================================================================

FALLBACK_PRICES = {
    "google/gemini-2.5-flash": {"prompt": 3.0e-7, "completion": 2.5e-6},
}


def fetch_model_pricing(model_id: str) -> Dict[str, float]:
    try:
        r = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
        r.raise_for_status()
        for m in r.json().get("data", []):
            if m.get("id") == model_id:
                p = m.get("pricing", {})
                return {
                    "prompt": float(p.get("prompt") or 0),
                    "completion": float(p.get("completion") or 0),
                }
    except Exception:
        pass
    return FALLBACK_PRICES.get(model_id, {"prompt": 0.0, "completion": 0.0})


def get_video_duration_sec(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        return total / fps if fps else 0.0
    finally:
        cap.release()


# ==========================================================================
# VLM CALL — returns (text, prompt_tokens, completion_tokens)
# ==========================================================================

def call_vlm(
    api_key: str,
    frame_base64: str,
    prompt: str,
    model: str = "google/gemini-2.5-flash",
) -> Tuple[str, int, int]:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/alcor-labs/vlm-orchestrator-eval",
        "X-Title": "VLM Orchestrator",
    }
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}},
            ],
        }],
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    return (
        text,
        int(usage.get("prompt_tokens", 0) or 0),
        int(usage.get("completion_tokens", 0) or 0),
    )


# ==========================================================================
# RUN STATE
# ==========================================================================

@dataclass
class RunState:
    clip: str = ""
    model: str = ""
    speed: float = 1.0
    video_duration_sec: float = 0.0
    current_video_time: float = 0.0

    api_calls: int = 0
    api_errors: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    call_latencies: list = field(default_factory=list)
    in_flight: int = 0

    prompts: deque = field(default_factory=lambda: deque(maxlen=5))
    last_response: str = ""

    events_emitted: int = 0
    current_expected_step: int = 1
    total_steps: int = 0
    completed_steps: set = field(default_factory=set)

    def mean_latency(self) -> float:
        return sum(self.call_latencies) / len(self.call_latencies) if self.call_latencies else 0.0


# ==========================================================================
# RICH LIVE DISPLAY
# ==========================================================================

@dataclass
class _DisplaySnapshot:
    clip: str
    model: str
    speed: float
    video_duration_sec: float
    current_video_time: float
    api_calls: int
    api_errors: int
    in_flight: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost_usd: float
    mean_latency: float
    events_emitted: int
    current_expected_step: int
    total_steps: int
    prompts: list
    prompts_maxlen: int
    last_response: str


class RunDisplay:
    def __init__(self, state: RunState, lock: threading.Lock):
        self.state = state
        self.lock = lock

    def _snapshot(self) -> _DisplaySnapshot:
        with self.lock:
            return _DisplaySnapshot(
                clip=self.state.clip,
                model=self.state.model,
                speed=self.state.speed,
                video_duration_sec=self.state.video_duration_sec,
                current_video_time=self.state.current_video_time,
                api_calls=self.state.api_calls,
                api_errors=self.state.api_errors,
                in_flight=self.state.in_flight,
                total_prompt_tokens=self.state.total_prompt_tokens,
                total_completion_tokens=self.state.total_completion_tokens,
                total_cost_usd=self.state.total_cost_usd,
                mean_latency=self.state.mean_latency(),
                events_emitted=self.state.events_emitted,
                current_expected_step=self.state.current_expected_step,
                total_steps=self.state.total_steps,
                prompts=list(self.state.prompts),
                prompts_maxlen=self.state.prompts.maxlen or 5,
                last_response=self.state.last_response,
            )

    def __rich__(self):
        snap = self._snapshot()
        layout = Layout()
        layout.split_column(
            Layout(self._header(snap), name="header", size=3),
            Layout(self._progress(snap), name="progress", size=3),
            Layout(name="body"),
        )
        layout["body"].split_row(
            Layout(self._prompts(snap), name="prompts", ratio=3),
            Layout(self._stats(snap), name="stats", ratio=2),
        )
        return layout

    def _header(self, s: _DisplaySnapshot) -> Panel:
        t = Text()
        t.append("VLM Orchestrator", style="bold cyan")
        t.append("   clip: ")
        t.append(s.clip, style="yellow")
        t.append("   model: ")
        t.append(s.model, style="magenta")
        t.append(f"   speed: {s.speed}x")
        return Panel(t, border_style="cyan")

    def _progress(self, s: _DisplaySnapshot) -> Panel:
        total = s.video_duration_sec or 1.0
        pct = min(s.current_video_time / total, 1.0)
        width = 48
        filled = int(pct * width)
        bar = Text()
        bar.append("█" * filled, style="cyan")
        bar.append("░" * (width - filled), style="dim")
        bar.append(f"  {s.current_video_time:6.1f} / {s.video_duration_sec:6.1f}s  ({pct * 100:5.1f}%)")
        return Panel(bar, title="Video progress")

    def _prompts(self, s: _DisplaySnapshot) -> Panel:
        if not s.prompts:
            body = Text("No prompts sent yet.", style="dim")
        else:
            body = Text()
            for ts, preview in s.prompts:
                body.append(f"{ts:6.1f}s  ", style="yellow")
                body.append(preview + "\n")
            if s.last_response:
                body.append("\n[last response] ", style="dim")
                body.append(s.last_response[:200].replace("\n", " "), style="dim green")
        return Panel(body, title=f"Recent prompts (last {s.prompts_maxlen})")

    def _stats(self, s: _DisplaySnapshot) -> Panel:
        t = Table.grid(padding=(0, 1))
        t.add_column(style="dim", justify="right")
        t.add_column(style="bold")
        errs = f" [red]({s.api_errors} err)[/red]" if s.api_errors else ""
        t.add_row("API calls:", f"{s.api_calls}{errs}")
        t.add_row("In flight:", str(s.in_flight))
        t.add_row("Tokens in:", f"{s.total_prompt_tokens:,}")
        t.add_row("Tokens out:", f"{s.total_completion_tokens:,}")
        t.add_row("Spend:", f"[green]${s.total_cost_usd:.4f}[/green]")
        t.add_row("Mean latency:", f"{s.mean_latency:.2f}s")
        t.add_row("Events:", str(s.events_emitted))
        t.add_row("Step:", f"{s.current_expected_step}/{s.total_steps}")
        return Panel(t, title="Stats")


# ==========================================================================
# TRACKER — thread-safe state mutation + JSONL sidecar log
# ==========================================================================

class RunTracker:
    def __init__(self, state: RunState, pricing: Dict[str, float], log_path: Path):
        self.state = state
        self.pricing = pricing
        self.lock = threading.Lock()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(log_path, "w")
        self._log_fh.write(json.dumps({
            "kind": "run_start",
            "clip": state.clip,
            "model": state.model,
            "speed": state.speed,
            "video_duration_sec": state.video_duration_sec,
            "pricing": pricing,
            "wall_time_sec": time.time(),
        }) + "\n")
        self._log_fh.flush()

    def close(self):
        self._log_fh.close()

    def note_time(self, ts: float):
        with self.lock:
            self.state.current_video_time = ts

    def note_prompt_dispatched(self, ts: float, prompt: str):
        preview = " ".join(prompt.split())[:90]
        with self.lock:
            self.state.prompts.append((ts, preview))
            self.state.in_flight += 1

    def note_call_complete(self, ts: float, prompt_tokens: int, completion_tokens: int,
                           latency: float, error: Optional[str], response: str):
        cost = prompt_tokens * self.pricing["prompt"] + completion_tokens * self.pricing["completion"]
        with self.lock:
            self.state.in_flight -= 1
            self.state.api_calls += 1
            if error:
                self.state.api_errors += 1
            else:
                self.state.total_prompt_tokens += prompt_tokens
                self.state.total_completion_tokens += completion_tokens
                self.state.total_cost_usd += cost
                self.state.call_latencies.append(latency)
                self.state.last_response = response
        self._write({
            "kind": "vlm_call",
            "video_time_sec": ts,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
            "latency_sec": latency,
            "error": error,
            "response": response[:500],
        })

    def note_event(self, event: dict):
        with self.lock:
            self.state.events_emitted += 1
        self._write({"kind": "event", **event})

    def advance_step(self, new_step: int):
        with self.lock:
            self.state.current_expected_step = new_step

    def _write(self, record: dict):
        record["wall_time_sec"] = time.time()
        with self.lock:
            self._log_fh.write(json.dumps(record) + "\n")
            self._log_fh.flush()


# ==========================================================================
# PIPELINE (C1 baseline: rate-limited single-prompt + threadpool)
# ==========================================================================

PROMPT_TEMPLATE = """You are monitoring a technician performing: {task_name}

Procedure steps:
{steps_block}

Current expected step: {current_id}. {current_desc}

Look at this frame. Has the current expected step just been completed? Is the technician making an error?

Respond with ONLY a JSON object (no markdown fencing, no prose), matching exactly:
{{"step_complete": <true|false>, "completed_step_id": <int|null>, "error": <string|null>, "reasoning": "<one short sentence>"}}"""


def extract_json_object(text: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class Pipeline:
    def __init__(
        self,
        harness: StreamingHarness,
        api_key: str,
        procedure: Dict[str, Any],
        tracker: RunTracker,
        model: str,
        call_interval_sec: float,
        max_workers: int = 3,
    ):
        self.harness = harness
        self.api_key = api_key
        self.tracker = tracker
        self.model = model
        self.call_interval_sec = call_interval_sec
        self.task_name = procedure.get("task") or procedure.get("task_name", "Unknown")
        self.steps = procedure["steps"]
        self.steps_block = "\n".join(f"{s['step_id']}. {s['description']}" for s in self.steps)
        self.last_step_id = max(s["step_id"] for s in self.steps)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vlm")
        self._last_call_video_time = -1e9
        self._state_lock = threading.Lock()

    def shutdown(self):
        self.executor.shutdown(wait=True)

    def on_frame(self, frame: np.ndarray, timestamp_sec: float, frame_base64: str):
        self.tracker.note_time(timestamp_sec)

        with self._state_lock:
            if self.tracker.state.current_expected_step > self.last_step_id:
                return
            if timestamp_sec - self._last_call_video_time < self.call_interval_sec:
                return
            self._last_call_video_time = timestamp_sec
            expected_id = self.tracker.state.current_expected_step

        self.executor.submit(self._process, timestamp_sec, frame_base64, expected_id)

    def on_audio(self, audio_bytes: bytes, start_sec: float, end_sec: float):
        pass  # C1: audio ignored

    def _process(self, ts: float, frame_base64: str, expected_id: int):
        current = next((s for s in self.steps if s["step_id"] == expected_id), None)
        if not current:
            return

        prompt = PROMPT_TEMPLATE.format(
            task_name=self.task_name,
            steps_block=self.steps_block,
            current_id=expected_id,
            current_desc=current["description"],
        )
        self.tracker.note_prompt_dispatched(ts, prompt)

        start = time.time()
        try:
            text, in_tok, out_tok = call_vlm(self.api_key, frame_base64, prompt, model=self.model)
            latency = time.time() - start
            self.tracker.note_call_complete(ts, in_tok, out_tok, latency, None, text)
        except Exception as e:
            self.tracker.note_call_complete(ts, 0, 0, time.time() - start, str(e), "")
            return

        parsed = extract_json_object(text)
        if not parsed:
            return

        completed_id = parsed.get("completed_step_id")
        if parsed.get("step_complete") and isinstance(completed_id, int):
            with self._state_lock:
                if completed_id in self.tracker.state.completed_steps:
                    completed_id = None
                else:
                    self.tracker.state.completed_steps.add(completed_id)
                    self.tracker.advance_step(completed_id + 1)

            if completed_id is not None:
                event = {
                    "timestamp_sec": ts,
                    "type": "step_completion",
                    "step_id": completed_id,
                    "confidence": 0.8,
                    "description": parsed.get("reasoning", "") or "",
                    "source": "video",
                    "vlm_observation": text[:500],
                }
                self.harness.emit_event(event)
                self.tracker.note_event(event)

        err = parsed.get("error")
        if err and isinstance(err, str) and err.strip().lower() not in ("none", "null", ""):
            event = {
                "timestamp_sec": ts,
                "type": "error_detected",
                "error_type": "wrong_action",
                "severity": "warning",
                "confidence": 0.6,
                "description": err,
                "spoken_response": f"Hold on — {err}",
                "source": "video",
                "vlm_observation": text[:500],
            }
            self.harness.emit_event(event)
            self.tracker.note_event(event)


# ==========================================================================
# MAIN
# ==========================================================================

def _load_api_key() -> Optional[str]:
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


@dataclass
class RunConfig:
    procedure: str
    video: str
    output: str
    ground_truth: Optional[str] = None
    speed: float = 1.0
    frame_fps: float = 2.0
    audio_chunk_sec: float = 5.0
    model: str = "google/gemini-2.5-flash"
    call_interval_sec: float = 2.0
    max_workers: int = 3
    tolerance_sec: float = 5.0


def load_config(config_path: Path) -> RunConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    clip = raw.get("clip", {})
    output = raw.get("output", {})
    harness = raw.get("harness", {})
    pipeline = raw.get("pipeline", {})
    evaluation = raw.get("evaluation", {})

    missing = [k for k in ("procedure", "video") if not clip.get(k)]
    if missing:
        raise ValueError(f"config {config_path} missing required clip.{missing} fields")

    return RunConfig(
        procedure=clip["procedure"],
        video=clip["video"],
        ground_truth=clip.get("ground_truth"),
        output=output.get("events", "output/events.json"),
        speed=float(harness.get("speed", 1.0)),
        frame_fps=float(harness.get("frame_fps", 2.0)),
        audio_chunk_sec=float(harness.get("audio_chunk_sec", 5.0)),
        model=pipeline.get("model", "google/gemini-2.5-flash"),
        call_interval_sec=float(pipeline.get("call_interval_sec", 2.0)),
        max_workers=int(pipeline.get("max_workers", 3)),
        tolerance_sec=float(evaluation.get("tolerance_sec", 5.0)),
    )


def main():
    parser = argparse.ArgumentParser(description="VLM Orchestrator Pipeline (C1 baseline)")
    parser.add_argument("--config", required=True, help="Path to YAML run config (see configs/).")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without running.")
    parser.add_argument("--no-live", action="store_true", help="Disable Rich live display.")
    args = parser.parse_args()

    console = Console()
    cfg = load_config(Path(args.config))
    procedure = load_procedure_json(cfg.procedure)
    validate_procedure_format(procedure)
    task_name = procedure.get("task") or procedure.get("task_name", "Unknown")
    clip_name = Path(cfg.procedure).stem

    console.print(
        f"[bold]VLM Orchestrator[/bold]   task: [yellow]{task_name}[/yellow]   "
        f"clip: [yellow]{clip_name}[/yellow]   config: [dim]{args.config}[/dim]"
    )

    if args.dry_run:
        console.print(f"[dim]Procedure OK ({len(procedure['steps'])} steps).[/dim]")
        if not Path(cfg.video).exists():
            console.print(f"[yellow]WARNING: video missing: {cfg.video}[/yellow]")
        else:
            console.print(f"[dim]Video OK: {get_video_duration_sec(cfg.video):.1f}s[/dim]")
        if cfg.ground_truth and not Path(cfg.ground_truth).exists():
            console.print(f"[yellow]WARNING: ground_truth missing: {cfg.ground_truth}[/yellow]")
        return

    if not Path(cfg.video).exists():
        console.print(f"[red]ERROR: video not found: {cfg.video}[/red]")
        sys.exit(1)

    api_key = _load_api_key()
    if not api_key:
        console.print("[red]ERROR: OPENROUTER_API_KEY not set (env or .env)[/red]")
        sys.exit(1)

    pricing = fetch_model_pricing(cfg.model)
    console.print(
        f"[dim]Pricing for {cfg.model}: ${pricing['prompt']*1e6:.3f}/M prompt, "
        f"${pricing['completion']*1e6:.3f}/M completion[/dim]"
    )

    video_duration = get_video_duration_sec(cfg.video)

    harness = StreamingHarness(
        video_path=cfg.video,
        procedure_path=cfg.procedure,
        speed=cfg.speed,
        frame_fps=cfg.frame_fps,
        audio_chunk_sec=cfg.audio_chunk_sec,
    )

    state = RunState(
        clip=clip_name,
        model=cfg.model,
        speed=cfg.speed,
        video_duration_sec=video_duration,
        total_steps=len(procedure["steps"]),
        current_expected_step=procedure["steps"][0]["step_id"],
    )

    log_path = Path(cfg.output).with_suffix(".jsonl")
    tracker = RunTracker(state, pricing, log_path)

    pipeline = Pipeline(
        harness=harness,
        api_key=api_key,
        procedure=procedure,
        tracker=tracker,
        model=cfg.model,
        call_interval_sec=cfg.call_interval_sec,
        max_workers=cfg.max_workers,
    )
    harness.on_frame(pipeline.on_frame)
    harness.on_audio(pipeline.on_audio)

    try:
        if args.no_live:
            results = harness.run()
        else:
            display = RunDisplay(state, tracker.lock)
            with Live(get_renderable=display.__rich__, console=console, refresh_per_second=4, screen=False):
                results = harness.run()
    finally:
        pipeline.shutdown()
        tracker.close()

    harness.save_results(results, cfg.output)

    console.print()
    console.print(Rule("[bold]Run summary[/bold]"))
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="dim", justify="right")
    summary.add_column(style="bold")
    summary.add_row("Clip:", clip_name)
    summary.add_row("Model:", cfg.model)
    summary.add_row("Speed:", f"{cfg.speed}x")
    summary.add_row("Video duration:", f"{results.video_duration_sec:.1f}s")
    summary.add_row("Wall duration:", f"{results.wall_duration_sec:.1f}s")
    summary.add_row("Frames delivered:", str(results.total_frames_delivered))
    summary.add_row("API calls:", f"{state.api_calls} ({state.api_errors} errors)")
    summary.add_row("Tokens in/out:", f"{state.total_prompt_tokens:,} / {state.total_completion_tokens:,}")
    summary.add_row("Total spend:", f"[green]${state.total_cost_usd:.4f}[/green]")
    summary.add_row("Mean call latency:", f"{state.mean_latency():.2f}s")
    summary.add_row("Events emitted:", str(state.events_emitted))
    summary.add_row("Mean detection delay:", f"{results.mean_detection_delay_sec:.2f}s")
    summary.add_row("Output JSON:", cfg.output)
    summary.add_row("JSONL log:", str(log_path))
    console.print(summary)

    if not results.events:
        console.print("[yellow]WARNING: no events emitted.[/yellow]")

    if cfg.ground_truth and Path(cfg.ground_truth).exists():
        console.print()
        console.print(Rule("[bold]Evaluation[/bold]"))
        m = evaluate_against_gt(cfg.output, cfg.ground_truth, cfg.tolerance_sec, verbose=False)
        latency_score = max(0.0, 1.0 - (m.mean_detection_delay_sec / 10.0))
        combined = 0.40 * m.step_f1 + 0.40 * m.error_f1 + 0.20 * latency_score

        ev = Table.grid(padding=(0, 2))
        ev.add_column(style="dim", justify="right")
        ev.add_column(style="bold")
        ev.add_row("Step P / R / F1:", f"{m.step_precision:.2f} / {m.step_recall:.2f} / [cyan]{m.step_f1:.3f}[/cyan]   ({m.step_tp}/{m.total_gt_steps} matched)")
        ev.add_row("Error P / R / F1:", f"{m.error_precision:.2f} / {m.error_recall:.2f} / [cyan]{m.error_f1:.3f}[/cyan]   ({m.error_tp}/{m.total_gt_errors} matched)")
        ev.add_row("Detection delay:", f"mean {m.mean_detection_delay_sec:.2f}s   p50 {m.p50_detection_delay_sec:.2f}s   p90 {m.p90_detection_delay_sec:.2f}s")
        ev.add_row("Latency score:", f"[cyan]{latency_score:.3f}[/cyan]")
        ev.add_row("Combined:", f"[bold green]{combined:.3f}[/bold green]   (0.4·step_f1 + 0.4·error_f1 + 0.2·latency)")
        ev.add_row("Cost / F1 pt:", (f"[green]${state.total_cost_usd / max(combined, 1e-6):.4f}[/green]") if combined > 0 else "n/a")
        console.print(ev)


if __name__ == "__main__":
    main()
