"""
VLM Orchestrator — Streaming Harness

Simulates real-time video frame delivery and measures detection latency.
This is the test harness candidates run their pipeline against.

The harness:
  1. Reads a video file and emits frames at real-time speed
  2. Calls the candidate's pipeline callback for each frame
  3. Collects emitted events and timestamps when they were emitted
  4. Measures detection delay: wall-clock time from when a video moment
     passed to when the pipeline reported a detection for it

Usage:
    from src.harness import StreamingHarness

    harness = StreamingHarness(
        video_path="path/to/video.mp4",
        procedure_path="data/procedures/change_circuit_breaker.json",
        speed=1.0,  # 1.0 = real-time, 2.0 = 2x speed, etc.
    )

    # Your pipeline registers a frame callback
    harness.on_frame(my_frame_handler)      # called with (frame, timestamp_sec, frame_base64)

    # When your pipeline detects something, it calls:
    harness.emit_event({
        "timestamp_sec": 49.7,
        "type": "step_completion",
        "step_id": 1,
        ...
    })

    # Run the simulation
    results = harness.run()
    # results contains events + detection delays + total time
"""

import json
import time
import io
import base64
import threading
from pathlib import Path
from typing import Callable, Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

import cv2
import numpy as np
from PIL import Image


@dataclass
class EmittedEvent:
    """An event emitted by the candidate's pipeline, with timing metadata."""
    event: Dict[str, Any]          # The event dict the candidate emitted
    wall_time: float               # Wall-clock time (seconds since harness start) when emitted
    video_time_at_emission: float  # What video timestamp the harness had reached when event was emitted
    detection_delay_sec: float     # video_time_at_emission - event["timestamp_sec"]


@dataclass
class HarnessResults:
    """Results from a streaming harness run."""
    task: str
    video_source: str
    procedure_path: str
    speed: float
    start_time: str
    end_time: str
    video_duration_sec: float
    wall_duration_sec: float
    total_frames_delivered: int
    events: List[Dict[str, Any]]     # Events in output schema format (with detection_delay_sec added)
    mean_detection_delay_sec: float
    max_detection_delay_sec: float


class StreamingHarness:
    """
    Simulates real-time video streaming and measures detection latency.

    The harness plays through a video at a configurable speed, delivering
    frames to a registered callback. When the candidate's pipeline detects
    an event, it calls emit_event(). The harness records the wall-clock
    time and computes detection delay.
    """

    def __init__(
        self,
        video_path: str,
        procedure_path: str,
        speed: float = 1.0,
        frame_fps: float = 2.0,
    ):
        """
        Args:
            video_path: Path to MP4 file
            procedure_path: Path to procedure JSON
            speed: Playback speed multiplier (1.0 = real-time, 2.0 = 2x faster)
            frame_fps: How many frames per second to deliver to the pipeline
        """
        self.video_path = video_path
        self.procedure_path = procedure_path
        self.speed = speed
        self.frame_fps = frame_fps

        self._frame_callbacks: List[Callable] = []
        self._emitted_events: List[EmittedEvent] = []
        self._start_wall_time: float = 0
        self._current_video_time: float = 0
        self._lock = threading.Lock()

        # Load procedure
        with open(procedure_path) as f:
            self.procedure = json.load(f)
        self.task_name = self.procedure.get("task") or self.procedure.get("task_name", "Unknown")

    def on_frame(self, callback: Callable[[np.ndarray, float, str], None]):
        """
        Register a frame callback.

        Your callback receives:
            frame: BGR numpy array
            timestamp_sec: current video timestamp
            frame_base64: JPEG-encoded base64 string (ready for VLM API)
        """
        self._frame_callbacks.append(callback)

    VALID_EVENT_TYPES = {"step_completion", "error_detected", "idle_detected"}
    VALID_SOURCES = {"video"}
    VALID_ERROR_TYPES = {"wrong_action", "wrong_sequence", "safety_violation", "improper_technique", "other"}
    VALID_SEVERITIES = {"info", "warning", "critical"}

    def _validate_event(self, event: Dict[str, Any]) -> List[str]:
        """Validate an event against the schema. Returns list of error messages (empty = valid)."""
        errors = []

        # Required fields
        if "timestamp_sec" not in event:
            errors.append("Missing required field: timestamp_sec")
        elif not isinstance(event["timestamp_sec"], (int, float)):
            errors.append(f"timestamp_sec must be a number, got {type(event['timestamp_sec']).__name__}")

        if "type" not in event:
            errors.append("Missing required field: type")
        elif event["type"] not in self.VALID_EVENT_TYPES:
            errors.append(f"Invalid event type: '{event['type']}'. Must be one of {self.VALID_EVENT_TYPES}")

        # Type-specific validation
        event_type = event.get("type")

        if event_type == "step_completion":
            if "step_id" not in event:
                errors.append("step_completion event missing required field: step_id")
            elif not isinstance(event["step_id"], int):
                errors.append(f"step_id must be an integer, got {type(event['step_id']).__name__}")

        if event_type == "error_detected":
            if "error_type" in event and event["error_type"] not in self.VALID_ERROR_TYPES:
                errors.append(f"Invalid error_type: '{event['error_type']}'. Must be one of {self.VALID_ERROR_TYPES}")
            if "severity" in event and event["severity"] not in self.VALID_SEVERITIES:
                errors.append(f"Invalid severity: '{event['severity']}'. Must be one of {self.VALID_SEVERITIES}")

        # Optional field validation
        if "confidence" in event:
            conf = event["confidence"]
            if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                errors.append(f"confidence must be a number between 0 and 1, got {conf}")

        if "source" in event and event["source"] not in self.VALID_SOURCES:
            errors.append(f"Invalid source: '{event['source']}'. Must be one of {self.VALID_SOURCES}")

        return errors

    def emit_event(self, event: Dict[str, Any]):
        """
        Call this from your pipeline when you detect an event.

        The harness records the wall-clock time and computes detection delay.
        Detection delay = wall_clock_elapsed * speed - event_timestamp.
        This measures how far past the event (in video-time) the real world
        has advanced by the time the pipeline reports it.

        You can call this from any thread.

        Args:
            event: Dict matching the event schema (must have timestamp_sec and type)

        Raises:
            ValueError: If the event fails schema validation
        """
        validation_errors = self._validate_event(event)
        if validation_errors:
            error_msg = "; ".join(validation_errors)
            raise ValueError(f"Invalid event: {error_msg}")

        wall_now = time.monotonic() - self._start_wall_time
        # Convert wall time back to video-time equivalent
        video_time_equivalent = wall_now * self.speed
        event_video_time = event.get("timestamp_sec", 0)
        delay = video_time_equivalent - event_video_time

        with self._lock:
            self._emitted_events.append(EmittedEvent(
                event=event,
                wall_time=wall_now,
                video_time_at_emission=video_time_equivalent,
                detection_delay_sec=max(0, delay),
            ))

    @staticmethod
    def frame_to_base64(frame: np.ndarray) -> str:
        """Convert BGR frame to base64 JPEG."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def run(self) -> HarnessResults:
        """
        Run the streaming simulation.

        Delivers frames at real-time speed (adjusted by self.speed).
        Returns results with all emitted events and timing data.
        """
        print(f"{'=' * 60}")
        print(f"  STREAMING HARNESS")
        print(f"{'=' * 60}")
        print(f"  Task:      {self.task_name}")
        print(f"  Video:     {self.video_path}")
        print(f"  Speed:     {self.speed}x real-time")
        print(f"  Frame FPS: {self.frame_fps}")
        print()

        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / video_fps

        # Compute frame interval (in video-time seconds)
        frame_interval = 1.0 / self.frame_fps

        print(f"  Video:     {video_duration:.1f}s @ {video_fps:.0f}fps ({total_frames} frames)")
        print(f"  Delivering ~{int(video_duration * self.frame_fps)} frames to pipeline")
        print()
        print(f"  Starting simulation...")
        print()

        self._start_wall_time = time.monotonic()
        start_dt = datetime.utcnow().isoformat() + "Z"

        frames_delivered = 0
        next_frame_video_time = 0.0

        while next_frame_video_time < video_duration:
            # Seek to the right frame
            frame_number = int(next_frame_video_time * video_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                break

            # Wait until real-time catches up (respecting speed multiplier)
            target_wall_time = next_frame_video_time / self.speed
            elapsed = time.monotonic() - self._start_wall_time
            if elapsed < target_wall_time:
                time.sleep(target_wall_time - elapsed)

            # Update current video time
            with self._lock:
                self._current_video_time = next_frame_video_time

            # Deliver frame
            frame_b64 = self.frame_to_base64(frame)
            for cb in self._frame_callbacks:
                try:
                    cb(frame, next_frame_video_time, frame_b64)
                except Exception as e:
                    print(f"  [harness] Frame callback error at {next_frame_video_time:.1f}s: {e}")

            frames_delivered += 1
            if frames_delivered % 10 == 0:
                print(f"  [{next_frame_video_time:.1f}s / {video_duration:.1f}s] "
                      f"{frames_delivered} frames, {len(self._emitted_events)} events detected")

            next_frame_video_time += frame_interval

        cap.release()

        # Final update
        with self._lock:
            self._current_video_time = video_duration

        wall_duration = time.monotonic() - self._start_wall_time
        end_dt = datetime.utcnow().isoformat() + "Z"

        # Build output events with detection_delay_sec
        output_events = []
        delays = []
        for ee in self._emitted_events:
            ev = dict(ee.event)
            ev["detection_delay_sec"] = round(ee.detection_delay_sec, 3)
            output_events.append(ev)
            delays.append(ee.detection_delay_sec)

        mean_delay = sum(delays) / len(delays) if delays else 0
        max_delay = max(delays) if delays else 0

        print()
        print(f"  {'=' * 56}")
        print(f"  Simulation complete")
        print(f"  {'=' * 56}")
        print(f"  Frames delivered:  {frames_delivered}")
        print(f"  Events detected:   {len(output_events)}")
        print(f"  Wall time:         {wall_duration:.1f}s")
        print(f"  Mean detect delay: {mean_delay:.2f}s")
        print(f"  Max detect delay:  {max_delay:.2f}s")

        return HarnessResults(
            task=self.task_name,
            video_source=self.video_path,
            procedure_path=self.procedure_path,
            speed=self.speed,
            start_time=start_dt,
            end_time=end_dt,
            video_duration_sec=video_duration,
            wall_duration_sec=wall_duration,
            total_frames_delivered=frames_delivered,
            events=output_events,
            mean_detection_delay_sec=round(mean_delay, 3),
            max_detection_delay_sec=round(max_delay, 3),
        )

    def save_results(self, results: HarnessResults, output_path: str):
        """Save results to JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(asdict(results), f, indent=2)
        print(f"  Results saved to: {output_path}")
