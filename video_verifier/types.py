from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


BBox = tuple[float, float, float, float]
SUPPORTED_CLASSES = ("fire", "smoke")


@dataclass(frozen=True, slots=True)
class ClipQuality:
    frequency_valid: bool
    actual_frame_count: int
    requested_frame_count: int
    effective_fps: float
    max_frame_gap: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VerificationRequest:
    event_id: str
    camera_id: str
    track_id: int
    candidate_class: str
    frames: tuple[Any, ...]
    timestamps: tuple[float, ...]
    bboxes: tuple[BBox, ...]
    confidences: tuple[float, ...]
    observed: tuple[bool, ...]
    frame_size: tuple[int, int]
    quality: ClipQuality

    def __post_init__(self) -> None:
        if not self.event_id or not self.camera_id:
            raise ValueError("event_id and camera_id must not be empty")
        if self.candidate_class not in SUPPORTED_CLASSES:
            raise ValueError(f"unsupported candidate class: {self.candidate_class}")
        lengths = {
            len(self.frames),
            len(self.timestamps),
            len(self.bboxes),
            len(self.confidences),
            len(self.observed),
        }
        if len(lengths) != 1 or not self.frames:
            raise ValueError("all clip sequences must be non-empty and have equal length")
        if any(not isfinite(value) for value in self.timestamps):
            raise ValueError("timestamps must be finite")
        if any(not 0.0 <= value <= 1.0 for value in self.confidences):
            raise ValueError("confidences must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class VerificationResult:
    event_id: str
    camera_id: str
    track_id: int
    candidate_class: str
    p_real: float
    clip_positive: bool
    threshold: float
    model_version: str
    quality: ClipQuality
    inference_ms: float


@dataclass(frozen=True, slots=True)
class AlarmDecision:
    event_id: str
    camera_id: str
    track_id: int
    candidate_class: str
    alarm: bool
    positive_votes: int
    window_size: int
    mean_probability: float
    timestamp: float
