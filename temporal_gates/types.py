from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Optional


BBox = tuple[float, float, float, float]
FrameSize = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Detection:
    """One candidate emitted by the small detector."""

    camera_id: str
    class_name: str
    confidence: float
    bbox: BBox
    timestamp: float
    frame_size: Optional[FrameSize] = None  # (width, height)

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id must not be empty")
        if not self.class_name:
            raise ValueError("class_name must not be empty")
        if not isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        x1, y1, x2, y2 = self.bbox
        if not all(isfinite(value) for value in self.bbox) or x2 <= x1 or y2 <= y1:
            raise ValueError(f"invalid bbox: {self.bbox}")
        if self.frame_size is not None:
            width, height = self.frame_size
            if width <= 0 or height <= 0:
                raise ValueError("frame_size values must be positive")


@dataclass(frozen=True, slots=True)
class TrackObservation:
    timestamp: float
    detected: bool
    bbox: Optional[BBox] = None
    confidence: Optional[float] = None


@dataclass(slots=True)
class Track:
    track_id: int
    camera_id: str
    class_name: str
    first_seen: float
    last_seen: float
    last_update: float
    bbox: BBox
    hit_count: int = 1
    miss_count: int = 0
    consecutive_misses: int = 0
    confidence_history: list[float] = field(default_factory=list)
    bbox_history: list[BBox] = field(default_factory=list)
    center_history: list[tuple[float, float]] = field(default_factory=list)
    area_history: list[float] = field(default_factory=list)
    observations: list[TrackObservation] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    @property
    def last_confidence(self) -> float:
        return self.confidence_history[-1]

    @property
    def last_observation_detected(self) -> bool:
        return bool(self.observations and self.observations[-1].detected)

    def record_detection(self, detection: Detection, max_history: int) -> None:
        self.last_seen = detection.timestamp
        self.last_update = detection.timestamp
        self.bbox = detection.bbox
        self.hit_count += 1
        self.consecutive_misses = 0
        self._append_detection(detection, max_history)

    def record_miss(self, timestamp: float, max_history: int) -> None:
        self.last_update = timestamp
        self.miss_count += 1
        self.consecutive_misses += 1
        self.observations.append(TrackObservation(timestamp=timestamp, detected=False))
        self._trim(max_history)

    def _append_detection(self, detection: Detection, max_history: int) -> None:
        x1, y1, x2, y2 = detection.bbox
        self.confidence_history.append(detection.confidence)
        self.bbox_history.append(detection.bbox)
        self.center_history.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        self.area_history.append((x2 - x1) * (y2 - y1))
        self.observations.append(
            TrackObservation(
                timestamp=detection.timestamp,
                detected=True,
                bbox=detection.bbox,
                confidence=detection.confidence,
            )
        )
        self._trim(max_history)

    def _trim(self, max_history: int) -> None:
        for history in (
            self.confidence_history,
            self.bbox_history,
            self.center_history,
            self.area_history,
            self.observations,
        ):
            if len(history) > max_history:
                del history[: len(history) - max_history]


class TriggerReason(str, Enum):
    HIGH_CONFIDENCE = "high_confidence"
    TEMPORAL_PERSISTENCE = "temporal_persistence"
    AREA_VARIATION = "area_variation"


@dataclass(frozen=True, slots=True)
class GateDecision:
    camera_id: str
    track_id: int
    class_name: str
    passed: bool
    already_triggered: bool
    reason: str
    window_hits: int
    window_samples: int
    duration: float
    max_confidence: float
    area_variation: float | None = None


@dataclass(frozen=True, slots=True)
class TriggerEvent:
    event_key: str
    camera_id: str
    track_id: int
    class_name: str
    timestamp: float
    bbox: BBox
    confidence: float
    reason: TriggerReason
    window_hits: int
    duration: float


@dataclass(frozen=True, slots=True)
class ProcessResult:
    camera_id: str
    timestamp: float
    tracks: tuple[Track, ...]
    decisions: tuple[GateDecision, ...]
    events: tuple[TriggerEvent, ...]
    accepted_detection_count: int
    ignored_detection_count: int
    reset_due_to_gap: bool = False
