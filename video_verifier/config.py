from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class VerifierRuntimeConfig:
    clip_seconds: float = 3.2
    buffer_fps: float = 10.0
    clip_frames: int = 32
    rgb_frames: int = 16
    image_size: int = 224
    roi_scale: float = 1.5
    minimum_frequency_fps: float = 8.0
    minimum_frequency_frames: int = 26
    maximum_frame_gap: float = 0.25
    verification_fps: float = 10.0
    window_stride_seconds: float = 0.5
    vote_window: int = 3
    minimum_positive_votes: int = 2
    maximum_verification_seconds: float = 10.0
    track_grace_seconds: float = 1.25
    reassociation_iou_threshold: float = 0.05
    reassociation_center_distance: float = 1.0
    alarm_cooldown_seconds: float = 10.0
    global_alarm_cooldown_seconds: float = 10.0
    thresholds: dict[str, float] = field(
        default_factory=lambda: {"fire": 0.5, "smoke": 0.5}
    )

    def __post_init__(self) -> None:
        if self.clip_seconds <= 0.0 or self.buffer_fps <= 0.0:
            raise ValueError("clip duration and buffer FPS must be positive")
        if self.clip_frames < 2 or not 1 <= self.rgb_frames <= self.clip_frames:
            raise ValueError("invalid clip frame counts")
        if self.clip_frames != 32 or self.rgb_frames != 16:
            raise ValueError("v1 verifier requires 32 clip frames and 16 RGB frames")
        if self.image_size != 224 or self.roi_scale < 1.0:
            raise ValueError("v1 verifier requires image_size=224 and roi_scale >= 1")
        if self.minimum_frequency_fps <= 0.0:
            raise ValueError("minimum_frequency_fps must be positive")
        if not 2 <= self.minimum_frequency_frames <= self.clip_frames:
            raise ValueError("minimum_frequency_frames is outside the clip")
        if self.maximum_frame_gap <= 0.0 or self.window_stride_seconds <= 0.0:
            raise ValueError("gap and window stride must be positive")
        if not 1 <= self.minimum_positive_votes <= self.vote_window:
            raise ValueError("minimum_positive_votes must be in the vote window")
        if self.maximum_verification_seconds <= 0.0:
            raise ValueError("maximum_verification_seconds must be positive")
        if (
            self.track_grace_seconds < 0.0
            or self.alarm_cooldown_seconds < 0.0
            or self.global_alarm_cooldown_seconds < 0.0
        ):
            raise ValueError("track grace and alarm cooldown must be non-negative")
        if not 0.0 <= self.reassociation_iou_threshold <= 1.0:
            raise ValueError("reassociation_iou_threshold must be between 0 and 1")
        if self.reassociation_center_distance < 0.0:
            raise ValueError("reassociation_center_distance must be non-negative")
        if set(self.thresholds) != {"fire", "smoke"}:
            raise ValueError("thresholds must contain fire and smoke")
        if any(not 0.0 <= value <= 1.0 for value in self.thresholds.values()):
            raise ValueError("verification thresholds must be between 0 and 1")


def load_verifier_config(path: str | Path) -> VerifierRuntimeConfig:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load verifier config") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    clip = raw.get("clip", {})
    frequency = raw.get("frequency", {})
    voting = raw.get("voting", {})
    return VerifierRuntimeConfig(
        clip_seconds=float(clip.get("seconds", 3.2)),
        buffer_fps=float(clip.get("buffer_fps", 10.0)),
        clip_frames=int(clip.get("frames", 32)),
        rgb_frames=int(clip.get("rgb_frames", 16)),
        image_size=int(clip.get("image_size", 224)),
        roi_scale=float(clip.get("roi_scale", 1.5)),
        minimum_frequency_fps=float(frequency.get("minimum_fps", 8.0)),
        minimum_frequency_frames=int(frequency.get("minimum_frames", 26)),
        maximum_frame_gap=float(frequency.get("maximum_frame_gap", 0.25)),
        verification_fps=float(raw.get("verification_fps", 10.0)),
        window_stride_seconds=float(voting.get("stride_seconds", 0.5)),
        vote_window=int(voting.get("window", 3)),
        minimum_positive_votes=int(voting.get("minimum_positive", 2)),
        maximum_verification_seconds=float(voting.get("maximum_seconds", 10.0)),
        track_grace_seconds=float(voting.get("track_grace_seconds", 1.25)),
        reassociation_iou_threshold=float(voting.get("reassociation_iou", 0.05)),
        reassociation_center_distance=float(
            voting.get("reassociation_center_distance", 1.0)
        ),
        alarm_cooldown_seconds=float(voting.get("alarm_cooldown_seconds", 10.0)),
        global_alarm_cooldown_seconds=float(
            voting.get("global_alarm_cooldown_seconds", 10.0)
        ),
        thresholds={
            "fire": float(raw.get("thresholds", {}).get("fire", 0.5)),
            "smoke": float(raw.get("thresholds", {}).get("smoke", 0.5)),
        },
    )
