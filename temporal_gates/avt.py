from __future__ import annotations

from dataclasses import dataclass
from statistics import pstdev

from .types import GateDecision, Track, TriggerEvent, TriggerReason


@dataclass(frozen=True, slots=True)
class AVTClassConfig:
    """Area Variation Technique parameters for one detector class."""

    window_size: int = 20
    area_threshold: float = 0.05
    min_samples: int = 2
    min_track_duration: float = 0.0

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
        if not 2 <= self.min_samples <= self.window_size:
            raise ValueError("min_samples must be in [2, window_size]")
        if self.area_threshold < 0.0:
            raise ValueError("area_threshold must be non-negative")
        if self.min_track_duration < 0.0:
            raise ValueError("min_track_duration must be non-negative")


DEFAULT_FIRE_AVT_CONFIG = AVTClassConfig()
DEFAULT_SMOKE_AVT_CONFIG = AVTClassConfig()


class AreaVariationGate:
    """Trigger when a tracked detection box exhibits sufficient area variation."""

    def __init__(
        self,
        class_configs: dict[str, AVTClassConfig] | None = None,
        default_config: AVTClassConfig = AVTClassConfig(),
    ) -> None:
        self.class_configs = {
            "fire": DEFAULT_FIRE_AVT_CONFIG,
            "smoke": DEFAULT_SMOKE_AVT_CONFIG,
            **(class_configs or {}),
        }
        self.default_config = default_config
        self._triggered: set[tuple[str, int]] = set()

    def evaluate(
        self, track: Track, timestamp: float
    ) -> tuple[GateDecision, TriggerEvent | None]:
        config = self.class_configs.get(track.class_name, self.default_config)
        areas = track.area_history[-config.window_size :]
        mean_area = sum(areas) / len(areas) if areas else 0.0
        area_variation = (
            pstdev(areas) / mean_area
            if len(areas) >= 2 and mean_area > 0.0
            else 0.0
        )
        key = (track.camera_id, track.track_id)
        already_triggered = key in self._triggered

        trigger_reason: TriggerReason | None = None
        if already_triggered:
            reason = "already_triggered"
        elif not track.last_observation_detected:
            reason = "latest_sample_missed"
        elif len(areas) < config.min_samples:
            reason = "insufficient_area_samples"
        elif track.duration < config.min_track_duration:
            reason = "duration_too_short"
        elif area_variation < config.area_threshold:
            reason = "area_variation_too_low"
        else:
            reason = TriggerReason.AREA_VARIATION.value
            trigger_reason = TriggerReason.AREA_VARIATION

        passed = trigger_reason is not None
        decision = GateDecision(
            camera_id=track.camera_id,
            track_id=track.track_id,
            class_name=track.class_name,
            passed=passed,
            already_triggered=already_triggered,
            reason=reason,
            window_hits=len(areas),
            window_samples=len(areas),
            duration=track.duration,
            max_confidence=max(track.confidence_history, default=0.0),
            area_variation=area_variation,
        )
        if not passed:
            return decision, None

        self._triggered.add(key)
        return decision, TriggerEvent(
            event_key=f"{track.camera_id}:{track.track_id}",
            camera_id=track.camera_id,
            track_id=track.track_id,
            class_name=track.class_name,
            timestamp=timestamp,
            bbox=track.bbox,
            confidence=track.last_confidence,
            reason=trigger_reason,
            window_hits=len(areas),
            duration=track.duration,
        )

    def forget(self, camera_id: str, track_ids: tuple[int, ...]) -> None:
        for track_id in track_ids:
            self._triggered.discard((camera_id, track_id))

    def reset(self, camera_id: str | None = None) -> None:
        if camera_id is None:
            self._triggered.clear()
            return
        self._triggered = {key for key in self._triggered if key[0] != camera_id}
