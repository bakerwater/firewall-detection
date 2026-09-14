from __future__ import annotations

from dataclasses import dataclass

from .types import GateDecision, Track, TriggerEvent, TriggerReason


@dataclass(frozen=True, slots=True)
class TPTClassConfig:
    window_size: int
    min_hits: int
    min_track_duration: float
    min_confidence: float = 0.25
    high_confidence: float = 0.80
    max_misses: int = 1

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 1 <= self.min_hits <= self.window_size:
            raise ValueError("min_hits must be in [1, window_size]")
        if self.min_track_duration < 0.0 or self.max_misses < 0:
            raise ValueError("duration and max_misses must be non-negative")
        if not 0.0 <= self.min_confidence <= self.high_confidence <= 1.0:
            raise ValueError(
                "confidence values must satisfy 0 <= min <= high <= 1"
            )


DEFAULT_FIRE_CONFIG = TPTClassConfig(
    window_size=3,
    min_hits=2,
    min_track_duration=1.0,
    min_confidence=0.25,
    high_confidence=0.80,
    max_misses=1,
)
DEFAULT_SMOKE_CONFIG = TPTClassConfig(
    window_size=5,
    min_hits=3,
    min_track_duration=2.0,
    min_confidence=0.25,
    high_confidence=0.85,
    max_misses=2,
)
DEFAULT_OTHER_CONFIG = TPTClassConfig(
    window_size=4,
    min_hits=2,
    min_track_duration=1.0,
    min_confidence=0.25,
    high_confidence=0.85,
    max_misses=1,
)


class TemporalPersistenceGate:
    def __init__(
        self,
        class_configs: dict[str, TPTClassConfig] | None = None,
        default_config: TPTClassConfig = DEFAULT_OTHER_CONFIG,
    ) -> None:
        self.class_configs = {
            "fire": DEFAULT_FIRE_CONFIG,
            "smoke": DEFAULT_SMOKE_CONFIG,
            **(class_configs or {}),
        }
        self.default_config = default_config
        self._triggered: set[tuple[str, int]] = set()

    def evaluate(self, track: Track, timestamp: float) -> tuple[GateDecision, TriggerEvent | None]:
        config = self.class_configs.get(track.class_name, self.default_config)
        observations = track.observations[-config.window_size :]
        detected = [item for item in observations if item.detected]
        window_hits = len(detected)
        confidence_max = max(
            (item.confidence or 0.0 for item in detected), default=0.0
        )
        window_misses = len(observations) - window_hits
        key = (track.camera_id, track.track_id)
        already_triggered = key in self._triggered

        reason = "waiting"
        trigger_reason: TriggerReason | None = None
        if already_triggered:
            reason = "already_triggered"
        elif not track.last_observation_detected:
            reason = "latest_sample_missed"
        elif track.last_confidence >= config.high_confidence:
            reason = TriggerReason.HIGH_CONFIDENCE.value
            trigger_reason = TriggerReason.HIGH_CONFIDENCE
        elif confidence_max < config.min_confidence:
            reason = "confidence_too_low"
        elif window_misses > config.max_misses:
            reason = "too_many_misses"
        elif window_hits < config.min_hits:
            reason = "insufficient_hits"
        elif track.duration < config.min_track_duration:
            reason = "duration_too_short"
        else:
            reason = TriggerReason.TEMPORAL_PERSISTENCE.value
            trigger_reason = TriggerReason.TEMPORAL_PERSISTENCE

        passed = trigger_reason is not None
        decision = GateDecision(
            camera_id=track.camera_id,
            track_id=track.track_id,
            class_name=track.class_name,
            passed=passed,
            already_triggered=already_triggered,
            reason=reason,
            window_hits=window_hits,
            window_samples=len(observations),
            duration=track.duration,
            max_confidence=confidence_max,
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
            window_hits=window_hits,
            duration=track.duration,
        )

    def forget(self, camera_id: str, track_ids: tuple[int, ...]) -> None:
        for track_id in track_ids:
            self._triggered.discard((camera_id, track_id))

    def reset(self, camera_id: str | None = None) -> None:
        if camera_id is None:
            self._triggered.clear()
            return
        self._triggered = {
            key for key in self._triggered if key[0] != camera_id
        }
