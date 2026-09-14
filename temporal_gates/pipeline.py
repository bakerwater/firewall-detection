from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .avt import AVTClassConfig, AreaVariationGate
from .tpt import TPTClassConfig, TemporalPersistenceGate
from .tracker import MultiCameraTracker, TrackerConfig
from .types import Detection, FrameSize, ProcessResult


@dataclass(frozen=True, slots=True)
class IgnoreRegion:
    bbox: tuple[float, float, float, float]
    camera_id: str | None = None
    class_name: str | None = None
    normalized: bool = True

    def contains_center(self, detection: Detection) -> bool:
        if self.camera_id is not None and self.camera_id != detection.camera_id:
            return False
        if self.class_name is not None and self.class_name != detection.class_name:
            return False
        x1, y1, x2, y2 = self.bbox
        if self.normalized:
            if detection.frame_size is None:
                raise ValueError("normalized ignore regions require detection.frame_size")
            width, height = detection.frame_size
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        center_x = (detection.bbox[0] + detection.bbox[2]) / 2.0
        center_y = (detection.bbox[1] + detection.bbox[3]) / 2.0
        return x1 <= center_x <= x2 and y1 <= center_y <= y2


@dataclass(frozen=True, slots=True)
class PostprocessorConfig:
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    class_gates: dict[str, TPTClassConfig] = field(default_factory=dict)
    candidate_confidence: dict[str, float] = field(
        default_factory=lambda: {"fire": 0.25, "smoke": 0.25}
    )
    default_candidate_confidence: float = 0.25
    ignore_regions: tuple[IgnoreRegion, ...] = ()
    temporal_mode: str = "tpt"
    avt_class_gates: dict[str, AVTClassConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.temporal_mode not in {"tpt", "avt"}:
            raise ValueError("temporal_mode must be 'tpt' or 'avt'")
        thresholds = [self.default_candidate_confidence, *self.candidate_confidence.values()]
        if any(not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("candidate confidence thresholds must be between 0 and 1")


class FireSmokePostprocessor:
    def __init__(self, config: PostprocessorConfig | None = None) -> None:
        self.config = config or PostprocessorConfig()
        self.tracker = MultiCameraTracker(self.config.tracker)
        if self.config.temporal_mode == "avt":
            self.gate = AreaVariationGate(self.config.avt_class_gates)
        else:
            self.gate = TemporalPersistenceGate(self.config.class_gates)

    def process(
        self,
        camera_id: str,
        timestamp: float,
        detections: Sequence[Detection],
    ) -> ProcessResult:
        accepted: list[Detection] = []
        ignored_count = 0
        for detection in detections:
            if detection.camera_id != camera_id or detection.timestamp != timestamp:
                raise ValueError("detection camera_id/timestamp must match process arguments")
            threshold = self.config.candidate_confidence.get(
                detection.class_name, self.config.default_candidate_confidence
            )
            is_ignored = any(
                region.contains_center(detection) for region in self.config.ignore_regions
            )
            if detection.confidence < threshold or is_ignored:
                ignored_count += 1
            else:
                accepted.append(detection)

        update = self.tracker.update(camera_id, accepted, timestamp)
        self.gate.forget(camera_id, update.expired_track_ids)
        decisions = []
        events = []
        for track in update.tracks:
            decision, event = self.gate.evaluate(track, timestamp)
            decisions.append(decision)
            if event is not None:
                events.append(event)

        return ProcessResult(
            camera_id=camera_id,
            timestamp=timestamp,
            tracks=update.tracks,
            decisions=tuple(decisions),
            events=tuple(events),
            accepted_detection_count=len(accepted),
            ignored_detection_count=ignored_count,
            reset_due_to_gap=update.reset_due_to_gap,
        )

    def process_ultralytics(
        self,
        result: Any,
        camera_id: str,
        timestamp: float,
    ) -> ProcessResult:
        detections = ultralytics_result_to_detections(result, camera_id, timestamp)
        return self.process(camera_id, timestamp, detections)

    def reset(self, camera_id: str | None = None) -> None:
        self.tracker.reset(camera_id)
        self.gate.reset(camera_id)


def ultralytics_result_to_detections(
    result: Any,
    camera_id: str,
    timestamp: float,
) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    names = result.names
    orig_shape = getattr(result, "orig_shape", None)
    frame_size: FrameSize | None = None
    if orig_shape is not None:
        frame_size = (int(orig_shape[1]), int(orig_shape[0]))

    xyxy_rows = boxes.xyxy.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    class_ids = boxes.cls.detach().cpu().tolist()
    detections = []
    for bbox, confidence, class_id_value in zip(xyxy_rows, confidences, class_ids):
        class_id = int(class_id_value)
        class_name = names[class_id] if isinstance(names, Mapping) else names[class_id]
        detections.append(
            Detection(
                camera_id=camera_id,
                class_name=str(class_name).strip().lower(),
                confidence=float(confidence),
                bbox=tuple(float(value) for value in bbox),
                timestamp=timestamp,
                frame_size=frame_size,
            )
        )
    return detections


def load_config(path: str | Path) -> PostprocessorConfig:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on deployment environment
        raise RuntimeError("PyYAML is required to load a YAML config") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    tracker = TrackerConfig(**raw.get("tracker", {}))
    temporal_mode = str(raw.get("temporal_mode", "tpt")).strip().lower()
    class_gates = {
        str(class_name).lower(): TPTClassConfig(**values)
        for class_name, values in raw.get("classes", {}).items()
    }
    avt_class_gates = {
        str(class_name).lower(): AVTClassConfig(**values)
        for class_name, values in raw.get("avt_classes", {}).items()
    }
    candidate_confidence = {
        str(name).lower(): float(value)
        for name, value in raw.get("candidate_confidence", {}).items()
    }
    regions = tuple(
        IgnoreRegion(
            bbox=tuple(float(value) for value in item["bbox"]),
            camera_id=item.get("camera_id"),
            class_name=item.get("class_name"),
            normalized=bool(item.get("normalized", True)),
        )
        for item in raw.get("ignore_regions", [])
    )
    return PostprocessorConfig(
        tracker=tracker,
        class_gates=class_gates,
        temporal_mode=temporal_mode,
        candidate_confidence=candidate_confidence
        or {"fire": 0.25, "smoke": 0.25},
        avt_class_gates=avt_class_gates,
        default_candidate_confidence=float(
            raw.get("default_candidate_confidence", 0.25)
        ),
        ignore_regions=regions,
    )
