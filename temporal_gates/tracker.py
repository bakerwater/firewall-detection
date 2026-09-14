from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from .types import BBox, Detection, Track, TrackObservation


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    iou_threshold: float = 0.15
    center_distance_threshold: float = 1.5
    max_misses: int = 2
    max_missed_seconds: float = 1.0
    max_history: int = 64
    max_timestamp_gap: float = 5.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if self.center_distance_threshold < 0.0:
            raise ValueError("center_distance_threshold must be non-negative")
        if self.max_misses < 0 or self.max_missed_seconds < 0.0:
            raise ValueError("miss limits must be non-negative")
        if self.max_history <= 0 or self.max_timestamp_gap <= 0.0:
            raise ValueError("history and timestamp gap must be positive")


@dataclass(frozen=True, slots=True)
class TrackerUpdate:
    tracks: tuple[Track, ...]
    expired_track_ids: tuple[int, ...]
    reset_due_to_gap: bool


def bbox_iou(first: BBox, second: BBox) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def normalized_center_distance(first: BBox, second: BBox) -> float:
    first_center = ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
    second_center = ((second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0)
    distance = hypot(first_center[0] - second_center[0], first_center[1] - second_center[1])
    first_diagonal = hypot(first[2] - first[0], first[3] - first[1])
    second_diagonal = hypot(second[2] - second[0], second[3] - second[1])
    scale = max((first_diagonal + second_diagonal) / 2.0, 1.0)
    return distance / scale


class MultiCameraTracker:
    """Class-aware greedy tracker isolated by camera."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._tracks: dict[str, dict[int, Track]] = {}
        self._last_timestamp: dict[str, float] = {}
        self._next_track_id = 1

    def update(
        self, camera_id: str, detections: list[Detection], timestamp: float
    ) -> TrackerUpdate:
        if any(d.camera_id != camera_id for d in detections):
            raise ValueError("all detections must belong to the updated camera")
        if any(d.timestamp != timestamp for d in detections):
            raise ValueError("all detection timestamps must equal the update timestamp")

        previous_timestamp = self._last_timestamp.get(camera_id)
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise ValueError(
                f"timestamp moved backwards for {camera_id}: {timestamp} < {previous_timestamp}"
            )

        reset_due_to_gap = bool(
            previous_timestamp is not None
            and timestamp - previous_timestamp > self.config.max_timestamp_gap
        )
        expired: list[int] = []
        if reset_due_to_gap:
            expired.extend(self.reset(camera_id))

        camera_tracks = self._tracks.setdefault(camera_id, {})
        # A stale track must not be revived just because a new box appears nearby.
        for track_id, track in list(camera_tracks.items()):
            if timestamp - track.last_seen > self.config.max_missed_seconds:
                expired.append(track_id)
                del camera_tracks[track_id]

        pairs: list[tuple[float, int, int]] = []
        for track_id, track in camera_tracks.items():
            for detection_index, detection in enumerate(detections):
                if track.class_name != detection.class_name:
                    continue
                iou = bbox_iou(track.bbox, detection.bbox)
                center_distance = normalized_center_distance(track.bbox, detection.bbox)
                if iou >= self.config.iou_threshold:
                    score = 2.0 + iou
                elif center_distance <= self.config.center_distance_threshold:
                    score = 1.0 / (1.0 + center_distance)
                else:
                    continue
                pairs.append((score, track_id, detection_index))

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _, track_id, detection_index in sorted(pairs, reverse=True):
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            camera_tracks[track_id].record_detection(
                detections[detection_index], self.config.max_history
            )
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        for track_id, track in list(camera_tracks.items()):
            if track_id in matched_tracks:
                continue
            track.record_miss(timestamp, self.config.max_history)
            if track.consecutive_misses > self.config.max_misses:
                expired.append(track_id)
                del camera_tracks[track_id]

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            track = self._new_track(detection)
            camera_tracks[track.track_id] = track

        self._last_timestamp[camera_id] = timestamp
        return TrackerUpdate(
            tracks=tuple(sorted(camera_tracks.values(), key=lambda item: item.track_id)),
            expired_track_ids=tuple(expired),
            reset_due_to_gap=reset_due_to_gap,
        )

    def reset(self, camera_id: str | None = None) -> tuple[int, ...]:
        if camera_id is None:
            expired = tuple(
                track_id for tracks in self._tracks.values() for track_id in tracks
            )
            self._tracks.clear()
            self._last_timestamp.clear()
            return expired
        tracks = self._tracks.pop(camera_id, {})
        self._last_timestamp.pop(camera_id, None)
        return tuple(tracks)

    def _new_track(self, detection: Detection) -> Track:
        x1, y1, x2, y2 = detection.bbox
        track = Track(
            track_id=self._next_track_id,
            camera_id=detection.camera_id,
            class_name=detection.class_name,
            first_seen=detection.timestamp,
            last_seen=detection.timestamp,
            last_update=detection.timestamp,
            bbox=detection.bbox,
            confidence_history=[detection.confidence],
            bbox_history=[detection.bbox],
            center_history=[((x1 + x2) / 2.0, (y1 + y2) / 2.0)],
            area_history=[(x2 - x1) * (y2 - y1)],
        )
        track.observations.append(
            TrackObservation(
                timestamp=detection.timestamp,
                detected=True,
                bbox=detection.bbox,
                confidence=detection.confidence,
            )
        )
        self._next_track_id += 1
        return track
