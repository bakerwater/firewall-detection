from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil
from statistics import median
from typing import Any, Iterable

from temporal_gates import Track

from .config import VerifierRuntimeConfig
from .types import BBox, ClipQuality, VerificationRequest


@dataclass(frozen=True, slots=True)
class BufferedFrame:
    timestamp: float
    frame: Any


class VideoRingBuffer:
    """Timestamp-aware frame buffer sampled independently from YOLO inference."""

    def __init__(self, config: VerifierRuntimeConfig) -> None:
        self.config = config
        capacity = ceil(config.clip_seconds * config.buffer_fps) + 8
        self._frames: deque[BufferedFrame] = deque(maxlen=capacity)
        self._last_input_timestamp: float | None = None
        self._next_sample_at: float | None = None

    def append(self, frame: Any, timestamp: float) -> bool:
        if self._last_input_timestamp is not None and timestamp < self._last_input_timestamp:
            self.reset()
        self._last_input_timestamp = timestamp
        if self._next_sample_at is not None and timestamp + 1e-9 < self._next_sample_at:
            return False
        self._frames.append(BufferedFrame(timestamp, frame.copy()))
        interval = 1.0 / self.config.buffer_fps
        if self._next_sample_at is None:
            self._next_sample_at = timestamp + interval
        else:
            while self._next_sample_at <= timestamp + 1e-9:
                self._next_sample_at += interval
        cutoff = timestamp - self.config.clip_seconds - interval
        while self._frames and self._frames[0].timestamp < cutoff:
            self._frames.popleft()
        return True

    def reset(self) -> None:
        self._frames.clear()
        self._last_input_timestamp = None
        self._next_sample_at = None

    @property
    def frames(self) -> tuple[BufferedFrame, ...]:
        return tuple(self._frames)


def _interpolate(
    values: list[tuple[float, tuple[float, ...]]], timestamp: float
) -> tuple[float, ...]:
    if timestamp <= values[0][0]:
        return values[0][1]
    if timestamp >= values[-1][0]:
        return values[-1][1]
    for (left_t, left), (right_t, right) in zip(values, values[1:]):
        if left_t <= timestamp <= right_t:
            ratio = (timestamp - left_t) / max(right_t - left_t, 1e-9)
            return tuple(a + ratio * (b - a) for a, b in zip(left, right))
    return values[-1][1]


def _square_roi(bboxes: Iterable[BBox], width: int, height: int, scale: float) -> BBox:
    boxes = tuple(bboxes)
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1, 2.0) * scale
    side = min(side, float(max(width, height)))
    return (
        center_x - side / 2.0,
        center_y - side / 2.0,
        center_x + side / 2.0,
        center_y + side / 2.0,
    )


def _crop_with_padding(frame: Any, roi: BBox, image_size: int) -> Any:
    import cv2
    import numpy as np

    x1, y1, x2, y2 = (int(round(value)) for value in roi)
    height, width = frame.shape[:2]
    left, top = max(0, -x1), max(0, -y1)
    right, bottom = max(0, x2 - width), max(0, y2 - height)
    if left or top or right or bottom:
        frame = cv2.copyMakeBorder(
            frame, top, bottom, left, right, cv2.BORDER_REFLECT_101
        )
        x1, x2 = x1 + left, x2 + left
        y1, y2 = y1 + top, y2 + top
    crop = frame[max(0, y1) : max(y1 + 1, y2), max(0, x1) : max(x1 + 1, x2)]
    if crop.size == 0:
        crop = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    else:
        crop = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _nearest_observation(track: Track, timestamp: float) -> bool:
    if not track.observations:
        return False
    return min(track.observations, key=lambda item: abs(item.timestamp - timestamp)).detected


class ClipBuilder:
    def __init__(self, config: VerifierRuntimeConfig) -> None:
        self.config = config

    def build(
        self,
        buffer: VideoRingBuffer,
        track: Track,
        event_id: str,
        end_timestamp: float,
    ) -> VerificationRequest:
        samples = [
            item
            for item in buffer.frames
            if end_timestamp - self.config.clip_seconds <= item.timestamp <= end_timestamp + 1e-9
        ]
        if not samples:
            raise RuntimeError("video buffer has no frames for verification")
        detected = [item for item in track.observations if item.detected and item.bbox]
        if not detected:
            raise RuntimeError("track has no detected observations")

        step = 1.0 / self.config.buffer_fps
        grid = tuple(
            end_timestamp - step * (self.config.clip_frames - 1 - index)
            for index in range(self.config.clip_frames)
        )
        tolerance = step * 0.6
        chosen: list[BufferedFrame] = []
        valid: list[bool] = []
        chosen_indexes: list[int] = []
        for timestamp in grid:
            index = min(
                range(len(samples)),
                key=lambda idx: abs(samples[idx].timestamp - timestamp),
            )
            chosen.append(samples[index])
            chosen_indexes.append(index)
            valid.append(abs(samples[index].timestamp - timestamp) <= tolerance)

        bbox_values = [(item.timestamp, tuple(item.bbox)) for item in detected if item.bbox]
        confidence_values = [
            (item.timestamp, (float(item.confidence or 0.0),)) for item in detected
        ]
        bboxes = tuple(_interpolate(bbox_values, timestamp) for timestamp in grid)
        confidences = tuple(
            max(0.0, min(1.0, _interpolate(confidence_values, timestamp)[0]))
            for timestamp in grid
        )
        observed = tuple(_nearest_observation(track, timestamp) for timestamp in grid)

        height, width = chosen[-1].frame.shape[:2]
        roi = _square_roi(bboxes, width, height, self.config.roi_scale)
        frames = tuple(
            _crop_with_padding(item.frame, roi, self.config.image_size) for item in chosen
        )

        unique_valid = {index for index, is_valid in zip(chosen_indexes, valid) if is_valid}
        source_times = [samples[index].timestamp for index in sorted(unique_valid)]
        gaps = [right - left for left, right in zip(source_times, source_times[1:])]
        effective_fps = 1.0 / median(gaps) if gaps and median(gaps) > 0.0 else 0.0
        max_gap = max(gaps, default=0.0)
        reasons = []
        if len(unique_valid) < self.config.minimum_frequency_frames:
            reasons.append("insufficient_actual_frames")
        if effective_fps < self.config.minimum_frequency_fps:
            reasons.append("source_fps_too_low")
        if max_gap > self.config.maximum_frame_gap:
            reasons.append("frame_gap_too_large")
        quality = ClipQuality(
            frequency_valid=not reasons,
            actual_frame_count=len(unique_valid),
            requested_frame_count=self.config.clip_frames,
            effective_fps=effective_fps,
            max_frame_gap=max_gap,
            reasons=tuple(reasons),
        )
        return VerificationRequest(
            event_id=event_id,
            camera_id=track.camera_id,
            track_id=track.track_id,
            candidate_class=track.class_name,
            frames=frames,
            timestamps=grid,
            bboxes=bboxes,
            confidences=confidences,
            observed=observed,
            frame_size=(width, height),
            quality=quality,
        )
