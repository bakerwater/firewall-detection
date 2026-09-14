from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from temporal_gates import Track, TriggerEvent
from temporal_gates.tracker import bbox_iou, normalized_center_distance

from .config import VerifierRuntimeConfig
from .types import AlarmDecision, BBox, VerificationResult


@dataclass(slots=True)
class VerificationSession:
    event_id: str
    camera_id: str
    track_id: int
    candidate_class: str
    started_at: float
    next_window_at: float
    probabilities: deque[float]
    positives: deque[bool]
    last_bbox: BBox
    last_track: Track | None = None
    last_track_seen_at: float | None = None


@dataclass(frozen=True, slots=True)
class ConfirmedRegion:
    camera_id: str
    candidate_class: str
    bbox: BBox
    timestamp: float


class VerificationSessionManager:
    """Schedules verification while surviving short misses and track ID switches."""

    def __init__(self, config: VerifierRuntimeConfig) -> None:
        self.config = config
        self._sessions: dict[tuple[str, int], VerificationSession] = {}
        self._confirmed: list[ConfirmedRegion] = []

    @property
    def has_active(self) -> bool:
        return bool(self._sessions)

    def start(
        self,
        event: TriggerEvent,
        track: Track | None = None,
    ) -> VerificationSession | None:
        self._purge_confirmed(event.timestamp)
        if any(
            region.camera_id == event.camera_id
            and region.candidate_class == event.class_name
            and (
                event.timestamp - region.timestamp
                <= self.config.global_alarm_cooldown_seconds
                or self._same_region(region.bbox, event.bbox)
            )
            for region in self._confirmed
        ):
            return None

        key = (event.camera_id, event.track_id)
        existing = self._sessions.get(key)
        if existing is not None:
            existing.last_bbox = event.bbox
            self._update_track(existing, track)
            return existing

        stitched = self._find_stitch_candidate(event)
        if stitched is not None:
            old_key = (stitched.camera_id, stitched.track_id)
            del self._sessions[old_key]
            stitched.track_id = event.track_id
            stitched.last_bbox = event.bbox
            self._update_track(stitched, track)
            self._sessions[key] = stitched
            return stitched

        session = VerificationSession(
            event_id=event.event_key,
            camera_id=event.camera_id,
            track_id=event.track_id,
            candidate_class=event.class_name,
            started_at=event.timestamp,
            next_window_at=event.timestamp,
            probabilities=deque(maxlen=self.config.vote_window),
            positives=deque(maxlen=self.config.vote_window),
            last_bbox=event.bbox,
            last_track=track,
            last_track_seen_at=track.last_seen if track is not None else event.timestamp,
        )
        self._sessions[key] = session
        return session

    def due(
        self,
        timestamp: float,
        active_tracks: Iterable[Track | int],
        camera_id: str,
    ) -> tuple[VerificationSession, ...]:
        active_by_id: dict[int, Track] = {}
        active_ids: set[int] = set()
        for item in active_tracks:
            if isinstance(item, int):
                active_ids.add(item)
            else:
                active_ids.add(item.track_id)
                active_by_id[item.track_id] = item

        due_sessions = []
        for key, session in list(self._sessions.items()):
            if session.camera_id != camera_id:
                continue
            track = active_by_id.get(session.track_id)
            if track is not None:
                self._update_track(session, track)
            last_seen = session.last_track_seen_at or session.started_at
            expired = timestamp - session.started_at > self.config.maximum_verification_seconds
            missing_too_long = (
                session.track_id not in active_ids
                and timestamp - last_seen > self.config.track_grace_seconds
            )
            stale_too_long = timestamp - last_seen > self.config.track_grace_seconds
            if expired or missing_too_long or stale_too_long:
                del self._sessions[key]
                continue
            if timestamp + 1e-9 < session.next_window_at:
                continue
            while session.next_window_at <= timestamp + 1e-9:
                session.next_window_at += self.config.window_stride_seconds
            due_sessions.append(session)
        return tuple(due_sessions)

    def record(
        self,
        result: VerificationResult,
        timestamp: float,
    ) -> AlarmDecision | None:
        key = (result.camera_id, result.track_id)
        session = self._sessions.get(key)
        if session is None:
            return None
        session.probabilities.append(result.p_real)
        session.positives.append(result.clip_positive)
        if len(session.positives) < self.config.vote_window:
            return None
        positive_votes = sum(session.positives)
        if positive_votes < self.config.minimum_positive_votes:
            return None
        del self._sessions[key]
        self._confirmed.append(
            ConfirmedRegion(
                camera_id=session.camera_id,
                candidate_class=session.candidate_class,
                bbox=session.last_bbox,
                timestamp=timestamp,
            )
        )
        for other_key, other in list(self._sessions.items()):
            if (
                other.camera_id == session.camera_id
                and other.candidate_class == session.candidate_class
                and self._same_region(session.last_bbox, other.last_bbox)
            ):
                del self._sessions[other_key]
        return AlarmDecision(
            event_id=session.event_id,
            camera_id=result.camera_id,
            track_id=result.track_id,
            candidate_class=result.candidate_class,
            alarm=True,
            positive_votes=positive_votes,
            window_size=self.config.vote_window,
            mean_probability=sum(session.probabilities) / len(session.probabilities),
            timestamp=timestamp,
        )

    def reset(self, camera_id: str | None = None) -> None:
        if camera_id is None:
            self._sessions.clear()
            self._confirmed.clear()
            return
        self._sessions = {
            key: value
            for key, value in self._sessions.items()
            if value.camera_id != camera_id
        }
        self._confirmed = [
            value for value in self._confirmed if value.camera_id != camera_id
        ]

    def _find_stitch_candidate(
        self, event: TriggerEvent
    ) -> VerificationSession | None:
        candidates = []
        for session in self._sessions.values():
            track = session.last_track
            last_seen = session.last_track_seen_at or session.started_at
            if (
                session.camera_id != event.camera_id
                or session.candidate_class != event.class_name
                or track is None
                or track.last_observation_detected
                or event.timestamp - last_seen > self.config.track_grace_seconds
                or not self._same_region(track.bbox, event.bbox)
            ):
                continue
            candidates.append((normalized_center_distance(track.bbox, event.bbox), session))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def _same_region(self, first: BBox, second: BBox) -> bool:
        return (
            bbox_iou(first, second) >= self.config.reassociation_iou_threshold
            or normalized_center_distance(first, second)
            <= self.config.reassociation_center_distance
        )

    @staticmethod
    def _update_track(session: VerificationSession, track: Track | None) -> None:
        if track is None:
            return
        session.last_track = track
        session.last_bbox = track.bbox
        if track.last_observation_detected:
            session.last_track_seen_at = track.last_seen

    def _purge_confirmed(self, timestamp: float) -> None:
        self._confirmed = [
            item
            for item in self._confirmed
            if timestamp - item.timestamp <= self.config.alarm_cooldown_seconds
        ]
