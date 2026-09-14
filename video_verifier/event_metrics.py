from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from .data import ManifestRecord, load_request


def summarize_event_predictions(
    records: list[ManifestRecord],
    manifest_path: str | Path,
    labels: list[int],
    probabilities: list[float],
    thresholds: dict[str, float],
    vote_window: int = 3,
    minimum_positive: int = 2,
) -> dict[str, Any]:
    root = Path(manifest_path).parent
    grouped: dict[str, list[tuple[float, int, float, ManifestRecord]]] = defaultdict(list)
    for record, label, probability in zip(records, labels, probabilities):
        sample_path = Path(record.sample)
        if not sample_path.is_absolute():
            sample_path = root / sample_path
        request = load_request(sample_path)
        grouped[request.event_id].append(
            (request.timestamps[-1], label, probability, record)
        )

    tp = fn = fp = tn = 0
    delays = []
    negative_alarm_events = 0
    source_durations: dict[str, float] = {}
    for windows in grouped.values():
        windows.sort(key=lambda item: item[0])
        event_labels = {item[1] for item in windows}
        if len(event_labels) != 1:
            raise ValueError("all windows of one event must share the same label")
        label = next(iter(event_labels))
        first_timestamp = windows[0][0]
        votes: deque[bool] = deque(maxlen=vote_window)
        alarm_timestamp = None
        for timestamp, _, probability, record in windows:
            votes.append(probability >= thresholds[record.candidate_class])
            if len(votes) == vote_window and sum(votes) >= minimum_positive:
                alarm_timestamp = timestamp
                break
        alarm = alarm_timestamp is not None
        tp += int(label == 1 and alarm)
        fn += int(label == 1 and not alarm)
        fp += int(label == 0 and alarm)
        tn += int(label == 0 and not alarm)
        negative_alarm_events += int(label == 0 and alarm)
        if label == 1 and alarm_timestamp is not None:
            delays.append(alarm_timestamp - first_timestamp)
        for _, _, _, record in windows:
            if record.source_duration_seconds is not None:
                source_durations[record.source_id] = max(
                    source_durations.get(record.source_id, 0.0),
                    record.source_duration_seconds,
                )

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    total_hours = sum(source_durations.values()) / 3600.0
    return {
        "events": len(grouped),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "false_positive_rate": ratio(fp, fp + tn),
        "false_alarms_per_hour": (
            negative_alarm_events / total_hours if total_hours > 0.0 else None
        ),
        "mean_alarm_delay_seconds": (
            sum(delays) / len(delays) if delays else None
        ),
        "duplicate_alarm_rate": 0.0 if grouped else None,
    }



def summarize_source_proxy_predictions(
    records: list[ManifestRecord],
    manifest_path: str | Path,
    labels: list[int],
    probabilities: list[float],
    thresholds: dict[str, float],
    vote_window: int = 3,
    minimum_positive: int = 2,
) -> dict[str, Any]:
    """Aggregate track alarms by source when only video-level labels exist."""

    root = Path(manifest_path).parent
    events: dict[str, list[tuple[float, int, bool, ManifestRecord]]] = defaultdict(list)
    for record, label, probability in zip(records, labels, probabilities):
        sample_path = Path(record.sample)
        if not sample_path.is_absolute():
            sample_path = root / sample_path
        request = load_request(sample_path)
        events[request.event_id].append(
            (
                request.timestamps[-1],
                label,
                probability >= thresholds[record.candidate_class],
                record,
            )
        )

    sources: dict[str, list[tuple[int, float | None, float, float | None]]] = defaultdict(list)
    for windows in events.values():
        windows.sort(key=lambda item: item[0])
        event_labels = {item[1] for item in windows}
        if len(event_labels) != 1:
            raise ValueError("all windows of one event must share the same label")
        votes: deque[bool] = deque(maxlen=vote_window)
        alarm_timestamp = None
        for timestamp, _, positive, _ in windows:
            votes.append(positive)
            if len(votes) == vote_window and sum(votes) >= minimum_positive:
                alarm_timestamp = timestamp
                break
        record = windows[0][3]
        sources[record.source_id].append(
            (
                windows[0][1],
                alarm_timestamp,
                windows[0][0],
                record.source_duration_seconds,
            )
        )

    tp = fn = fp = tn = 0
    delays = []
    total_negative_seconds = 0.0
    for source_events in sources.values():
        source_labels = {item[0] for item in source_events}
        if len(source_labels) != 1:
            raise ValueError("all events of one source must share the same label")
        label = next(iter(source_labels))
        alarm_times = [item[1] for item in source_events if item[1] is not None]
        alarm = bool(alarm_times)
        tp += int(label == 1 and alarm)
        fn += int(label == 1 and not alarm)
        fp += int(label == 0 and alarm)
        tn += int(label == 0 and not alarm)
        if label == 1 and alarm_times:
            first_window = min(item[2] for item in source_events)
            delays.append(min(alarm_times) - first_window)
        if label == 0:
            durations = [item[3] for item in source_events if item[3] is not None]
            if durations:
                total_negative_seconds += max(durations)

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    negative_hours = total_negative_seconds / 3600.0
    return {
        "sources": len(sources),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "false_positive_rate": ratio(fp, fp + tn),
        "false_alarms_per_hour": fp / negative_hours if negative_hours > 0.0 else None,
        "mean_alarm_delay_seconds": sum(delays) / len(delays) if delays else None,
    }
