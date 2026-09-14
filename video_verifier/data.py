from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .features import preprocess_request
from .types import ClipQuality, VerificationRequest


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    sample: str
    source_id: str
    split: str
    candidate_class: str
    label: int | None
    license: str | None = None
    source_duration_seconds: float | None = None


def save_request(path: str | Path, request: VerificationRequest) -> None:
    import numpy as np

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frames=np.stack(request.frames).astype(np.uint8),
        timestamps=np.asarray(request.timestamps, dtype=np.float64),
        bboxes=np.asarray(request.bboxes, dtype=np.float32),
        confidences=np.asarray(request.confidences, dtype=np.float32),
        observed=np.asarray(request.observed, dtype=np.bool_),
        frame_size=np.asarray(request.frame_size, dtype=np.int32),
        event_id=np.asarray(request.event_id),
        camera_id=np.asarray(request.camera_id),
        track_id=np.asarray(request.track_id, dtype=np.int64),
        candidate_class=np.asarray(request.candidate_class),
        quality=np.asarray(json.dumps(asdict(request.quality), ensure_ascii=True)),
    )


def load_request(path: str | Path) -> VerificationRequest:
    import numpy as np

    with np.load(path, allow_pickle=False) as payload:
        quality_raw = json.loads(str(payload["quality"]))
        quality_raw["reasons"] = tuple(quality_raw.get("reasons", ()))
        return VerificationRequest(
            event_id=str(payload["event_id"]),
            camera_id=str(payload["camera_id"]),
            track_id=int(payload["track_id"]),
            candidate_class=str(payload["candidate_class"]),
            frames=tuple(payload["frames"]),
            timestamps=tuple(float(value) for value in payload["timestamps"]),
            bboxes=tuple(tuple(float(item) for item in row) for row in payload["bboxes"]),
            confidences=tuple(float(value) for value in payload["confidences"]),
            observed=tuple(bool(value) for value in payload["observed"]),
            frame_size=tuple(int(value) for value in payload["frame_size"]),
            quality=ClipQuality(**quality_raw),
        )


def append_manifest(path: str | Path, record: ManifestRecord) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = ManifestRecord(**json.loads(line))
            except Exception as exc:
                raise ValueError(f"invalid manifest line {line_number}: {exc}") from exc
            if record.split not in {"train", "val", "test", "unlabeled"}:
                raise ValueError(f"invalid split on line {line_number}: {record.split}")
            if record.label not in {None, 0, 1}:
                raise ValueError(f"invalid label on line {line_number}: {record.label}")
            if record.label is not None and not record.license:
                raise ValueError(f"labeled data requires license metadata on line {line_number}")
            records.append(record)
    return records


def validate_group_splits(records: Iterable[ManifestRecord]) -> None:
    source_splits: dict[str, set[str]] = {}
    for record in records:
        if record.split == "unlabeled":
            continue
        source_splits.setdefault(record.source_id, set()).add(record.split)
    leaked = {source: splits for source, splits in source_splits.items() if len(splits) > 1}
    if leaked:
        examples = ", ".join(f"{key}:{sorted(value)}" for key, value in list(leaked.items())[:5])
        raise ValueError(f"source videos cross dataset splits: {examples}")


class VerificationDataset:
    def __init__(
        self,
        manifest: str | Path,
        split: str,
        rgb_frames: int = 16,
        augment: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest)
        all_records = load_manifest(self.manifest_path)
        validate_group_splits(all_records)
        self.records = [
            record
            for record in all_records
            if record.split == split and record.label is not None
        ]
        if not self.records:
            raise ValueError(f"manifest has no labeled records for split={split}")
        self.rgb_frames = rgb_frames
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        record = self.records[index]
        sample_path = Path(record.sample)
        if not sample_path.is_absolute():
            sample_path = self.manifest_path.parent / sample_path
        request = load_request(sample_path)
        if request.candidate_class != record.candidate_class:
            raise ValueError(f"manifest/sample class mismatch: {sample_path}")
        output = preprocess_request(
            request, rgb_frames=self.rgb_frames, augment=self.augment
        )
        output["label"] = torch.tensor(float(record.label), dtype=torch.float32)
        output["source_id"] = record.source_id
        output["sample_path"] = str(sample_path)
        return output
