from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from video_io import probe_video


SUPPORTED_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".webm"}


@dataclass(frozen=True, slots=True)
class SourceSpec:
    path: Path
    relative_path: str
    target_class: str
    label: int
    duration_seconds: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete YOLO + temporal gate + verifier system on a video dataset."
    )
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--model", default="yolo26s-p2-1024-recall/weights/best.pt")
    parser.add_argument("--weights", default="weights/mvit_v2_s_fire_smoke.pt")
    parser.add_argument("--config", default="config/temporal.yaml")
    parser.add_argument("--verifier-config", default="config/verifier.yaml")
    parser.add_argument("--temporal", choices=("tpt", "avt"), default="avt")
    parser.add_argument("--inference-fps", type=float, default=5.0)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--verifier-device", default="cuda:0")
    parser.add_argument(
        "--output",
        default="evaluation_results/system_dataset_avt.json",
    )
    parser.add_argument("--events-dir")
    parser.add_argument("--transcode-cache")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Optional substring filter; may be supplied more than once.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _source_label(path: Path) -> tuple[str, int]:
    lowered = [part.lower() for part in path.parts]
    target_class = "fire" if any("fire_videos" in part for part in lowered) else None
    if any("smoke_videos" in part for part in lowered):
        target_class = "smoke"
    if target_class is None:
        raise ValueError(f"cannot infer fire/smoke dataset from path: {path}")
    if "pos" in lowered:
        label = 1
    elif "neg" in lowered:
        label = 0
    else:
        raise ValueError(f"cannot infer pos/neg label from path: {path}")
    return target_class, label


def discover_sources(root: str | Path, include: Iterable[str] = ()) -> list[SourceSpec]:
    dataset_root = Path(root)
    filters = tuple(include)
    sources = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        relative = str(path.relative_to(dataset_root))
        if filters and not any(pattern in relative for pattern in filters):
            continue
        target_class, label = _source_label(path)
        probe = probe_video(path)
        sources.append(
            SourceSpec(
                path=path,
                relative_path=relative,
                target_class=target_class,
                label=label,
                duration_seconds=probe.duration_seconds if probe else None,
            )
        )
    return sources


def load_events(path: Path) -> list[dict[str, Any]]:
    events = []
    if not path.is_file():
        return events
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid event JSON at {path}:{line_number}") from exc
    return events


def summarize_video(spec: SourceSpec, events: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    gate_events = [item for item in events if "event_key" in item]
    verification = [item for item in events if item.get("message_type") == "verification"]
    alarms = [item for item in events if item.get("message_type") == "alarm"]
    suppressed = [
        item for item in events if item.get("message_type") == "verification_suppressed"
    ]
    errors = [item for item in events if item.get("message_type") == "verification_error"]
    target_alarms = [
        item for item in alarms if item.get("candidate_class") == spec.target_class
    ]
    off_target_alarms = [
        item for item in alarms if item.get("candidate_class") != spec.target_class
    ]
    first_alarm = min(
        (float(item["timestamp"]) for item in target_alarms if item.get("timestamp") is not None),
        default=None,
    )
    inference_times = [
        float(item["inference_ms"])
        for item in verification
        if item.get("inference_ms") is not None
    ]
    frequency_valid = sum(
        bool(item.get("quality", {}).get("frequency_valid")) for item in verification
    )
    return {
        **asdict(spec),
        "path": str(spec.path),
        "status": "completed",
        "elapsed_seconds": elapsed,
        "gate_events": len(gate_events),
        "verification_windows": len(verification),
        "positive_windows": sum(bool(item.get("clip_positive")) for item in verification),
        "frequency_valid_windows": frequency_valid,
        "alarms": len(alarms),
        "target_alarms": len(target_alarms),
        "off_target_alarms": len(off_target_alarms),
        "suppressed_duplicates": len(suppressed),
        "verification_errors": len(errors),
        "target_alarm": bool(target_alarms),
        "any_alarm": bool(alarms),
        "first_target_alarm_seconds": first_alarm,
        "mean_verifier_inference_ms": (
            sum(inference_times) / len(inference_times) if inference_times else None
        ),
    }


def binary_metrics(videos: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in videos if item.get("status") == "completed"]
    tp = sum(item["label"] == 1 and item["target_alarm"] for item in completed)
    fn = sum(item["label"] == 1 and not item["target_alarm"] for item in completed)
    fp = sum(item["label"] == 0 and item["target_alarm"] for item in completed)
    tn = sum(item["label"] == 0 and not item["target_alarm"] for item in completed)

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    negative_seconds = sum(
        float(item["duration_seconds"] or 0.0)
        for item in completed
        if item["label"] == 0
    )
    negative_alarm_events = sum(
        int(item["target_alarms"]) for item in completed if item["label"] == 0
    )
    delays = [
        float(item["first_target_alarm_seconds"])
        for item in completed
        if item["label"] == 1 and item["first_target_alarm_seconds"] is not None
    ]
    return {
        "videos": len(completed),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "false_positive_rate": ratio(fp, fp + tn),
        "accuracy": ratio(tp + tn, len(completed)),
        "false_alarm_events_per_negative_hour": (
            negative_alarm_events / (negative_seconds / 3600.0)
            if negative_seconds > 0.0
            else None
        ),
        "mean_alarm_from_video_start_seconds": (
            sum(delays) / len(delays) if delays else None
        ),
    }


def aggregate(videos: list[dict[str, Any]], settings: dict[str, Any]) -> dict[str, Any]:
    completed = [item for item in videos if item.get("status") == "completed"]
    failed = [item for item in videos if item.get("status") != "completed"]
    inference_times = [
        float(item["mean_verifier_inference_ms"])
        for item in completed
        if item.get("mean_verifier_inference_ms") is not None
    ]
    return {
        "settings": settings,
        "videos_discovered": len(videos),
        "videos_completed": len(completed),
        "videos_failed": len(failed),
        "metrics": {
            "overall": binary_metrics(completed),
            "fire": binary_metrics([item for item in completed if item["target_class"] == "fire"]),
            "smoke": binary_metrics([item for item in completed if item["target_class"] == "smoke"]),
        },
        "runtime_counts": {
            "gate_events": sum(int(item["gate_events"]) for item in completed),
            "verification_windows": sum(int(item["verification_windows"]) for item in completed),
            "positive_windows": sum(int(item["positive_windows"]) for item in completed),
            "frequency_valid_windows": sum(
                int(item["frequency_valid_windows"]) for item in completed
            ),
            "alarms": sum(int(item["alarms"]) for item in completed),
            "off_target_alarms": sum(int(item["off_target_alarms"]) for item in completed),
            "suppressed_duplicates": sum(
                int(item["suppressed_duplicates"]) for item in completed
            ),
            "verification_errors": sum(
                int(item["verification_errors"]) for item in completed
            ),
            "mean_verifier_inference_ms_per_video": (
                sum(inference_times) / len(inference_times) if inference_times else None
            ),
        },
        "videos": videos,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.inference_fps <= 0.0 or args.timeout_seconds <= 0.0:
        raise ValueError("inference FPS and timeout must be positive")
    output = Path(args.output)
    events_dir = Path(args.events_dir) if args.events_dir else output.with_suffix("").with_name(output.stem + "_events")
    transcode_cache = (
        Path(args.transcode_cache)
        if args.transcode_cache
        else output.parent / "transcoded_cache"
    )
    sources = discover_sources(args.dataset, args.include)
    if args.max_videos is not None:
        if args.max_videos <= 0:
            raise ValueError("--max-videos must be positive")
        sources = sources[: args.max_videos]
    if not sources:
        raise ValueError("no labeled videos found under the dataset path")

    settings = {
        "dataset": str(args.dataset),
        "model": str(args.model),
        "weights": str(args.weights),
        "temporal": args.temporal,
        "inference_fps": args.inference_fps,
        "imgsz": args.imgsz,
        "device": args.device,
        "verifier_device": args.verifier_device,
    }
    previous: dict[str, dict[str, Any]] = {}
    if args.resume and output.is_file():
        old = json.loads(output.read_text(encoding="utf-8"))
        if old.get("settings") != settings:
            raise ValueError(
                "--resume report settings do not match the requested model, gate, or devices"
            )
        previous = {
            item["relative_path"]: item
            for item in old.get("videos", [])
            if item.get("status") == "completed"
        }

    videos: list[dict[str, Any]] = []
    runner = Path(__file__).with_name("run_temporal.py")
    for index, spec in enumerate(sources, start=1):
        cached = previous.get(spec.relative_path)
        if cached is not None:
            videos.append(cached)
            print(f"[{index}/{len(sources)}] resume {spec.relative_path}", flush=True)
            continue

        safe_name = spec.relative_path.replace("/", "__").replace("\\", "__")
        event_path = events_dir / f"{safe_name}.jsonl"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        event_path.unlink(missing_ok=True)
        command = [
            sys.executable,
            str(runner),
            "--source",
            str(spec.path),
            "--model",
            args.model,
            "--config",
            args.config,
            "--temporal",
            args.temporal,
            "--camera-id",
            safe_name,
            "--inference-fps",
            str(args.inference_fps),
            "--imgsz",
            str(args.imgsz),
            "--device",
            args.device,
            "--verifier-config",
            args.verifier_config,
            "--verifier-weights",
            args.weights,
            "--verifier-device",
            args.verifier_device,
            "--events",
            str(event_path),
            "--transcode-cache",
            str(transcode_cache),
        ]
        print(f"[{index}/{len(sources)}] run {spec.relative_path}", flush=True)
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=args.timeout_seconds,
            )
            elapsed = time.perf_counter() - started
            if completed.returncode != 0:
                result = {
                    **asdict(spec),
                    "path": str(spec.path),
                    "status": "failed",
                    "return_code": completed.returncode,
                    "elapsed_seconds": elapsed,
                    "error": completed.stderr.strip()[-4000:],
                }
            else:
                result = summarize_video(spec, load_events(event_path), elapsed)
        except subprocess.TimeoutExpired as exc:
            result = {
                **asdict(spec),
                "path": str(spec.path),
                "status": "timeout",
                "elapsed_seconds": time.perf_counter() - started,
                "error": str(exc),
            }
        videos.append(result)
        write_report(output, aggregate(videos, settings))

    payload = aggregate(videos, settings)
    write_report(output, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "videos"}, ensure_ascii=False, indent=2))
    return 1 if payload["videos_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

