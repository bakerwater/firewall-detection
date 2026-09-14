from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from temporal_gates import FireSmokePostprocessor, PostprocessorConfig, load_config
from video_io import prepare_video_source

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO fire/smoke detection with selectable TPT/AVT gating."
    )
    parser.add_argument("--source", required=True, help="Video path, stream URL, or camera index")
    parser.add_argument(
        "--model",
        default="yolo26s-p2-1024-recall/weights/best.pt",
        help="Ultralytics model path",
    )
    parser.add_argument("--config", default="config/temporal.yaml")
    parser.add_argument("--temporal", choices=("tpt", "avt"))
    parser.add_argument("--camera-id", default="camera_001")
    parser.add_argument("--inference-fps", type=float, default=5.0)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--candidate-confidence-fire",
        type=float,
        help="Override the fire confidence threshold from the temporal config",
    )
    parser.add_argument(
        "--candidate-confidence-smoke",
        type=float,
        help="Override the smoke confidence threshold from the temporal config",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save", help="Optional annotated video output path")
    parser.add_argument("--events", help="Optional JSONL trigger event output path")
    parser.add_argument(
        "--transcode-cache",
        help="Cache directory for AVI codecs that are unsafe in the OpenCV backend",
    )
    parser.add_argument(
        "--force-transcode",
        action="store_true",
        help="Normalize a local video through FFmpeg before opening it",
    )
    parser.add_argument("--verifier-config", default="config/verifier.yaml")
    parser.add_argument("--verifier-weights", help="Enable second-stage verification")
    parser.add_argument("--verifier-device", help="Verifier device, e.g. cuda:0 or cpu")
    parser.add_argument("--verifier-threshold-fire", type=float)
    parser.add_argument("--verifier-threshold-smoke", type=float)
    parser.add_argument(
        "--candidate-dir",
        help="Export triggered clips for later manual labeling",
    )
    return parser.parse_args()


def source_value(raw: str) -> str | int:
    return int(raw) if raw.isdigit() else raw


def override_candidate_confidence(
    config: PostprocessorConfig,
    fire: float | None = None,
    smoke: float | None = None,
) -> PostprocessorConfig:
    overrides = {"fire": fire, "smoke": smoke}
    invalid = {
        name: value
        for name, value in overrides.items()
        if value is not None and not 0.0 <= value <= 1.0
    }
    if invalid:
        values = ", ".join(f"{name}={value}" for name, value in invalid.items())
        raise ValueError(f"candidate confidence must be in [0, 1]: {values}")
    thresholds = dict(config.candidate_confidence)
    thresholds.update(
        {name: value for name, value in overrides.items() if value is not None}
    )
    return replace(config, candidate_confidence=thresholds)


def video_timestamp(
    capture: object,
    source: str | int,
    frame_index: int,
    source_fps: float,
    started_at: float,
) -> float:
    import cv2

    if isinstance(source, int):
        return time.monotonic() - started_at
    milliseconds = float(capture.get(cv2.CAP_PROP_POS_MSEC))
    if milliseconds > 0.0:
        return milliseconds / 1000.0
    return frame_index / source_fps if source_fps > 0.0 else time.monotonic() - started_at


def draw_result(frame: object, process_result: object) -> None:
    import cv2

    decision_by_track = {item.track_id: item for item in process_result.decisions}
    for track in process_result.tracks:
        x1, y1, x2, y2 = (int(value) for value in track.bbox)
        decision = decision_by_track[track.track_id]
        color = (0, 0, 255) if decision.passed or decision.already_triggered else (0, 200, 255)
        temporal_metric = (
            f"cv={decision.area_variation:.3f}"
            if decision.area_variation is not None
            else f"{decision.window_hits}/{decision.window_samples}"
        )
        label = (
            f"{track.class_name} #{track.track_id} "
            f"{track.last_confidence:.2f} {temporal_metric} "
            f"{decision.reason}"
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )


def emit_payload(payload: dict[str, object], event_file: object | None) -> None:
    line = json.dumps(payload, ensure_ascii=False)
    print(line, flush=True)
    if event_file is not None:
        event_file.write(line + "\n")
        event_file.flush()


def export_candidate(
    request: object,
    candidate_dir: Path,
    source_id: str,
    source_duration_seconds: float | None,
) -> None:
    from video_verifier.data import ManifestRecord, append_manifest, save_request

    safe_event = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.event_id)
    stamp = int(round(request.timestamps[-1] * 1000.0))
    sample_name = f"{safe_event}_{stamp:012d}.npz"
    sample_path = candidate_dir / "samples" / sample_name
    save_request(sample_path, request)
    append_manifest(
        candidate_dir / "manifest.jsonl",
        ManifestRecord(
            sample=str(Path("samples") / sample_name),
            source_id=source_id,
            split="unlabeled",
            candidate_class=request.candidate_class,
            label=None,
            license=None,
            source_duration_seconds=source_duration_seconds,
        ),
    )


def main() -> int:
    import cv2

    from checkpoint_compat import install_pathlib_checkpoint_compat

    install_pathlib_checkpoint_compat()
    from ultralytics import YOLO

    args = parse_args()
    if args.inference_fps <= 0.0:
        raise ValueError("--inference-fps must be positive")

    original_source = source_value(args.source)
    prepared_source = prepare_video_source(
        original_source,
        cache_dir=args.transcode_cache,
        force_transcode=args.force_transcode,
    )
    source = prepared_source.value
    if prepared_source.transcoded:
        print(
            f"Using FFmpeg-normalized source for OpenCV: {args.source} -> {source}",
            file=sys.stderr,
            flush=True,
        )
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        prepared_source.cleanup()
        raise RuntimeError(f"cannot open source: {args.source}")

    model = YOLO(args.model)
    config = load_config(args.config)
    if args.temporal is not None:
        config = replace(config, temporal_mode=args.temporal)
    config = override_candidate_confidence(
        config,
        fire=args.candidate_confidence_fire,
        smoke=args.candidate_confidence_smoke,
    )
    postprocessor = FireSmokePostprocessor(config)
    verifier_config = None
    clip_builder = None
    video_buffer = None
    predictor = None
    sessions = None
    candidate_dir = Path(args.candidate_dir) if args.candidate_dir else None
    if args.verifier_weights or candidate_dir is not None:
        from video_verifier import (
            ClipBuilder,
            VerificationSessionManager,
            VideoRingBuffer,
            load_verifier_config,
        )

        verifier_config = load_verifier_config(args.verifier_config)
        clip_builder = ClipBuilder(verifier_config)
        video_buffer = VideoRingBuffer(verifier_config)
    if args.verifier_weights:
        from video_verifier import VerifierPredictor

        threshold_overrides = {
            name: value
            for name, value in (
                ("fire", args.verifier_threshold_fire),
                ("smoke", args.verifier_threshold_smoke),
            )
            if value is not None
        }
        predictor = VerifierPredictor(
            args.verifier_weights,
            verifier_config,
            device=args.verifier_device,
            threshold_overrides=threshold_overrides,
        )
        sessions = VerificationSessionManager(predictor.runtime_config)
        verifier_config = predictor.runtime_config
    elif candidate_dir is not None:
        sessions = VerificationSessionManager(verifier_config)
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_duration_seconds = (
        source_frame_count / source_fps
        if source_fps > 0.0 and source_frame_count > 0.0
        else None
    )
    output_fps = source_fps if source_fps > 0.0 else 25.0
    writer = None
    event_file = None
    if args.events:
        event_path = Path(args.events)
        event_path.parent.mkdir(parents=True, exist_ok=True)
        event_file = event_path.open("a", encoding="utf-8")

    frame_index = 0
    next_inference_at = 0.0
    last_process_result = None
    started_at = time.monotonic()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = video_timestamp(capture, source, frame_index, source_fps, started_at)
            if video_buffer is not None:
                video_buffer.append(frame, timestamp)
            if timestamp + 1e-9 >= next_inference_at:
                prediction = model.predict(
                    frame,
                    conf=0.01,
                    imgsz=args.imgsz,
                    device=args.device,
                    verbose=False,
                )[0]
                last_process_result = postprocessor.process_ultralytics(
                    prediction, args.camera_id, timestamp
                )
                track_by_id = {
                    track.track_id: track for track in last_process_result.tracks
                }
                for event in last_process_result.events:
                    payload = asdict(event)
                    payload["reason"] = event.reason.value
                    emit_payload(payload, event_file)
                    if sessions is not None:
                        started_session = sessions.start(
                            event, track_by_id.get(event.track_id)
                        )
                        if started_session is None:
                            emit_payload(
                                {
                                    "message_type": "verification_suppressed",
                                    "event_id": event.event_key,
                                    "candidate_class": event.class_name,
                                    "reason": "alarm_cooldown",
                                    "timestamp": timestamp,
                                },
                                event_file,
                            )

                if sessions is not None:
                    for session in sessions.due(timestamp, track_by_id.values(), args.camera_id):
                        track = track_by_id.get(session.track_id) or session.last_track
                        if track is None:
                            continue
                        try:
                            request = clip_builder.build(
                                video_buffer, track, session.event_id, timestamp
                            )
                            if candidate_dir is not None:
                                export_candidate(
                                    request,
                                    candidate_dir,
                                    str(args.source),
                                    source_duration_seconds,
                                )
                            if predictor is None:
                                continue
                            verification = predictor.predict(request)
                        except RuntimeError as exc:
                            emit_payload(
                                {
                                    "message_type": "verification_error",
                                    "event_id": session.event_id,
                                    "error": str(exc),
                                },
                                event_file,
                            )
                            continue
                        verification_payload = asdict(verification)
                        verification_payload["message_type"] = "verification"
                        verification_payload["timestamp"] = timestamp
                        emit_payload(verification_payload, event_file)
                        alarm = sessions.record(verification, timestamp)
                        if alarm is not None:
                            alarm_payload = asdict(alarm)
                            alarm_payload["message_type"] = "alarm"
                            emit_payload(alarm_payload, event_file)

                active_fps = (
                    verifier_config.verification_fps
                    if sessions is not None and sessions.has_active
                    else args.inference_fps
                )
                next_inference_at = timestamp + 1.0 / active_fps

            if last_process_result is not None:
                draw_result(frame, last_process_result)

            if args.save:
                if writer is None:
                    output_path = Path(args.save)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"cannot create output: {output_path}")
                writer.write(frame)

            if args.show:
                cv2.imshow("Fire/Smoke Detection", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if event_file is not None:
            event_file.close()
        if args.show:
            cv2.destroyAllWindows()
        prepared_source.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
