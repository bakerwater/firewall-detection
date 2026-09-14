from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from temporal_gates import FireSmokePostprocessor, load_config


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
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save", help="Optional annotated video output path")
    parser.add_argument("--events", help="Optional JSONL trigger event output path")
    return parser.parse_args()


def source_value(raw: str) -> str | int:
    return int(raw) if raw.isdigit() else raw


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


def main() -> int:
    import cv2
    from ultralytics import YOLO

    args = parse_args()
    if args.inference_fps <= 0.0:
        raise ValueError("--inference-fps must be positive")

    source = source_value(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source: {args.source}")

    model = YOLO(args.model)
    config = load_config(args.config)
    if args.temporal is not None:
        config = replace(config, temporal_mode=args.temporal)
    postprocessor = FireSmokePostprocessor(config)
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
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
                next_inference_at = timestamp + 1.0 / args.inference_fps
                for event in last_process_result.events:
                    payload = asdict(event)
                    payload["reason"] = event.reason.value
                    line = json.dumps(payload, ensure_ascii=False)
                    print(line, flush=True)
                    if event_file is not None:
                        event_file.write(line + "\n")
                        event_file.flush()

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
                cv2.imshow("Fire/Smoke TPT", frame)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
