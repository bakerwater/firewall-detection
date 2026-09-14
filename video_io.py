from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


# OpenCV 5's FFmpeg backend aborts the process while decoding the rawvideo AVI
# files in the bundled datasets. A subprocess cannot catch that native abort, so
# these codecs are normalized before VideoCapture sees them.
UNSAFE_OPENCV_CODECS = frozenset({"rawvideo", "indeo5"})


@dataclass(frozen=True, slots=True)
class VideoProbe:
    codec: str | None
    duration_seconds: float | None
    width: int | None
    height: int | None
    frame_rate: float | None


@dataclass(slots=True)
class PreparedVideoSource:
    value: str | int
    original: str | int
    probe: VideoProbe | None = None
    transcoded: bool = False
    _temporary: tempfile.TemporaryDirectory[str] | None = None

    def cleanup(self) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None


def _fraction(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else None
    return float(value)


def probe_video(path: str | Path) -> VideoProbe | None:
    if shutil.which("ffprobe") is None:
        return None
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout)
        stream = payload.get("streams", [{}])[0]
        duration_raw = payload.get("format", {}).get("duration")
        return VideoProbe(
            codec=str(stream["codec_name"]).lower() if stream.get("codec_name") else None,
            duration_seconds=float(duration_raw) if duration_raw not in {None, "N/A"} else None,
            width=int(stream["width"]) if stream.get("width") else None,
            height=int(stream["height"]) if stream.get("height") else None,
            frame_rate=_fraction(stream.get("avg_frame_rate")),
        )
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _cache_name(path: Path) -> str:
    stat = path.stat()
    identity = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    digest = hashlib.sha256(identity).hexdigest()[:16]
    return f"{path.stem}_{digest}.mp4"


def _transcode(path: Path, output: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"ffmpeg is required to decode the {path.suffix} source safely: {path}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.name + ".partial.mp4")
    completed = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        temporary_output.unlink(missing_ok=True)
        detail = completed.stderr.strip()[-2000:]
        raise RuntimeError(f"ffmpeg failed to transcode {path}: {detail}")
    temporary_output.replace(output)


def prepare_video_source(
    source: str | int,
    cache_dir: str | Path | None = None,
    force_transcode: bool = False,
) -> PreparedVideoSource:
    if isinstance(source, int):
        return PreparedVideoSource(value=source, original=source)
    path = Path(source)
    if not path.is_file():
        return PreparedVideoSource(value=source, original=source)

    probe = probe_video(path)
    should_transcode = force_transcode or bool(
        probe is not None and probe.codec in UNSAFE_OPENCV_CODECS
    )
    if not should_transcode:
        return PreparedVideoSource(value=str(path), original=source, probe=probe)

    temporary = None
    if cache_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="fireware_video_")
        output = Path(temporary.name) / _cache_name(path)
    else:
        output = Path(cache_dir) / _cache_name(path)
    if not output.is_file() or output.stat().st_size == 0:
        _transcode(path, output)
    return PreparedVideoSource(
        value=str(output),
        original=source,
        probe=probe,
        transcoded=True,
        _temporary=temporary,
    )

