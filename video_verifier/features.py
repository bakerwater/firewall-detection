from __future__ import annotations

from typing import Any

from .types import SUPPORTED_CLASSES, VerificationRequest


FREQUENCY_FEATURES = 51
TRACK_FEATURES = 6


def _frequency_features(request: VerificationRequest) -> Any:
    import numpy as np

    if not request.quality.frequency_valid:
        return np.zeros(FREQUENCY_FEATURES, dtype=np.float32)
    frames = np.stack(request.frames).astype(np.float32) / 255.0
    luma = 0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]
    motion = np.zeros(len(frames), dtype=np.float32)
    motion[1:] = np.abs(np.diff(luma, axis=0)).mean(axis=(1, 2))
    width, height = request.frame_size
    frame_area = max(float(width * height), 1.0)
    areas = np.asarray(
        [(x2 - x1) * (y2 - y1) / frame_area for x1, y1, x2, y2 in request.bboxes],
        dtype=np.float32,
    )
    confidences = np.asarray(request.confidences, dtype=np.float32)
    series = np.stack((motion, areas, confidences))
    series = (series - series.mean(axis=1, keepdims=True)) / (
        series.std(axis=1, keepdims=True) + 1e-6
    )
    spectrum = np.fft.rfft(series * np.hanning(series.shape[1]), axis=1)
    features = np.log1p(np.abs(spectrum) ** 2).astype(np.float32)
    return features.reshape(-1)


def _track_features(request: VerificationRequest) -> Any:
    import numpy as np

    width, height = request.frame_size
    frame_area = max(float(width * height), 1.0)
    areas = np.asarray(
        [(x2 - x1) * (y2 - y1) / frame_area for x1, y1, x2, y2 in request.bboxes],
        dtype=np.float32,
    )
    observed = np.asarray(request.observed, dtype=np.float32)
    confidences = np.asarray(request.confidences, dtype=np.float32)
    selected = confidences[observed > 0.5]
    mean_area = float(areas.mean()) if len(areas) else 0.0
    area_growth = float((areas[-1] - areas[0]) / (mean_area + 1e-6))
    area_cv = float(areas.std() / (mean_area + 1e-6))
    hit_ratio = float(observed.mean()) if len(observed) else 0.0
    return np.asarray(
        [
            np.clip(area_growth, -5.0, 5.0),
            np.clip(area_cv, 0.0, 5.0),
            float(selected.mean()) if len(selected) else 0.0,
            float(selected.max()) if len(selected) else 0.0,
            hit_ratio,
            1.0 - hit_ratio,
        ],
        dtype=np.float32,
    )


def preprocess_request(
    request: VerificationRequest,
    rgb_frames: int = 16,
    augment: bool = False,
) -> dict[str, Any]:
    import numpy as np
    import torch

    frames = np.stack(request.frames).astype(np.float32) / 255.0
    indices = np.linspace(0, len(frames) - 1, rgb_frames).round().astype(int)
    rgb = torch.from_numpy(frames[indices]).permute(3, 0, 1, 2)
    if augment and bool(torch.rand(()) < 0.5):
        rgb = rgb.flip(-1)
    if augment:
        brightness = float(torch.empty(()).uniform_(0.9, 1.1))
        contrast = float(torch.empty(()).uniform_(0.9, 1.1))
        mean = rgb.mean(dim=(-2, -1), keepdim=True)
        rgb = ((rgb - mean) * contrast + mean) * brightness
        rgb = rgb.clamp(0.0, 1.0)
    rgb = (rgb - 0.45) / 0.225
    return {
        "rgb": rgb,
        "frequency": torch.from_numpy(_frequency_features(request)),
        "track": torch.from_numpy(_track_features(request)),
        "class_id": torch.tensor(SUPPORTED_CLASSES.index(request.candidate_class)),
        "frequency_valid": torch.tensor(float(request.quality.frequency_valid)),
    }
