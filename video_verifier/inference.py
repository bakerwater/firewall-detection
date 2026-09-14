from __future__ import annotations

import time
from dataclasses import replace

import torch

from .config import VerifierRuntimeConfig
from .features import preprocess_request
from .model import load_model_checkpoint
from .types import VerificationRequest, VerificationResult


class VerifierPredictor:
    def __init__(
        self,
        weights: str,
        runtime_config: VerifierRuntimeConfig,
        device: str | None = None,
        threshold_overrides: dict[str, float] | None = None,
    ) -> None:
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, checkpoint_thresholds, self.metadata = load_model_checkpoint(
            weights, self.device
        )
        thresholds = {**runtime_config.thresholds, **checkpoint_thresholds}
        if threshold_overrides:
            thresholds.update(threshold_overrides)
        self.runtime_config = replace(runtime_config, thresholds=thresholds)
        self.model_version = self.model.config.model_version

    def predict(self, request: VerificationRequest) -> VerificationResult:
        inputs = preprocess_request(
            request, rgb_frames=self.runtime_config.rgb_frames, augment=False
        )
        inputs = {
            name: value.unsqueeze(0).to(self.device, non_blocking=True)
            for name, value in inputs.items()
        }
        started = time.perf_counter()
        autocast_enabled = self.device.type == "cuda"
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if autocast_enabled else torch.bfloat16,
            enabled=autocast_enabled,
        ):
            logit = self.model(**inputs)
            probability = float(torch.sigmoid(logit)[0].float().cpu())
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - started) * 1000.0
        threshold = self.runtime_config.thresholds[request.candidate_class]
        return VerificationResult(
            event_id=request.event_id,
            camera_id=request.camera_id,
            track_id=request.track_id,
            candidate_class=request.candidate_class,
            p_real=probability,
            clip_positive=probability >= threshold,
            threshold=threshold,
            model_version=self.model_version,
            quality=request.quality,
            inference_ms=inference_ms,
        )
