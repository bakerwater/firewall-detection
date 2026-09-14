from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .features import FREQUENCY_FEATURES, TRACK_FEATURES


@dataclass(frozen=True, slots=True)
class VerifierModelConfig:
    backbone: str = "mvit_v2_s"
    mode: str = "fusion"
    frequency_dim: int = 128
    track_dim: int = 32
    class_embedding_dim: int = 16
    fusion_dim: int = 256
    dropout: float = 0.3
    model_version: str = "mvit-v2-s-verifier-v1"

    def __post_init__(self) -> None:
        if self.backbone not in {"mvit_v2_s", "r3d_18"}:
            raise ValueError("backbone must be mvit_v2_s or r3d_18")
        if self.mode not in {"fusion", "rgb", "frequency"}:
            raise ValueError("mode must be fusion, rgb, or frequency")


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if name == "mvit_v2_s":
        from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s

        weights = MViT_V2_S_Weights.DEFAULT if pretrained else None
        backbone = mvit_v2_s(weights=weights)
        feature_dim = backbone.head[1].in_features
        backbone.head = nn.Identity()
        return backbone, feature_dim
    from torchvision.models.video import R3D_18_Weights, r3d_18

    weights = R3D_18_Weights.DEFAULT if pretrained else None
    backbone = r3d_18(weights=weights)
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feature_dim


class FireSmokeVideoVerifier(nn.Module):
    def __init__(
        self,
        config: VerifierModelConfig | None = None,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.config = config or VerifierModelConfig()
        self.backbone, self.rgb_dim = _build_backbone(self.config.backbone, pretrained)
        self.frequency_encoder = nn.Sequential(
            nn.Linear(FREQUENCY_FEATURES + 1, 256),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(256, self.config.frequency_dim),
            nn.GELU(),
        )
        self.track_encoder = nn.Sequential(
            nn.Linear(TRACK_FEATURES, self.config.track_dim), nn.GELU()
        )
        self.class_embedding = nn.Embedding(2, self.config.class_embedding_dim)
        input_dim = (
            self.rgb_dim
            + self.config.frequency_dim
            + self.config.track_dim
            + self.config.class_embedding_dim
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.config.fusion_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.class_heads = nn.ModuleList(
            [nn.Linear(self.config.fusion_dim, 1) for _ in range(2)]
        )

    def forward(
        self,
        rgb: torch.Tensor,
        frequency: torch.Tensor,
        track: torch.Tensor,
        class_id: torch.Tensor,
        frequency_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch = rgb.shape[0]
        if self.config.mode == "frequency":
            rgb_features = torch.zeros(
                (batch, self.rgb_dim), device=rgb.device, dtype=rgb.dtype
            )
        else:
            rgb_features = self.backbone(rgb)

        validity = frequency_valid.float().reshape(batch, 1)
        frequency_input = torch.cat((frequency.float(), validity), dim=1)
        frequency_features = self.frequency_encoder(frequency_input) * validity
        track_features = self.track_encoder(track.float())
        if self.config.mode == "rgb":
            frequency_features = torch.zeros_like(frequency_features)
            track_features = torch.zeros_like(track_features)
        class_features = self.class_embedding(class_id.long())
        fused = self.fusion(
            torch.cat(
                (rgb_features.float(), frequency_features, track_features, class_features),
                dim=1,
            )
        )
        logits = torch.cat([head(fused) for head in self.class_heads], dim=1)
        return logits.gather(1, class_id.long().reshape(-1, 1)).squeeze(1)

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_last_blocks(self, count: int = 2) -> None:
        self.freeze_backbone()
        if self.config.backbone == "mvit_v2_s":
            for block in self.backbone.blocks[-count:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad = True
        else:
            for parameter in self.backbone.layer4.parameters():
                parameter.requires_grad = True

    def checkpoint(
        self,
        thresholds: dict[str, float],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "format_version": 1,
            "model_config": asdict(self.config),
            "state_dict": self.state_dict(),
            "thresholds": thresholds,
            "extra": extra or {},
        }


def load_model_checkpoint(
    path: str,
    device: str | torch.device = "cpu",
) -> tuple[FireSmokeVideoVerifier, dict[str, float], dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported verifier checkpoint format")
    config = VerifierModelConfig(**payload["model_config"])
    model = FireSmokeVideoVerifier(config=config, pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, dict(payload["thresholds"]), dict(payload.get("extra", {}))
