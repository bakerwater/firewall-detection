from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from video_verifier.data import VerificationDataset
from video_verifier.metrics import calibrate_threshold, summarize_predictions
from video_verifier.model import FireSmokeVideoVerifier, VerifierModelConfig


MODEL_INPUTS = ("rgb", "frequency", "track", "class_id", "frequency_valid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the second-stage video verifier.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="weights/video_verifier.pt")
    parser.add_argument("--backbone", choices=("mvit_v2_s", "r3d_18"), default="mvit_v2_s")
    parser.add_argument("--mode", choices=("fusion", "rgb", "frequency"), default="fusion")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--head-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def model_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: batch[name].to(device, non_blocking=True) for name in MODEL_INPUTS}


def configure_backbone_mode(
    model: FireSmokeVideoVerifier, stage: str
) -> None:
    model.backbone.eval()
    if stage != "finetune":
        return
    if model.config.backbone == "mvit_v2_s":
        for block in model.backbone.blocks[-2:]:
            block.train()
        model.backbone.norm.train()
    else:
        model.backbone.layer4.train()


def class_positive_weights(dataset: VerificationDataset, device: torch.device) -> torch.Tensor:
    counts = {name: [0, 0] for name in ("fire", "smoke")}
    for record in dataset.records:
        counts[record.candidate_class][int(record.label)] += 1
    values = []
    for class_name in ("fire", "smoke"):
        negative, positive = counts[class_name]
        values.append(negative / positive if positive else 1.0)
    return torch.tensor(values, dtype=torch.float32, device=device)


def collect_predictions(
    model: FireSmokeVideoVerifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, list[int], list[float], list[str]]:
    model.eval()
    losses, labels, probabilities, class_names = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            target = batch["label"].to(device)
            logits = model(**model_inputs(batch, device))
            losses.append(float(F.binary_cross_entropy_with_logits(logits, target)))
            labels.extend(int(value) for value in target.cpu().tolist())
            probabilities.extend(torch.sigmoid(logits).float().cpu().tolist())
            class_names.extend(
                "fire" if int(value) == 0 else "smoke"
                for value in batch["class_id"].tolist()
            )
    return sum(losses) / max(len(losses), 1), labels, probabilities, class_names


def calibrated_thresholds(
    labels: list[int], probabilities: list[float], class_names: list[str]
) -> dict[str, float]:
    thresholds = {}
    for class_name in ("fire", "smoke"):
        indexes = [index for index, name in enumerate(class_names) if name == class_name]
        thresholds[class_name] = calibrate_threshold(
            [labels[index] for index in indexes],
            [probabilities[index] for index in indexes],
        )
    return thresholds


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.head_epochs < 0 or args.finetune_epochs < 0:
        raise ValueError("batch size and epoch counts must be non-negative")
    if args.head_epochs + args.finetune_epochs <= 0:
        raise ValueError("at least one training epoch is required")
    device = torch.device(args.device)
    train_data = VerificationDataset(args.manifest, "train", augment=True)
    val_data = VerificationDataset(args.manifest, "val", augment=False)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    config = VerifierModelConfig(backbone=args.backbone, mode=args.mode)
    model = FireSmokeVideoVerifier(config, pretrained=not args.no_pretrained).to(device)
    positive_weights = class_positive_weights(train_data, device)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_pr_auc = -1.0
    epochs_without_improvement = 0
    history = []

    def run_stage(stage: str, epochs: int, learning_rate: float) -> None:
        nonlocal best_pr_auc, epochs_without_improvement
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=args.weight_decay,
        )
        for epoch in range(1, epochs + 1):
            model.train()
            configure_backbone_mode(model, stage)
            total_loss = 0.0
            started = time.monotonic()
            for batch in train_loader:
                target = batch["label"].to(device, non_blocking=True)
                class_id = batch["class_id"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    logits = model(**model_inputs(batch, device))
                    loss = F.binary_cross_entropy_with_logits(
                        logits, target, pos_weight=positive_weights[class_id]
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach())

            val_loss, labels, probabilities, class_names = collect_predictions(
                model, val_loader, device
            )
            thresholds = calibrated_thresholds(labels, probabilities, class_names)
            summary = summarize_predictions(labels, probabilities, class_names, thresholds)
            class_pr_auc = [
                summary["by_class"][name]["metrics"]["pr_auc"]
                for name in ("fire", "smoke")
            ]
            pr_auc = sum(float(value or 0.0) for value in class_pr_auc) / 2.0
            row = {
                "stage": stage,
                "epoch": epoch,
                "train_loss": total_loss / max(len(train_loader), 1),
                "val_loss": val_loss,
                "thresholds": thresholds,
                "metrics": summary,
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                epochs_without_improvement = 0
                torch.save(
                    model.checkpoint(
                        thresholds,
                        extra={"best_pr_auc": best_pr_auc, "history": history},
                    ),
                    output,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    break

    model.freeze_backbone()
    run_stage("head", args.head_epochs, args.head_lr)
    epochs_without_improvement = 0
    if args.finetune_epochs:
        model.unfreeze_last_blocks(2)
        run_stage("finetune", args.finetune_epochs, args.finetune_lr)
    print(f"Best checkpoint written to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
