from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_verifier import collect_predictions
from video_verifier.data import VerificationDataset
from video_verifier.metrics import summarize_predictions
from video_verifier.event_metrics import (
    summarize_event_predictions,
    summarize_source_proxy_predictions,
)
from video_verifier.model import load_model_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a video verifier checkpoint.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    dataset = VerificationDataset(args.manifest, args.split, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model, thresholds, metadata = load_model_checkpoint(args.weights, device)
    loss, labels, probabilities, class_names = collect_predictions(model, loader, device)
    payload = {
        "split": args.split,
        "samples": len(labels),
        "loss": loss,
        "thresholds": thresholds,
        "metrics": summarize_predictions(labels, probabilities, class_names, thresholds),
        "event_metrics": summarize_event_predictions(
            dataset.records, args.manifest, labels, probabilities, thresholds
        ),
        "source_proxy_metrics": summarize_source_proxy_predictions(
            dataset.records, args.manifest, labels, probabilities, thresholds
        ),
        "checkpoint_metadata": metadata,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
