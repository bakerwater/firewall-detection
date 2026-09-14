from __future__ import annotations

from typing import Iterable


def binary_metrics(
    labels: Iterable[int],
    probabilities: Iterable[float],
    threshold: float,
) -> dict[str, float | int | None]:
    pairs = [(int(label), float(probability)) for label, probability in zip(labels, probabilities)]
    tp = sum(label == 1 and probability >= threshold for label, probability in pairs)
    fn = sum(label == 1 and probability < threshold for label, probability in pairs)
    fp = sum(label == 0 and probability >= threshold for label, probability in pairs)
    tn = sum(label == 0 and probability < threshold for label, probability in pairs)

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0.0
        else None
    )
    return {
        "threshold": threshold,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": ratio(fp, fp + tn),
        "accuracy": ratio(tp + tn, len(pairs)),
        "pr_auc": average_precision(
            [label for label, _ in pairs], [probability for _, probability in pairs]
        ),
    }


def average_precision(labels: list[int], probabilities: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    grouped: dict[float, list[int]] = {}
    for probability, label in zip(probabilities, labels):
        grouped.setdefault(probability, []).append(label)
    true_positives = 0
    predicted_positives = 0
    area = 0.0
    for probability in sorted(grouped, reverse=True):
        group = grouped[probability]
        group_positives = sum(group)
        true_positives += group_positives
        predicted_positives += len(group)
        precision = true_positives / predicted_positives
        area += precision * group_positives / positives
    return area


def calibrate_threshold(labels: list[int], probabilities: list[float]) -> float:
    candidates = sorted({0.0, 1.0, *probabilities})

    def rank(threshold: float) -> tuple[float, float, float, float]:
        metrics = binary_metrics(labels, probabilities, threshold)
        return (
            float(metrics["f1"] or 0.0),
            -float(metrics["false_positive_rate"] or 0.0),
            float(metrics["recall"] or 0.0),
            threshold,
        )

    return max(candidates, key=rank)


def fpr_at_recall(
    labels: list[int], probabilities: list[float], target_recall: float = 0.95
) -> dict[str, float | None]:
    candidates = sorted({0.0, 1.0, *probabilities})
    valid = []
    for threshold in candidates:
        metrics = binary_metrics(labels, probabilities, threshold)
        recall = metrics["recall"]
        fpr = metrics["false_positive_rate"]
        if recall is not None and fpr is not None and recall >= target_recall:
            valid.append((float(fpr), -threshold, float(recall)))
    if not valid:
        return {"target_recall": target_recall, "fpr": None, "threshold": None}
    fpr, negative_threshold, recall = min(valid)
    return {
        "target_recall": target_recall,
        "actual_recall": recall,
        "fpr": fpr,
        "threshold": -negative_threshold,
    }


def summarize_predictions(
    labels: list[int],
    probabilities: list[float],
    class_names: list[str],
    thresholds: dict[str, float],
) -> dict[str, object]:
    by_class = {}
    for class_name in ("fire", "smoke"):
        indexes = [index for index, value in enumerate(class_names) if value == class_name]
        class_labels = [labels[index] for index in indexes]
        class_probabilities = [probabilities[index] for index in indexes]
        by_class[class_name] = {
            "metrics": binary_metrics(
                class_labels, class_probabilities, thresholds[class_name]
            ),
            "fpr_at_95_recall": fpr_at_recall(class_labels, class_probabilities),
        }
    overall_tp = overall_fn = overall_fp = overall_tn = 0
    for label, probability, class_name in zip(labels, probabilities, class_names):
        positive = probability >= thresholds[class_name]
        overall_tp += int(label == 1 and positive)
        overall_fn += int(label == 1 and not positive)
        overall_fp += int(label == 0 and positive)
        overall_tn += int(label == 0 and not positive)
    overall_probabilities = [
        probability - thresholds[class_name] + 0.5
        for probability, class_name in zip(probabilities, class_names)
    ]
    overall = binary_metrics(labels, overall_probabilities, 0.5)
    overall.update(
        {"tp": overall_tp, "fn": overall_fn, "fp": overall_fp, "tn": overall_tn}
    )
    return {"overall": overall, "by_class": by_class}
