"""Reusable data and model quality checks for ticket monitoring."""

from bisect import bisect_right
from collections import Counter
import math


def data_quality_checks(records, baseline, thresholds):
    if not records:
        raise ValueError("Production data is empty.")

    required = {"ticket_id", "subject", "body", "language"}
    languages = Counter()
    char_counts = []
    word_counts = []
    missing = Counter()

    for record in records:
        if not required.issubset(record):
            raise ValueError("A production record is missing required fields.")

        subject = (record["subject"] or "").strip()
        body = (record["body"] or "").strip()
        text = (subject + " " + body).strip()
        language = (record["language"] or "").strip().lower() or "unknown"

        languages[language] += 1
        char_counts.append(len(text))
        word_counts.append(len(text.split()))
        missing["missing_subject"] += int(not subject)
        missing["missing_body"] += int(not body)
        missing["empty_text"] += int(not text)
        missing["unknown_language"] += int(language == "unknown")

    count = len(records)
    checks = []

    def add(name, value, threshold):
        checks.append({
            "check": name,
            "value": float(value),
            "threshold": float(threshold),
            "status": "ALERT" if value > threshold else "OK",
        })

    reference_languages = baseline["language_proportions"]
    all_languages = set(reference_languages) | set(languages)

    distance = 0.5 * sum(
        abs(reference_languages.get(language, 0.0)
            - languages.get(language, 0) / count)
        for language in all_languages
    )
    add("language_distribution", distance,
        thresholds["distribution_distance"])

    for column, values in [
        ("text_char_count", char_counts),
        ("text_word_count", word_counts),
    ]:
        reference = baseline["numeric_features"][column]
        bin_counts = [0] * len(reference["bin_proportions"])

        for value in values:
            index = bisect_right(reference["cut_points"], value)
            bin_counts[index] += 1

        distance = 0.5 * sum(
            abs(observed / count - expected)
            for observed, expected in zip(
                bin_counts, reference["bin_proportions"]
            )
        )
        add(f"{column}_distribution", distance,
            thresholds["distribution_distance"])

    for column in ["missing_subject", "missing_body"]:
        increase = max(
            0.0,
            missing[column] / count - baseline["missing_rates"][column],
        )
        add(f"{column}_rate_increase", increase,
            thresholds["missing_rate_increase"])

    for column in ["empty_text", "unknown_language"]:
        add(f"{column}_rate", missing[column] / count,
            thresholds[f"{column}_rate"])

    unseen_count = sum(
        number for language, number in languages.items()
        if language not in reference_languages
    )
    add("unseen_language_rate", unseen_count / count,
        thresholds["unseen_language_rate"])

    return checks


def classification_metrics(actual, predicted, queue_labels):
    """Calculate accuracy and F1 across the fixed set of routing queues."""
    if not actual or len(actual) != len(predicted):
        raise ValueError("Labels and predictions must have equal, nonzero length.")

    if not queue_labels or len(set(queue_labels)) != len(queue_labels):
        raise ValueError("Queue labels must be nonempty and unique.")

    allowed = set(queue_labels)
    if not set(actual).issubset(allowed):
        raise ValueError("Ground truth contains an unexpected queue.")
    if not set(predicted).issubset(allowed):
        raise ValueError("Predictions contain an unexpected queue.")

    actual_counts = Counter(actual)
    predicted_counts = Counter(predicted)
    correct_counts = Counter(
        truth for truth, prediction in zip(actual, predicted)
        if truth == prediction
    )

    f1_scores = {}
    for queue in queue_labels:
        denominator = actual_counts[queue] + predicted_counts[queue]
        f1_scores[queue] = (
            2.0 * correct_counts[queue] / denominator
            if denominator else 0.0
        )

    count = len(actual)
    return {
        "accuracy": sum(correct_counts.values()) / count,
        "macro_f1": sum(f1_scores.values()) / len(queue_labels),
        "weighted_f1": sum(
            f1_scores[queue] * actual_counts[queue]
            for queue in queue_labels
        ) / count,
    }


def model_quality_checks(metrics, reference_metrics, allowed_drop):
    if not 0.0 <= allowed_drop <= 1.0:
        raise ValueError("Allowed drop must be between zero and one.")

    checks = []
    for metric in ["accuracy", "macro_f1", "weighted_f1"]:
        current = float(metrics[metric])
        reference = float(reference_metrics[metric])

        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in (current, reference)
        ):
            raise ValueError(f"Invalid value for {metric}.")

        minimum = max(0.0, reference - allowed_drop)
        checks.append({
            "metric": metric,
            "test_reference": reference,
            "production_value": current,
            "minimum_allowed": minimum,
            "status": "ALERT" if current < minimum else "OK",
        })

    return checks
