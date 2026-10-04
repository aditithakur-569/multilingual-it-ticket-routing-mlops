import hashlib
import json
from pathlib import Path

import joblib


CONTENT_TYPE = "application/jsonlines"


def model_fn(model_dir):
    """Load the trained pipeline once when the prediction server starts."""
    model_path = Path(model_dir) / "model.joblib"

    return {
        "pipeline": joblib.load(model_path),
        "model_sha256": hashlib.sha256(
            model_path.read_bytes()
        ).hexdigest(),
    }


def input_fn(request_body, content_type):
    """Read one or more JSON Lines ticket records."""
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type != CONTENT_TYPE:
        raise ValueError(f"Expected {CONTENT_TYPE}")

    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")

    records = [
        json.loads(line)
        for line in request_body.splitlines()
        if line.strip()
    ]

    if not records:
        raise ValueError("No ticket records provided")

    required_fields = {"ticket_id", "subject", "body", "language"}

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each ticket must be a JSON object")

        if not required_fields.issubset(record):
            raise ValueError("A ticket is missing required fields")

        if "queue" in record:
            raise ValueError("Prediction inputs must not contain queue labels")

        if not isinstance(record["ticket_id"], str):
            raise ValueError("ticket_id must be a string")
        if not record["ticket_id"].strip():
            raise ValueError("ticket_id must not be empty")

        for field in ["subject", "body", "language"]:
            value = record[field]
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field} must be a string or null")

    return records


def predict_fn(records, model):
    """Apply the same text preparation used during training."""
    subjects = [(record["subject"] or "").strip() for record in records]
    bodies = [(record["body"] or "").strip() for record in records]

    texts = [
        (subject + " " + body).strip()
        for subject, body in zip(subjects, bodies)
    ]

    if any(not text for text in texts):
        raise ValueError("Each ticket needs a nonempty subject or body")

    pipeline = model["pipeline"]
    probabilities = pipeline.predict_proba(texts)
    classes = pipeline.classes_
    winning_indices = probabilities.argmax(axis=1)

    results = []

    for index, record in enumerate(records):
        winning_index = int(winning_indices[index])
        subject = subjects[index]
        body = bodies[index]
        text = texts[index]

        results.append({
            "ticket_id": record["ticket_id"],
            "language": (
                (record["language"] or "").strip().lower() or "unknown"
            ),
            "predicted_queue": str(classes[winning_index]),
            "max_probability": float(
                probabilities[index, winning_index]
            ),
            "has_subject": int(bool(subject)),
            "subject_char_count": len(subject),
            "body_char_count": len(body),
            "text_char_count": len(text),
            "subject_word_count": len(subject.split()),
            "body_word_count": len(body.split()),
            "text_word_count": len(text.split()),
            "model_sha256": model["model_sha256"],
        })

    return results


def output_fn(predictions, accept):
    """Return one JSON line per ticket, preserving input order."""
    media_type = accept.split(";", 1)[0].strip().lower()
    if media_type != CONTENT_TYPE:
        raise ValueError(f"Expected response type {CONTENT_TYPE}")

    response_body = "\n".join(
        json.dumps(record, ensure_ascii=False, allow_nan=False)
        for record in predictions
    )

    return response_body, CONTENT_TYPE
