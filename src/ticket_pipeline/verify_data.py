import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import boto3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_dataset(s3, split_name, reference):
    location = urlparse(reference["uri"])
    require(
        location.scheme == "s3" and bool(location.netloc),
        f"{split_name}: invalid S3 URI."
    )
    require(
        reference.get("version_id") not in (None, "", "null"),
        f"{split_name}: a saved S3 version is required."
    )

    response = s3.get_object(
        Bucket=location.netloc,
        Key=location.path.lstrip("/"),
        VersionId=reference["version_id"],
    )
    try:
        content = response["Body"].read()
    finally:
        response["Body"].close()

    require(
        response.get("VersionId") == reference["version_id"],
        f"{split_name}: S3 version mismatch."
    )

    fingerprint = hashlib.sha256(content).hexdigest()
    require(
        fingerprint == reference["sha256"],
        f"{split_name}: dataset fingerprint mismatch."
    )

    reader = csv.DictReader(
        io.StringIO(content.decode("utf-8-sig"), newline="")
    )
    required_columns = {"ticket_id", "combined_text", "queue"}
    require(
        required_columns.issubset(set(reader.fieldnames or [])),
        f"{split_name}: required columns are missing."
    )
    rows = list(reader)

    require(
        len(rows) == int(reference["rows"]) and len(rows) > 0,
        f"{split_name}: unexpected row count."
    )

    ticket_ids = []
    queues = set()

    for row_number, row in enumerate(rows, start=2):
        ticket_id = row.get("ticket_id")
        text = row.get("combined_text")
        queue = row.get("queue")

        require(
            isinstance(ticket_id, str) and bool(ticket_id.strip()),
            f"{split_name}: missing ticket ID at row {row_number}."
        )
        require(
            isinstance(text, str) and bool(text.strip()),
            f"{split_name}: empty text at row {row_number}."
        )
        require(
            isinstance(queue, str) and bool(queue.strip()),
            f"{split_name}: missing queue at row {row_number}."
        )

        ticket_ids.append(ticket_id)
        queues.add(queue)

    require(
        len(set(ticket_ids)) == len(ticket_ids),
        f"{split_name}: duplicate ticket IDs."
    )

    summary = {
        "uri": reference["uri"],
        "version_id": reference["version_id"],
        "sha256": fingerprint,
        "rows": len(rows),
        "queue_labels": sorted(queues),
    }

    return content, set(ticket_ids), queues, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output-dir",
        default="/opt/ml/processing/output",
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    datasets = config["datasets"]

    require(
        set(datasets) == {"train", "validation"},
        "This step accepts only training and validation datasets."
    )

    frozen_datasets = config["training_configuration"]["feature_datasets"]
    for split_name in ("train", "validation"):
        require(
            datasets[split_name] == frozen_datasets[split_name],
            f"{split_name}: dataset differs from the training configuration."
        )

    s3 = boto3.client("s3", region_name=config["aws_region"])
    verified = {
        name: verify_dataset(s3, name, datasets[name])
        for name in ("train", "validation")
    }

    require(
        verified["train"][1].isdisjoint(verified["validation"][1]),
        "Training and validation contain overlapping ticket IDs."
    )
    require(
        len(verified["train"][2]) == 10,
        "Training data must contain the project's 10 queues."
    )
    require(
        verified["validation"][2] == verified["train"][2],
        "Training and validation must contain the same 10 queues."
    )

    # Write only after both datasets pass every check.
    # Preserve the original bytes so training can verify the same hashes.
    output_dir = Path(args.output_dir)
    for split_name, result in verified.items():
        destination = output_dir / split_name / f"{split_name}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(result[0])

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "ticket_ids_disjoint": True,
        "datasets": {
            name: result[3] for name, result in verified.items()
        },
    }

    report_path = output_dir / "verification" / "data_verification.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
