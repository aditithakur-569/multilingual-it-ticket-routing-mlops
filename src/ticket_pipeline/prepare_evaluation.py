import argparse
import hashlib
import io
import json
import math
import tarfile
from pathlib import Path
from urllib.parse import urlparse

import boto3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def download_artifact(s3, uri):
    location = urlparse(uri)
    require(
        location.scheme == "s3" and bool(location.netloc),
        "Invalid S3 artifact URI."
    )

    response = s3.get_object(
        Bucket=location.netloc,
        Key=location.path.lstrip("/"),
    )
    try:
        content = response["Body"].read()
    finally:
        response["Body"].close()

    version_id = response.get("VersionId")
    require(
        version_id not in (None, "", "null"),
        "Artifact must have an S3 version ID."
    )

    reference = {
        "uri": uri,
        "version_id": version_id,
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    return content, reference


def read_archive_json(content, filename):
    # Read the requested JSON directly without extracting the archive.
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
        matches = [
            member for member in archive.getmembers()
            if member.isfile() and Path(member.name).name == filename
        ]
        require(
            len(matches) == 1,
            f"Expected exactly one {filename} in the archive."
        )
        with archive.extractfile(matches[0]) as stream:
            return json.load(stream)


def validate_metrics(metrics, description):
    for name in ("accuracy", "macro_f1", "weighted_f1"):
        value = metrics.get(name)
        require(
            type(value) in (int, float)
            and math.isfinite(value)
            and 0 <= value <= 1,
            f"{description}: invalid {name}."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-job-name", required=True)
    parser.add_argument(
        "--output-dir",
        default="/opt/ml/processing/output",
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    session = boto3.Session(region_name=config["aws_region"])
    sagemaker = session.client("sagemaker")
    s3 = session.client("s3")

    job = sagemaker.describe_training_job(
        TrainingJobName=args.training_job_name
    )
    require(
        job["TrainingJobStatus"] == "Completed",
        "The training job has not completed successfully."
    )
    require(
        job["AlgorithmSpecification"]["TrainingImage"]
        == config["training_image"],
        "Training image differs from the pipeline configuration."
    )

    model_uri = job["ModelArtifacts"]["S3ModelArtifacts"]
    output_uri = model_uri.rsplit("/", 1)[0] + "/output.tar.gz"

    model_bytes, model_reference = download_artifact(s3, model_uri)
    output_bytes, output_reference = download_artifact(s3, output_uri)

    metadata = read_archive_json(model_bytes, "model_metadata.json")
    evaluation = read_archive_json(output_bytes, "evaluation.json")

    require(
        metadata["configuration"] == config["training_configuration"],
        "The model settings differ from the frozen training configuration."
    )
    require(
        len(set(metadata["queue_labels"])) == 10,
        "The model must contain 10 distinct queue labels."
    )
    require(
        evaluation["evaluation_split"] == "validation",
        "The pipeline quality gate must use validation results."
    )
    require(
        evaluation["training_rows"] == config["datasets"]["train"]["rows"],
        "Training row count mismatch."
    )
    require(
        evaluation["validation_rows"]
        == config["datasets"]["validation"]["rows"],
        "Validation row count mismatch."
    )

    validate_metrics(evaluation["model"], "Model results")
    validate_metrics(evaluation["benchmark"], "Benchmark results")

    report = {
        "evaluation_split": "validation",
        "training_job_name": args.training_job_name,
        "training_rows": evaluation["training_rows"],
        "evaluated_tickets": evaluation["validation_rows"],
        "multiclass_classification_metrics": {
            name: {"value": evaluation["model"][name]}
            for name in ("accuracy", "macro_f1", "weighted_f1")
        },
        "benchmark": evaluation["benchmark"],
        "accuracy_improvement": (
            evaluation["model"]["accuracy"]
            - evaluation["benchmark"]["accuracy"]
        ),
        "evaluation_dataset": config["datasets"]["validation"],
    }

    assets = {
        "training_job_name": args.training_job_name,
        "model_artifact": model_reference,
        "training_output_artifact": output_reference,
        "training_configuration_verified": True,
        "queue_labels": metadata["queue_labels"],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, value in (
        ("evaluation.json", report),
        ("model_assets.json", assets),
    ):
        (output_dir / filename).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n"
        )

    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
