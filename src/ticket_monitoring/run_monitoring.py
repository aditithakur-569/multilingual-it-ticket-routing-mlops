"""Run ticket monitoring and save reports in a SageMaker Processing job."""

import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

if __package__:
    from .checks import classification_metrics, data_quality_checks, model_quality_checks
else:
    from checks import classification_metrics, data_quality_checks, model_quality_checks


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_saved_bytes(client, source):
    location = urlparse(source["uri"])
    version = source["version_id"]
    require(version and version != "null", "Missing S3 version ID.")

    response = client.get_object(
        Bucket=location.netloc,
        Key=location.path.lstrip("/"),
        VersionId=version,
    )
    require(response["VersionId"] == version, "S3 version mismatch.")

    with response["Body"] as body:
        content = body.read()

    if source.get("sha256"):
        require(
            hashlib.sha256(content).hexdigest() == source["sha256"],
            "File fingerprint mismatch.",
        )
    return content


def read_csv_records(content):
    return list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))


def index_records(records):
    result = {}
    for record in records:
        ticket_id = record.get("ticket_id")
        require(isinstance(ticket_id, str) and ticket_id.strip(),
                "Missing or invalid ticket ID.")
        require(ticket_id not in result, "Duplicate ticket ID.")
        result[ticket_id] = record
    return result


def summarize_job(details, kind):
    prefix = "Training" if kind == "training" else "Transform"
    status = details[f"{prefix}JobStatus"]
    start = details.get(f"{prefix}StartTime")
    end = details.get(f"{prefix}EndTime")

    state = (
        "OK" if status == "Completed"
        else "ALERT" if status in {"Failed", "Stopped", "Stopping"}
        else "PENDING"
    )

    return {
        "job_type": kind,
        "job_name": details[f"{prefix}JobName"],
        "aws_status": status,
        "check_status": state,
        "duration_seconds": (
            (end - start).total_seconds() if start and end else None
        ),
        "start_time": start.isoformat() if start else None,
        "end_time": end.isoformat() if end else None,
        "failure_reason": details.get("FailureReason"),
    }


def alert_count(checks):
    return sum(item["status"] == "ALERT" for item in checks)


def save_json(folder, filename, value):
    (folder / filename).write_text(
        json.dumps(value, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def run_monitoring(config, output, session, publish_metrics=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    sagemaker = session.client("sagemaker")

    # Check the jobs before reading their output files.
    jobs = [
        summarize_job(
            sagemaker.describe_training_job(
                TrainingJobName=config["training_job_name"],
            ),
            "training",
        ),
        summarize_job(
            sagemaker.describe_transform_job(
                TransformJobName=config["batch_job_name"],
            ),
            "batch_prediction",
        ),
    ]
    infrastructure_alerts = sum(
        job["check_status"] == "ALERT" for job in jobs
    )
    save_json(output, "infrastructure_status.json", {
        "checked_at": now.isoformat(),
        "jobs": jobs,
        "alert_count": infrastructure_alerts,
    })
    save_json(output, "monitoring_config.json", config)

    metric_data = []

    def add_metric(name, value, unit="None"):
        metric_data.append({
            "MetricName": name,
            "Dimensions": config["cloudwatch"]["dimensions"],
            "Timestamp": now,
            "Value": float(value),
            "Unit": unit,
        })

    def publish(items):
        if not publish_metrics or not items:
            return 0
        response = session.client("cloudwatch").put_metric_data(
            Namespace=config["cloudwatch"]["namespace"],
            MetricData=items,
        )
        require(response["ResponseMetadata"]["HTTPStatusCode"] == 200,
                "CloudWatch metric publication failed.")
        return len(items)

    add_metric("InfrastructureAlertCount", infrastructure_alerts, "Count")
    for job in jobs:
        prefix = "Training" if job["job_type"] == "training" else "Batch"
        if job["duration_seconds"] is not None:
            add_metric(f"{prefix}DurationSeconds",
                       job["duration_seconds"], "Seconds")
        add_metric(f"{prefix}JobAlert",
                   int(job["check_status"] == "ALERT"), "Count")

    # Send job alerts first so missing data cannot hide them.
    infrastructure_metric_count = len(metric_data)
    published_count = publish(metric_data[:])
    s3 = session.client("s3")

    data_checks = None
    production = None
    production_rows = None
    data_reason = None
    reference_source = None

    try:
        baseline = json.loads(read_saved_bytes(s3, config["training_baseline"]))
        reference_source = baseline["source"]
        records = read_csv_records(read_saved_bytes(s3, config["production_data"]))
        production_rows = len(records)
        require(production_rows == config["expected_production_rows"],
                "Unexpected production row count.")
        require(all("queue" not in row for row in records),
                "Production inputs contain queue labels.")
        production = index_records(records)
        data_checks = data_quality_checks(
            records, baseline, config["data_thresholds"],
        )
    except Exception as error:
        # Record why the checks could not run; do not report zero alerts.
        data_reason = f"{type(error).__name__}: {error}"

    data_status = "AVAILABLE" if data_checks is not None else "UNAVAILABLE"
    data_alerts = alert_count(data_checks) if data_checks is not None else None
    save_json(output, "production_data_quality.json", {
        "status": data_status,
        "reason": data_reason,
        "reference_source": reference_source,
        "production_source": config["production_data"],
        "production_rows": production_rows,
        "thresholds": config["data_thresholds"],
        "checks": data_checks,
        "alert_count": data_alerts,
    })

    metrics = None
    model_checks = None
    model_reason = None
    batch_status = jobs[1]["aws_status"]

    if batch_status != "Completed":
        model_reason = f"Batch job status is {batch_status}; predictions were not evaluated."
    elif data_checks is None:
        model_reason = "Production data could not be verified. See the data quality report."
    else:
        try:
            require(config["prediction_artifacts"], "No prediction files are configured.")
            prediction_rows = []
            for source in config["prediction_artifacts"]:
                content = read_saved_bytes(s3, source).decode("utf-8")
                prediction_rows.extend(
                    json.loads(line) for line in content.splitlines() if line.strip()
                )
            predictions = index_records(prediction_rows)
            labels = index_records(
                read_csv_records(read_saved_bytes(s3, config["production_labels"]))
            )
            require(set(production) == set(predictions) == set(labels),
                    "Input, prediction, and label ticket IDs do not match.")

            ticket_ids = list(production)
            actual = [labels[ticket_id]["queue"] for ticket_id in ticket_ids]
            predicted = [
                predictions[ticket_id]["predicted_queue"] for ticket_id in ticket_ids
            ]
            candidate_metrics = classification_metrics(
                actual, predicted, config["queue_labels"],
            )
            candidate_checks = model_quality_checks(
                candidate_metrics,
                config["reference_metrics"],
                config["model_allowed_absolute_drop"],
            )
            metrics, model_checks = candidate_metrics, candidate_checks
        except Exception as error:
            model_reason = f"{type(error).__name__}: {error}"

    model_status = "AVAILABLE" if metrics is not None else "UNAVAILABLE"
    model_alerts = alert_count(model_checks) if model_checks is not None else None
    save_json(output, "production_quality.json", {
        "status": model_status,
        "reason": model_reason,
        "batch_job_name": config["batch_job_name"],
        "ground_truth_source": config.get("production_labels"),
        "metrics": {
            "evaluation_split": "simulated_production",
            "tickets": production_rows,
            **metrics,
        } if metrics is not None else None,
    })
    save_json(output, "model_quality_checks.json", {
        "status": model_status,
        "reason": model_reason,
        "batch_job_name": config["batch_job_name"],
        "reference_metrics": config["reference_metrics"],
        "allowed_absolute_drop": config["model_allowed_absolute_drop"],
        "checks": model_checks,
        "alert_count": model_alerts,
    })

    if data_checks is not None:
        add_metric("DataAlertCount", data_alerts, "Count")
        metric_names = {
            "language_distribution": "LanguageDistributionDistance",
            "text_char_count_distribution": "TextCharDistributionDistance",
            "text_word_count_distribution": "TextWordDistributionDistance",
            "missing_subject_rate_increase": "MissingSubjectRateIncrease",
            "missing_body_rate_increase": "MissingBodyRateIncrease",
            "empty_text_rate": "EmptyTextRate",
            "unknown_language_rate": "UnknownLanguageRate",
            "unseen_language_rate": "UnseenLanguageRate",
        }
        for check in data_checks:
            add_metric(metric_names[check["check"]], check["value"])

    if metrics is not None:
        for metric, name in [
            ("accuracy", "ProductionAccuracy"),
            ("macro_f1", "ProductionMacroF1"),
            ("weighted_f1", "ProductionWeightedF1"),
        ]:
            add_metric(name, metrics[metric])
        add_metric("ModelQualityAlertCount", model_alerts, "Count")

    summary = {
        "checked_at": now.isoformat(),
        "batch_job_name": config["batch_job_name"],
        "production_rows": production_rows,
        "metrics": metrics,
        "data_quality_status": data_status,
        "model_quality_status": model_status,
        "data_quality_reason": data_reason,
        "model_quality_reason": model_reason,
        "data_alert_count": data_alerts,
        "model_alert_count": model_alerts,
        "infrastructure_alert_count": infrastructure_alerts,
        "cloudwatch_metrics_published": False,
        "metric_count": published_count,
    }
    save_json(output, "monitoring_summary.json", summary)
    published_count += publish(metric_data[infrastructure_metric_count:])
    summary["cloudwatch_metrics_published"] = publish_metrics
    summary["metric_count"] = published_count
    save_json(output, "monitoring_summary.json", summary)
    return summary


def main():
    import boto3

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("monitoring_config.json")),
    )
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--publish-metrics", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    session = boto3.Session(region_name=config["aws_region"])
    summary = run_monitoring(config, args.output_dir, session, args.publish_metrics)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


