"""Check monitoring reports with fake AWS clients and small ticket examples."""

import hashlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.ticket_monitoring.run_monitoring import run_monitoring


class FakeSession:
    def __init__(self, files, batch_status="Completed"):
        self.files = files
        self.batch_status = batch_status
        self.events = []
        self.metrics = []

    def client(self, service):
        self.events.append(("client", service))
        return self

    def get_object(self, *, Bucket, Key, VersionId):
        self.events.append(("read", Key))
        if Bucket != "test-bucket" or VersionId != "v1":
            raise AssertionError("Unexpected bucket or version.")
        if Key not in self.files:
            raise FileNotFoundError(Key)
        return {"VersionId": "v1", "Body": io.BytesIO(self.files[Key])}

    def describe_training_job(self, *, TrainingJobName):
        self.events.append(("describe", "training"))
        return self.job("Training", TrainingJobName, "Completed", 10)

    def describe_transform_job(self, *, TransformJobName):
        self.events.append(("describe", "batch"))
        return self.job("Transform", TransformJobName, self.batch_status, 20)

    def job(self, prefix, name, status, seconds):
        start = datetime(2026, 10, 4, tzinfo=timezone.utc)
        return {
            f"{prefix}JobName": name,
            f"{prefix}JobStatus": status,
            f"{prefix}StartTime": start,
            f"{prefix}EndTime": start + timedelta(seconds=seconds),
            "FailureReason": "Example batch failure" if status == "Failed" else None,
        }

    def put_metric_data(self, *, Namespace, MetricData):
        self.events.append(("publish", Namespace))
        self.metrics.extend(MetricData)
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}


def make_example(batch_status="Completed"):
    files = {}

    def source(name, content):
        content = content.encode("utf-8")
        files[name] = content
        return {
            "uri": f"s3://test-bucket/{name}",
            "version_id": "v1",
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    baseline = {
        "source": {"uri": "s3://test-bucket/training.csv"},
        "language_proportions": {"en": 1.0},
        "numeric_features": {
            name: {"cut_points": [], "bin_proportions": [1.0]}
            for name in ["text_char_count", "text_word_count"]
        },
        "missing_rates": {"missing_subject": 0.0, "missing_body": 0.0},
    }
    config = {
        "aws_region": "us-east-1",
        "training_job_name": "example-training",
        "batch_job_name": "example-batch",
        "expected_production_rows": 4,
        "queue_labels": ["A", "B"],
        "reference_metrics": {"accuracy": 0.75, "macro_f1": 11 / 15,
                              "weighted_f1": 11 / 15},
        "model_allowed_absolute_drop": 0.05,
        "data_thresholds": {
            "distribution_distance": 0.1, "missing_rate_increase": 0.05,
            "empty_text_rate": 0.0, "unknown_language_rate": 0.0,
            "unseen_language_rate": 0.0,
        },
        "cloudwatch": {"namespace": "OfflineTests", "dimensions": []},
        "training_baseline": source("baseline.json", json.dumps(baseline)),
        "production_data": source(
            "production.csv", "ticket_id,subject,body,language\n" +
            "".join(f"{i},Help,Please help,en\n" for i in range(1, 5))
        ),
        "production_labels": source(
            "labels.csv", "ticket_id,queue\n1,A\n2,A\n3,B\n4,B\n"
        ),
        "prediction_artifacts": [source(
            "predictions.jsonl", "\n".join(
                json.dumps({"ticket_id": str(i), "predicted_queue": queue})
                for i, queue in enumerate(["A", "A", "A", "B"], 1)
            )
        )],
    }
    return config, FakeSession(files, batch_status)


class MonitoringRunnerTests(unittest.TestCase):
    def run_example(self, config, session, publish_metrics=True):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        output = Path(folder.name)
        summary = run_monitoring(config, output, session, publish_metrics)
        reports = {
            path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in output.glob("*.json")
        }
        expected = {
            "infrastructure_status.json", "production_data_quality.json",
            "production_quality.json", "model_quality_checks.json",
            "monitoring_config.json", "monitoring_summary.json",
        }
        self.assertEqual(set(reports), expected)
        self.assertEqual(summary, reports["monitoring_summary.json"])
        return summary, reports

    def assert_model_unavailable(self, summary, reports, session):
        self.assertIsNone(summary["metrics"])
        self.assertIsNone(summary["model_alert_count"])
        for name in ["production_quality.json", "model_quality_checks.json"]:
            self.assertEqual(reports[name]["status"], "UNAVAILABLE")
            self.assertTrue(reports[name]["reason"])
        self.assertIsNone(reports["production_quality.json"]["metrics"])
        self.assertIsNone(reports["model_quality_checks.json"]["checks"])
        self.assertIsNone(reports["model_quality_checks.json"]["alert_count"])
        forbidden = {"ProductionAccuracy", "ProductionMacroF1",
                     "ProductionWeightedF1", "ModelQualityAlertCount"}
        self.assertTrue(forbidden.isdisjoint(m["MetricName"] for m in session.metrics))

    def test_completed_run_publishes_18_metrics_and_expected_scores(self):
        config, session = make_example()
        summary, reports = self.run_example(config, session)
        self.assertEqual(summary["production_rows"], 4)
        for key, expected in config["reference_metrics"].items():
            self.assertAlmostEqual(summary["metrics"][key], expected)
        self.assertEqual(summary["metric_count"], 18)
        self.assertEqual(len(session.metrics), 18)
        values = {item["MetricName"]: item["Value"] for item in session.metrics}
        self.assertEqual(len(values), 18)
        self.assertEqual(values["ProductionAccuracy"], 0.75)
        self.assertAlmostEqual(values["ProductionMacroF1"], 11 / 15)
        self.assertAlmostEqual(values["ProductionWeightedF1"], 11 / 15)
        self.assertEqual(values["TrainingDurationSeconds"], 10)
        self.assertEqual(values["BatchDurationSeconds"], 20)
        self.assertEqual(len(reports["production_data_quality.json"]["checks"]), 8)
        for key in ["data_alert_count", "model_alert_count", "infrastructure_alert_count"]:
            self.assertEqual(summary[key], 0)
        publish_index = session.events.index(("publish", "OfflineTests"))
        self.assertLess(publish_index, session.events.index(("client", "s3")))

        config, session = make_example()
        summary, reports = self.run_example(config, session, publish_metrics=False)
        self.assertFalse(summary["cloudwatch_metrics_published"])
        self.assertEqual(summary["metric_count"], 0)
        self.assertEqual(session.metrics, [])
        self.assertAlmostEqual(summary["metrics"]["accuracy"], 0.75)

    def test_failed_batch_reports_failure_without_reading_predictions(self):
        config, session = make_example("Failed")
        del session.files["predictions.jsonl"]
        summary, reports = self.run_example(config, session)
        self.assertEqual(summary["infrastructure_alert_count"], 1)
        self.assertEqual(summary["data_alert_count"], 0)
        self.assertEqual(summary["metric_count"], 14)
        values = {item["MetricName"]: item["Value"] for item in session.metrics}
        self.assertEqual(values["InfrastructureAlertCount"], 1)
        self.assertEqual(values["BatchJobAlert"], 1)
        self.assertNotIn(("read", "predictions.jsonl"), session.events)
        self.assertNotIn(("read", "labels.csv"), session.events)
        self.assert_model_unavailable(summary, reports, session)

    def test_missing_predictions_or_labels_does_not_hide_data_checks(self):
        for missing in ["predictions.jsonl", "labels.csv"]:
            with self.subTest(missing=missing):
                config, session = make_example()
                del session.files[missing]
                summary, reports = self.run_example(config, session)
                self.assertEqual(summary["infrastructure_alert_count"], 0)
                self.assertEqual(summary["data_alert_count"], 0)
                self.assertEqual(summary["metric_count"], 14)
                self.assert_model_unavailable(summary, reports, session)

    def test_bad_prediction_hash_or_ticket_ids_prevents_scoring(self):
        for problem in ["hash", "ticket_ids"]:
            with self.subTest(problem=problem):
                config, session = make_example()
                artifact = config["prediction_artifacts"][0]
                if problem == "hash":
                    artifact["sha256"] = "0" * 64
                else:
                    content = session.files["predictions.jsonl"].replace(b'"4"', b'"5"')
                    session.files["predictions.jsonl"] = content
                    artifact["sha256"] = hashlib.sha256(content).hexdigest()
                summary, reports = self.run_example(config, session)
                self.assertEqual(summary["infrastructure_alert_count"], 0)
                self.assert_model_unavailable(summary, reports, session)

    def test_missing_production_still_publishes_infrastructure(self):
        config, session = make_example("Failed")
        del session.files["production.csv"]
        summary, reports = self.run_example(config, session)
        self.assertEqual(summary["infrastructure_alert_count"], 1)
        self.assertEqual(summary["metric_count"], 5)
        self.assertIsNone(summary["data_alert_count"])
        data = reports["production_data_quality.json"]
        self.assertEqual(data["status"], "UNAVAILABLE")
        self.assertTrue(data["reason"])
        self.assertIsNone(data["checks"])
        self.assertIsNone(data["alert_count"])
        self.assertNotIn("DataAlertCount", [m["MetricName"] for m in session.metrics])
        self.assert_model_unavailable(summary, reports, session)


if __name__ == "__main__":
    unittest.main()
