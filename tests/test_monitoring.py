import copy
import unittest

from src.ticket_monitoring.checks import (
    classification_metrics,
    data_quality_checks,
    model_quality_checks,
)


class MonitoringTests(unittest.TestCase):

    def setUp(self):
        self.records = [
            {
                "ticket_id": str(i),
                "subject": "Help",
                "body": "Login failed",
                "language": language,
            }
            for i, language in enumerate(["en", "en", "de", "de"])
        ]
        self.baseline = {
            "language_proportions": {"en": 0.5, "de": 0.5},
            "numeric_features": {
                "text_char_count": {
                    "cut_points": [10, 30],
                    "bin_proportions": [0.0, 1.0, 0.0],
                },
                "text_word_count": {
                    "cut_points": [2, 5],
                    "bin_proportions": [0.0, 1.0, 0.0],
                },
            },
            "missing_rates": {
                "missing_subject": 0.0,
                "missing_body": 0.0,
            },
        }
        self.thresholds = {
            "distribution_distance": 0.10,
            "missing_rate_increase": 0.05,
            "empty_text_rate": 0.0,
            "unknown_language_rate": 0.0,
            "unseen_language_rate": 0.0,
        }

    def check_data(self, records):
        return {
            item["check"]: item
            for item in data_quality_checks(
                records, self.baseline, self.thresholds
            )
        }

    def test_matching_data_has_no_alerts(self):
        checks = self.check_data(self.records)
        self.assertEqual(len(checks), 8)
        self.assertTrue(all(c["status"] == "OK" for c in checks.values()))

    def test_empty_tickets_trigger_alert(self):
        records = copy.deepcopy(self.records)
        for record in records:
            record["subject"] = " "
            record["body"] = None

        check = self.check_data(records)["empty_text_rate"]
        self.assertEqual(check["value"], 1.0)
        self.assertEqual(check["status"], "ALERT")

    def test_unseen_language_triggers_alert(self):
        records = copy.deepcopy(self.records)
        records[0]["language"] = "zz"

        check = self.check_data(records)["unseen_language_rate"]
        self.assertEqual(check["value"], 0.25)
        self.assertEqual(check["status"], "ALERT")

    def test_language_distribution_change_triggers_alert(self):
        records = copy.deepcopy(self.records)
        for record in records:
            record["language"] = "en"

        check = self.check_data(records)["language_distribution"]
        self.assertEqual(check["value"], 0.5)
        self.assertEqual(check["status"], "ALERT")

    def test_empty_or_incomplete_data_is_rejected(self):
        for records in ([], [{"ticket_id": "missing-fields"}]):
            with self.subTest(records=records):
                with self.assertRaises(ValueError):
                    self.check_data(records)

    def test_metrics_include_an_absent_class(self):
        result = classification_metrics(
            ["A", "A", "B", "B"],
            ["A", "A", "A", "A"],
            ["A", "B", "C"],
        )
        # Hand-calculated results; C is absent but included in macro F1.
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], 2 / 9)
        self.assertAlmostEqual(result["weighted_f1"], 1 / 3)

    def test_invalid_labels_are_rejected(self):
        cases = [
            ([], [], ["A", "B"]),
            (["A"], ["A", "B"], ["A", "B"]),
            (["A"], ["unknown"], ["A", "B"]),
            (["unknown"], ["A"], ["A", "B"]),
            (["A"], ["A"], ["A", "A"]),
        ]
        for actual, predicted, labels in cases:
            with self.subTest(case=(actual, predicted, labels)):
                with self.assertRaises(ValueError):
                    classification_metrics(actual, predicted, labels)

    def test_quality_threshold_boundary_and_degradation(self):
        names = ("accuracy", "macro_f1", "weighted_f1")
        reference = dict.fromkeys(names, 0.75)

        boundary = model_quality_checks(
            dict.fromkeys(names, 0.50), reference, allowed_drop=0.25
        )
        self.assertTrue(all(c["status"] == "OK" for c in boundary))

        degraded = model_quality_checks(
            dict.fromkeys(names, 0.49), reference, allowed_drop=0.25
        )
        self.assertTrue(all(c["status"] == "ALERT" for c in degraded))

    def test_invalid_metric_values_are_rejected(self):
        names = ("accuracy", "macro_f1", "weighted_f1")
        reference = dict.fromkeys(names, 0.75)

        for invalid in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=invalid):
                metrics = dict.fromkeys(names, 0.75)
                metrics["accuracy"] = invalid
                with self.assertRaises(ValueError):
                    model_quality_checks(metrics, reference, 0.05)


if __name__ == "__main__":
    unittest.main()
