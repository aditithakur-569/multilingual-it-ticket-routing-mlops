import json
import unittest
from unittest.mock import Mock

import numpy as np

from src.ticket_inference.inference import (
    CONTENT_TYPE,
    input_fn,
    predict_fn,
    output_fn,
)


class InferenceTests(unittest.TestCase):

    def setUp(self):
        self.ticket = {
            "ticket_id": "ticket-1",
            "subject": "Zugriff benötigt",
            "body": "Bitte helfen Sie mir.",
            "language": "de",
        }

    def test_multilingual_input_accepts_utf8_bytes(self):
        payload = json.dumps(
            self.ticket, ensure_ascii=False
        ).encode("utf-8")

        records = input_fn(
            payload, CONTENT_TYPE + "; charset=utf-8"
        )
        self.assertEqual(records, [self.ticket])

    def test_queue_labels_are_rejected(self):
        ticket = dict(self.ticket, queue="Technical Support")
        with self.assertRaises(ValueError):
            input_fn(json.dumps(ticket), CONTENT_TYPE)

    def test_invalid_ticket_fields_are_rejected(self):
        missing_body = dict(self.ticket)
        del missing_body["body"]

        cases = [
            missing_body,
            dict(self.ticket, ticket_id=" "),
            dict(self.ticket, ticket_id=123),
            dict(self.ticket, body=123),
            ["this is not a ticket object"],
        ]

        for ticket in cases:
            with self.subTest(ticket=ticket):
                with self.assertRaises(ValueError):
                    input_fn(json.dumps(ticket), CONTENT_TYPE)

    def test_empty_malformed_and_wrong_content_type_are_rejected(self):
        for payload in ("", " \n ", "{invalid-json"):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    input_fn(payload, CONTENT_TYPE)

        with self.assertRaises(ValueError):
            input_fn(json.dumps(self.ticket), "text/csv")

    def test_predictions_preserve_order_and_prepare_text(self):
        records = [
            {
                "ticket_id": "de-1",
                "subject": " Hilfe ",
                "body": " Anmeldung fehlgeschlagen ",
                "language": " DE ",
            },
            {
                "ticket_id": "en-2",
                "subject": None,
                "body": " Password reset ",
                "language": None,
            },
        ]

        # Fixed probabilities test the serving contract, not model accuracy.
        predictor = Mock()
        predictor.classes_ = np.array([
            "Billing and Payments",
            "Technical Support",
        ])
        predictor.predict_proba.return_value = np.array([
            [0.2, 0.8],
            [0.9, 0.1],
        ])

        model = {
            "pipeline": predictor,
            "model_sha256": "example-model-fingerprint",
        }
        predictions = predict_fn(records, model)

        predictor.predict_proba.assert_called_once_with([
            "Hilfe Anmeldung fehlgeschlagen",
            "Password reset",
        ])

        self.assertEqual(
            [p["ticket_id"] for p in predictions],
            ["de-1", "en-2"],
        )
        self.assertEqual(
            [p["predicted_queue"] for p in predictions],
            ["Technical Support", "Billing and Payments"],
        )
        self.assertEqual(
            [p["language"] for p in predictions],
            ["de", "unknown"],
        )
        self.assertEqual(
            [p["has_subject"] for p in predictions], [1, 0]
        )
        self.assertEqual(
            [p["text_word_count"] for p in predictions], [3, 2]
        )
        self.assertAlmostEqual(predictions[0]["max_probability"], 0.8)
        self.assertAlmostEqual(predictions[1]["max_probability"], 0.9)
        self.assertTrue(all(
            p["model_sha256"] == "example-model-fingerprint"
            for p in predictions
        ))

    def test_empty_text_is_rejected_before_prediction(self):
        predictor = Mock()
        model = {
            "pipeline": predictor,
            "model_sha256": "example-model-fingerprint",
        }
        records = [dict(self.ticket, subject=" ", body=None)]

        with self.assertRaises(ValueError):
            predict_fn(records, model)

        predictor.predict_proba.assert_not_called()

    def test_output_is_json_lines_and_preserves_unicode(self):
        predictions = [
            {"ticket_id": "1", "example_text": "Zugriff benötigt"},
            {"ticket_id": "2", "example_text": "Olá"},
        ]

        body, content_type = output_fn(predictions, CONTENT_TYPE)

        self.assertEqual(content_type, CONTENT_TYPE)
        self.assertEqual(
            [json.loads(line) for line in body.splitlines()],
            predictions,
        )
        self.assertIn("Olá", body)

    def test_output_rejects_invalid_numbers_and_content_type(self):
        with self.assertRaises(ValueError):
            output_fn(
                [{"max_probability": float("nan")}],
                CONTENT_TYPE,
            )

        with self.assertRaises(ValueError):
            output_fn([{"ticket_id": "1"}], "text/csv")


if __name__ == "__main__":
    unittest.main()
