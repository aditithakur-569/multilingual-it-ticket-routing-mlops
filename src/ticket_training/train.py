import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline


def load_verified_data(directory, split_name, config):
    path = Path(directory) / f"{split_name}.csv"
    expected = config["feature_datasets"][split_name]

    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    if fingerprint != expected["sha256"]:
        raise ValueError(f"{split_name}: dataset fingerprint mismatch")

    df = pd.read_csv(
        path,
        dtype={"ticket_id": "string"},
        keep_default_na=False,
    )

    if len(df) != expected["rows"]:
        raise ValueError(f"{split_name}: unexpected row count")
    if not df["ticket_id"].is_unique:
        raise ValueError(f"{split_name}: duplicate ticket IDs")

    return df


def calculate_metrics(actual, predicted, labels):
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(
            actual, predicted,
            labels=labels, average="macro", zero_division=0,
        )),
        "weighted_f1": float(f1_score(
            actual, predicted,
            labels=labels, average="weighted", zero_division=0,
        )),
    }


def main():
    config_path = Path(__file__).with_name("training_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    train = load_verified_data(
        os.environ["SM_CHANNEL_TRAIN"], "train", config
    )
    validation = load_verified_data(
        os.environ["SM_CHANNEL_VALIDATION"], "validation", config
    )

    if set(train["ticket_id"]) & set(validation["ticket_id"]):
        raise ValueError("Training and validation tickets overlap")

    text_column = config["input_column"]
    labels = sorted(train["queue"].unique())

    if len(labels) != 10:
        raise ValueError("Expected 10 training queues")
    if not set(validation["queue"]).issubset(labels):
        raise ValueError("Validation contains unknown queue labels")

    for name, df in [("train", train), ("validation", validation)]:
        if not df[text_column].str.strip().ne("").all():
            raise ValueError(f"{name}: empty ticket text")

    tfidf_settings = dict(config["tfidf"])
    tfidf_settings["ngram_range"] = tuple(
        tfidf_settings["ngram_range"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_settings)),
        ("classifier", LogisticRegression(
            C=config["C"],
            class_weight=config["class_weight"],
            solver=config["solver"],
            max_iter=config["max_iter"],
            random_state=config["random_state"],
        )),
    ])

    print(f"Training on {len(train):,} tickets.", flush=True)
    model.fit(train[text_column], train["queue"])

    predictions = model.predict(validation[text_column])
    metrics = calculate_metrics(
        validation["queue"], predictions, labels
    )

    benchmark = DummyClassifier(strategy="most_frequent")
    benchmark.fit(train[[text_column]], train["queue"])
    benchmark_predictions = benchmark.predict(
        validation[[text_column]]
    )
    benchmark_metrics = calculate_metrics(
        validation["queue"], benchmark_predictions, labels
    )

    model_dir = Path(os.environ["SM_MODEL_DIR"])
    output_dir = Path(os.environ["SM_OUTPUT_DATA_DIR"])
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "model.joblib")

    report = {
        "evaluation_split": "validation",
        "model": metrics,
        "benchmark": benchmark_metrics,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "solver_iterations": int(
            model["classifier"].n_iter_.max()
        ),
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    queue_report = classification_report(
        validation["queue"], predictions,
        labels=labels, output_dict=True, zero_division=0,
    )
    pd.DataFrame(queue_report).T.to_csv(
        output_dir / "validation_by_queue.csv",
        index_label="queue",
    )

    language_rows = []
    evaluation_rows = validation[["language", "queue"]].copy()
    evaluation_rows["prediction"] = predictions

    for language, group in evaluation_rows.groupby("language"):
        language_rows.append({
            "language": language,
            "tickets": len(group),
            "queues_present": group["queue"].nunique(),
            **calculate_metrics(
                group["queue"], group["prediction"], labels
            ),
        })

    pd.DataFrame(language_rows).to_csv(
        output_dir / "validation_by_language.csv", index=False
    )

    pd.DataFrame(
        confusion_matrix(
            validation["queue"], predictions, labels=labels
        ),
        index=labels,
        columns=labels,
    ).to_csv(
        output_dir / "validation_confusion_matrix.csv",
        index_label="actual_queue",
    )

    metadata = {
        "configuration": config,
        "queue_labels": labels,
        "packages": {
            name: version(name)
            for name in [
                "scikit-learn", "numpy", "scipy",
                "pandas", "joblib", "threadpoolctl",
            ]
        },
    }
    (model_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    for metric_name, value in metrics.items():
        print(f"validation:{metric_name}={value:.8f}", flush=True)

    print("Model and validation reports saved.", flush=True)


if __name__ == "__main__":
    main()
