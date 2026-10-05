# Multilingual IT Ticket Routing MLOps

AAI-540 Group 3: Aditi Jha, Beakal Zekaryas, and Saman Tavasoli.

This project classifies multilingual support tickets into 10 queues.
It demonstrates data preparation, feature storage, model training,
batch prediction, monitoring, and automated quality checks using AWS
SageMaker and GitHub Actions.

The implemented model is TF-IDF with Logistic Regression. Its measured
performance supports a course demonstration and further development;
it is not sufficient for reliable autonomous ticket routing.

## Project links

- [Project board](https://github.com/users/aditithakur-569/projects/1)
- [Dataset source](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets/data)
- [Successful GitHub Actions run](https://github.com/aditithakur-569/multilingual-it-ticket-routing-mlops/actions/runs/37259244337)
- AWS implementation branch: `sam-aws-implementation`

## Data and evaluation splits

We start from Beakal's cleaned master dataset: 44,278 tickets across
10 queues and five languages: English, German, Spanish, French, and Portuguese.

| Split | Tickets | Approximate share | Purpose |
|---|---:|---:|---|
| Training | 17,710 | 40% | Fit the model and learned transformations |
| Validation | 4,428 | 10% | Compare features and select model settings |
| Test | 4,428 | 10% | Evaluate the selected model |
| Simulated production | 17,712 | 40% | Run batch predictions and demonstrate monitoring |

Production inputs exclude queue labels. Labels are stored separately
and joined after prediction to evaluate simulated production performance.
The simulated production data is held-out historical data, not a live feed.

## Features and selected model

Ticket subject and body are combined into one text input. Language and
seven structural features were also prepared and stored with training
and validation records in SageMaker Feature Store.

We compared text-only Logistic Regression with a version using text,
language, and structural features. The additional features did not
meaningfully improve the initial validation results, so we selected
the simpler text-only model.

The selected configuration uses TF-IDF unigrams and bigrams with
Logistic Regression (`C=10`, `class_weight="balanced"`).
The majority-class benchmark always predicts Technical Support.
Answer and tag fields are excluded to reduce leakage risk.
Priority and type are excluded from model inputs because their
availability at routing time was not established.

## Recorded results

| Model and evaluation split | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Majority-class benchmark — test | 0.2956 | 0.0456 | 0.1349 |
| Selected Logistic Regression — test | 0.4634 | 0.4464 | 0.4653 |
| Selected Logistic Regression — simulated production | 0.4591 | 0.4380 | 0.4605 |

Model settings were selected using validation results. Test and production
results were not used for the reported hyperparameter selection.
Detailed results, including performance by queue and language, are in
`reports/model_development/`, `reports/managed_training/`, and
`reports/final_test/`.

Language and queue distributions are imbalanced. Spanish, French, and
Portuguese evaluation samples are small, so their scores need cautious
interpretation. Real-world routing benefits have not been measured.

## Notebook order

| Order | Notebook | Purpose |
|---|---|---|
| 1 | `project_setup.ipynb` | Validate the master data, create splits, store data in S3, and configure Athena |
| 2 | `feature_engineering.ipynb` | Prepare features, ingest them into Feature Store, and verify retrieval |
| 3 | `model_training.ipynb` | Compare models, tune on validation data, run managed training, and evaluate the test set |
| 4 | `model_deployment.ipynb` | Run and verify batch predictions on simulated production tickets |
| 5 | `model_monitoring.ipynb` | Check data, model quality, and job status; run managed monitoring and publish dashboard metrics |
| 6 | `model_registry.ipynb` | Register the tested model and its evaluation evidence for review |
| 7 | `ml_pipeline.ipynb` | Build and verify the automated training pipeline and prepare CI tests |

Beakal's earlier notebooks are retained in `EDA/` and
`Feature Engineering & Baseline Pipeline/`. Their original experiments
precede the AWS implementation and may use different data splits.

## Run or resume the project

Use Python 3.12. For the AWS notebooks, use SageMaker Studio with an
authorized execution role and an active lab session.

Install the pinned notebook dependencies in the notebook kernel:

```python
%pip install -r requirements-notebooks.txt
```

Restart the kernel after installation. The notebooks use SageMaker SDK
2.257.6; their SDK v2 imports are not compatible with SDK v3.

The notebooks were developed in SageMaker Studio with additional
preinstalled packages, including boto3, PyArrow, and plotting tools.
`requirements-notebooks.txt` pins the main ML and SageMaker dependencies;
it is not a complete lockfile for a fresh operating system.

After a lab restart, run the relevant notebook's setup or restore cells
to reload paths, `project_config.json`, and AWS clients. Check saved job
names and status before running any cell that submits a job.
Do not use Run All simply to restore a session.

`project_config.json` records this lab account's resources, S3 versions,
fingerprints, and completed jobs. Running in another AWS account requires
updating the configuration and provisioning the required resources.
S3 access requires AWS permissions.

## Automated code checks and ML pipeline

GitHub Actions checks Python syntax and runs 17 monitoring and inference
unit tests on pushes to `main` and `sam-aws-implementation`, and on pull
requests targeting `main`.

To run the same checks locally from the repository root:

```bash
python -m pip install -r requirements-ci.txt
python -m compileall -q src tests
python -m unittest discover -s tests -p "test_*.py" -v
```

These tests require no AWS credentials. Inference unit tests use controlled
prediction outputs; the separate batch job verifies actual model serving.

The SageMaker pipeline performs data verification, training, evaluation,
and a validation quality gate. Registration requires macro F1 of at least
0.40 and accuracy above the majority-class benchmark. Passing models are
registered as `PendingManualApproval`. The threshold is a course
demonstration criterion, not approval for operational use.

A successful execution registered model version 2. A separate selective
execution used a temporary 0.95 threshold and verified that the failure
branch stopped registration without creating new training or processing jobs.

GitHub CI does not automatically start the AWS pipeline. Pipeline execution,
batch deployment, and monitoring submission are currently initiated manually.
There is no automatic deployment after model registration.

## Deployment and monitoring

The tested original model was served through SageMaker Batch Transform
for all 17,712 simulated production tickets. No real-time endpoint is used.

Custom monitoring runs in a SageMaker Processing job. It checks language
and text-length distributions, missing or empty text, unfamiliar languages,
model-quality changes, and training/batch job status and runtime.

CloudWatch receives 18 custom metrics. The dashboard is
`AAI540-Group3-Ticket-Monitoring` in `us-east-1`.
Monitoring currently runs on demand. Alarms are configured, but notification
actions are disabled. A successful snapshot does not establish continuous
monitoring or acceptable absolute model quality.

Registry version 1 represents the original tested model used for batch
predictions. Version 2 is the pipeline-trained candidate evaluated on
validation data. The reported test and production scores belong to the
original model, not to a separate test evaluation of version 2.

## Data storage and reproducibility

S3 bucket versioning is enabled. Exact object versions, dataset fingerprints,
model references, and job identifiers are recorded in `project_config.json`
and the saved reports.

Five original CSV files are archived unchanged under `raw/kaggle/` in the project S3 bucket. Each uploaded version was downloaded and verified against its local SHA-256 fingerprint.

See `raw_data_setup.ipynb` for the archive workflow and [the raw-data manifest](reports/raw_data/raw_data_manifest.json) for file locations, S3 version IDs, row counts, columns, and fingerprints.

Beakal's cleaning notes identify the three multilingual files as the master's inputs. The two German-only files are archived for reference. All 44,278 master tickets matched raw subject/body text after comparison normalization of whitespace, case, and literal line breaks, with no queue-label disagreements. This verifies normalized text coverage; it does not reproduce every cleaning step or validate all metadata. The original Kaggle release number remains unverified.

The following locations document the cleaned master data and derived datasets.

| Dataset | S3 location |
|---|---|
| master | `s3://aai540-group3-tickets-390568313988-us-east-1/datasets/c142eef90e9f/split-v1-seed42/cleaned/master.csv` |
| train | `s3://aai540-group3-tickets-390568313988-us-east-1/datasets/c142eef90e9f/split-v1-seed42/splits/train/train.csv` |
| validation | `s3://aai540-group3-tickets-390568313988-us-east-1/datasets/c142eef90e9f/split-v1-seed42/splits/validation/validation.csv` |
| test | `s3://aai540-group3-tickets-390568313988-us-east-1/datasets/c142eef90e9f/split-v1-seed42/splits/test/test.csv` |
| production | `s3://aai540-group3-tickets-390568313988-us-east-1/datasets/c142eef90e9f/split-v1-seed42/splits/production/production.csv` |
| production_labels | `s3://aai540-group3-tickets-390568313988-us-east-1/datasets/c142eef90e9f/split-v1-seed42/ground_truth/production_labels.csv` |
| master_parquet | `s3://aai540-group3-tickets-390568313988-us-east-1/catalog/c142eef90e9f/master/master.parquet` |

Feature Store exports are listed under `feature_store.datasets` in
`project_config.json`. Generated local datasets and model files under
`data/` are excluded from Git; their AWS references are retained.

## Evidence and source code

- `src/ticket_training/`: managed training script and configuration.
- `src/ticket_inference/`: serving code and dependencies.
- `src/ticket_monitoring/`: reusable checks and managed monitoring script.
- `src/ticket_pipeline/`: pipeline scripts, configuration, and definition.
- `tests/`: monitoring and inference unit tests.
- `reports/monitoring/` and `reports/managed_monitoring/`: monitoring evidence.
- `reports/model_registry/`: original model registration evidence.
- `reports/pipeline/first_execution/`: verified successful pipeline reports.
- `reports/pipeline/gate_failure_test_verification.json`: intentional failure evidence.
- `reports/ci/github_actions_success.json`: successful GitHub CI run reference.
