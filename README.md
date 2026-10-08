# Multilingual IT Ticket Routing MLOps

**AAI-540 — Group 3**

- Aditi Jha
- Beakal Zekaryas
- Saman Tavasoli

We built a machine learning system to sort support tickets into 10 queues. The project covers data preparation, feature storage, model training, batch predictions, monitoring, and automated checks using AWS SageMaker and GitHub Actions.

Our selected model uses TF-IDF and Logistic Regression. It performs better than our simple benchmark, but its current accuracy needs improvement before it could route real tickets without staff review.

## Project links

- [GitHub Projects board](https://github.com/users/aditithakur-569/projects/1)
- [Kaggle dataset](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets/data)
- [Recorded successful GitHub Actions run](https://github.com/aditithakur-569/multilingual-it-ticket-routing-mlops/actions/runs/37259244337)
- AWS implementation branch: `sam-aws-implementation`

## Start here

Open [master_project.ipynb](master_project.ipynb) to review the project from start to finish. It combines the code, charts, and saved results from our nine notebooks, with short explanations for each stage.

The master notebook is for review. We have not run it as one continuous session. **Do not use Run All.** Some cells start AWS jobs or update project settings. Use the original notebooks to run or resume individual stages.

Keep the scripts, tests, configuration, requirements, data, and reports with the project. The master notebook depends on those files.

## Dataset and splits

We used Beakal's cleaned master dataset of **44,278 tickets**, with **10 queues** and **five languages**: English, German, Spanish, French, and Portuguese.

| Split | Tickets | Approximate share | Purpose |
|---|---:|---:|---|
| Training | 17,710 | 40% | Train the model and build the text vocabulary |
| Validation | 4,428 | 10% | Compare features and choose model settings |
| Test | 4,428 | 10% | Evaluate the selected model |
| Simulated production | 17,712 | 40% | Run batch predictions and monitoring |

We kept the production queue labels separate from the prediction inputs. After predictions were complete, we matched them with the labels to calculate the scores. These tickets came from the same original dataset; they are not a live stream of new tickets.

## Features and model

We combined each ticket's subject and body into one text field. We also prepared language and seven features describing subject presence and text lengths, and saved the training and validation records in SageMaker Feature Store.

We compared text-only Logistic Regression with a model using text, language, and the extra features. The extra features made little difference in the initial validation results, so we selected text only.

Our selected model uses:

- TF-IDF for individual words and two-word combinations.
- Logistic Regression with `C=10` and `class_weight="balanced"`.
- Macro F1 to choose model settings, giving each queue equal weight in the score.

Our simple benchmark predicts **Technical Support** for every ticket because it is the largest training queue.

We excluded answers and tags to reduce the risk of using information that would not be available when a ticket arrives. We also excluded priority and type because we had not confirmed whether they would be available at routing time.

## Results

| Model and evaluation split | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Majority-class benchmark — test | 0.2956 | 0.0456 | 0.1349 |
| Selected Logistic Regression — validation | 0.4557 | 0.4362 | 0.4569 |
| Selected Logistic Regression — test | 0.4634 | 0.4464 | 0.4653 |
| Selected Logistic Regression — simulated production | 0.4591 | 0.4380 | 0.4605 |

We chose model settings using validation results. We did not use the test or production scores to tune those settings.

The selected model correctly predicted the queue for **46.34% of test tickets**. Results vary by queue and language. Spanish, French, and Portuguese have small test samples, so we need more data to judge performance in those languages. We have not measured business benefits in a real service desk.

Detailed reports are saved in:

- `reports/model_development/`
- `reports/managed_training/`
- `reports/final_test/`
- `reports/monitoring/production_quality.json`

## Notebook guide

This is the review order used in the master notebook. Each original notebook contains its own setup and saved results.

| Order | Notebook | What it covers |
|---|---|---|
| 1 | `project_setup.ipynb` | Dataset checks, splits, S3, Athena, and bucket security checks |
| 2 | `raw_data_setup.ipynb` | Original CSV files, source comparisons, and S3 archive |
| 3 | `EDA/multilingual-it-ticket-routing-Data- Analysis.ipynb` | Missing values, duplicates, ticket counts, and charts |
| 4 | `feature_engineering.ipynb` | Feature preparation, Feature Store uploads, and retrieval checks |
| 5 | `model_training.ipynb` | Benchmark, feature comparisons, tuning, SageMaker training, and test results |
| 6 | `model_deployment.ipynb` | Batch predictions and simulated production results |
| 7 | `model_monitoring.ipynb` | Data, model, and job checks; SageMaker monitoring job; CloudWatch |
| 8 | `model_registry.ipynb` | Register the tested model and check its saved files |
| 9 | `ml_pipeline.ipynb` | Automated training pipeline, failure test, and GitHub tests |

Beakal's earlier feature engineering and baseline experiments are in `Feature Engineering & Baseline Pipeline/`. They were completed before the AWS implementation and may use different data splits.

## Run or resume a stage

We used Python 3.12 in SageMaker Studio with an active AWS Academy lab session and the lab execution role.

Install the saved package versions in the notebook kernel:

```python
%pip install -r requirements-notebooks.txt
```

Restart the kernel after installation. The notebooks use SageMaker SDK **2.257.6** and its v2 imports.

SageMaker Studio also provided packages such as boto3, PyArrow, and plotting tools. `requirements-notebooks.txt` records the main ML and SageMaker packages; it does not include every dependency needed for a completely fresh environment.

After restarting the lab, run the relevant setup or restore cells to load paths, `project_config.json`, and AWS clients. Check saved job names and their status before starting another job. Do not run the whole notebook just to restore the session.

`project_config.json` contains this lab account's resources, S3 file versions, SHA-256 hashes, and job details. Running the project in another AWS account requires updating the settings and creating the required resources. Reading the S3 data requires AWS permissions.

## GitHub tests and automated pipeline

We use GitHub Actions to check our code whenever we push changes to `main` or `sam-aws-implementation`, or open a pull request to `main`.

The checks:

- Look for Python syntax mistakes in our scripts.
- Look for Python syntax mistakes in our notebook cells.
- Run 22 tests for the monitoring and prediction code, including checks for failed jobs and missing files.

The notebook check uses `scripts/check_notebooks.py`. It reads the code without running notebook cells or starting AWS jobs. Passing this check means the Python syntax is valid; it does not prove that every calculation or AWS operation will work.

To run the checks from the project folder:

```bash
python -m pip install -r requirements-ci.txt
python -m compileall -q src tests scripts
python scripts/check_notebooks.py
python -m unittest discover -s tests -p "test_*.py" -v
```

These checks do not need AWS credentials. The prediction tests use fixed example outputs. We checked predictions from the actual model separately through the SageMaker batch job.

[The GitHub run for commit `d3dfe64`](https://github.com/aditithakur-569/multilingual-it-ticket-routing-mlops/actions/runs/37712169108) passed syntax checks for all 11 notebooks and the 17 tests available at that time. We then added five monitoring tests and confirmed that all 22 tests passed locally.

The earlier report in `reports/ci/github_actions_success.json` records the run for commit `6bc0b0b`.

Our SageMaker pipeline runs these steps:

1. Check the training and validation data.
2. Train the selected model.
3. Read and check the validation results.
4. Require macro F1 of at least **0.40** and accuracy above the benchmark.
5. Register a passing model as `PendingManualApproval`, or stop if the scores are too low.

The first successful run registered model version 2. We then tested the failure path with a temporary macro F1 requirement of **0.95**. That test reused earlier results and stopped registration without creating new training or processing jobs.

The score requirements are for our course demonstration. Passing them does not mean a model is ready to route real tickets without review.

We start the AWS pipeline manually. Once started, its steps run automatically. GitHub tests do not launch the pipeline, and registration does not automatically deploy a model.

## Batch predictions, monitoring, and registry

We used SageMaker Batch Transform to predict queues for all **17,712 simulated production tickets**. The project uses batch predictions rather than a real-time endpoint.

Our custom monitoring code runs in a SageMaker Processing job. It checks:

- Changes in language and ticket-length distributions.
- Missing text, empty tickets, and unfamiliar languages.
- Changes in accuracy, macro F1, and weighted F1.
- Training and batch job status and duration.

We publish **18 custom metrics** to CloudWatch. The dashboard is `AAI540-Group3-Ticket-Monitoring` in `us-east-1`.

We run monitoring manually. CloudWatch alarms are configured, but notification actions are disabled. Zero alerts means our configured checks passed for that run; it does not mean the model's overall accuracy is high enough for real use. We have not added CPU, memory, or live-service monitoring.

Registry **version 1** is the original tested model used for batch predictions. **Version 2** was trained by the pipeline and evaluated on validation data. The test and production scores in this README belong to version 1. Both versions were left pending manual approval.

## Original data and S3 storage

We saved five original CSV files unchanged under `raw/kaggle/` in the project bucket. We downloaded each uploaded version and checked its SHA-256 hash against the local file.

The [raw-data manifest](reports/raw_data/raw_data_manifest.json) lists the files, row counts, columns, S3 locations, version IDs, and hashes. The archive code is in `raw_data_setup.ipynb`.

Beakal's notes identify three multilingual files as the sources for the cleaned master. We kept the two German-only files for reference. After making capitalization, spaces, and literal line breaks consistent for comparison, we matched the text of all **44,278 master tickets** to the raw data with matching queue labels. We did not repeat the full cleaning process or check every metadata field. The original Kaggle release number is still unverified.

S3 versioning is enabled. `project_config.json` and the saved reports record file versions, hashes, model files, and job names.

| Dataset | S3 location |
|---|---|
| master | `s3://aai540-group3-tickets-390568313988-us-east-1/datasets/c142eef90e9f/split-v1-seed42/cleaned/master.csv` |
| train | `s3://aai540-group3-tickets-390568313988-us-east-1/datasets/c142eef90e9f/split-v1-seed42/splits/train/train.csv` |
| validation | `s3://aai540-group3-tickets-390568313988-us-east-1/datasets/c142eef90e9f/split-v1-seed42/splits/validation/validation.csv` |
| test | `s3://aai540-group3-tickets-390568313988-us-east-1/datasets/c142eef90e9f/split-v1-seed42/splits/test/test.csv` |
| production | `s3://aai540-group3-tickets-390568313988-us-east-1/datasets/c142eef90e9f/split-v1-seed42/splits/production/production.csv` |
| production_labels | `s3://aai540-group3-tickets-390568313988-us-east-1/datasets/c142eef90e9f/split-v1-seed42/ground_truth/production_labels.csv` |
| master_parquet | `s3://aai540-group3-tickets-390568313988-us-east-1/catalog/c142eef90e9f/master/master.parquet` |

Feature Store exports are listed under `feature_store.datasets` in `project_config.json`. Generated datasets and model files in the local `data/` folder are excluded from Git.

We also checked that the project bucket has default AES-256 encryption and all four bucket-level public-access block settings enabled. The saved check is in `reports/security/s3_security_checks.json`. These checks are not a full review of privacy or access permissions.

## Code and saved reports

- `src/ticket_training/`: training script and settings.
- `src/ticket_inference/`: prediction code and dependencies.
- `src/ticket_monitoring/`: reusable checks and monitoring script.
- `src/ticket_pipeline/`: pipeline scripts, settings, and definition.
- `tests/`: monitoring and prediction tests.
- `reports/monitoring/` and `reports/managed_monitoring/`: monitoring reports.
- `reports/model_registry/`: original model registration details.
- `reports/pipeline/first_execution/`: reports from the successful pipeline run.
- `reports/pipeline/gate_failure_test_verification.json`: results of the failed-score test.
- `reports/ci/github_actions_success.json`: recorded GitHub test result.
- `reports/raw_data/raw_data_manifest.json`: original dataset archive details.
- `reports/security/s3_security_checks.json`: S3 security settings checked during the project.

