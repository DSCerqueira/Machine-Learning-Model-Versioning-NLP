# ml_work_flow

This project versions the SST-2 training data, fits TF-IDF sentiment models with Logistic Regression, SGD Classifier, or LightGBM, and provides a FastAPI control surface with MLflow, Prometheus, and Grafana integration.

## NLP drift monitoring

The classifier's TF-IDF columns are model inputs; the primary monitoring features are interpretable measurements extracted from text and fitted-model output:

- `word_count`: whitespace-delimited word count.
- `char_count`: character count.
- `avg_word_length`: mean word length, or zero for empty text.
- `tfidf_nonzero_count`: number of nonzero values in the fitted sparse TF-IDF row.
- `positive_probability`: probability associated with the positive model class.
- `predicted_sentiment`: predicted model class, monitored as a categorical distribution.

The optional supplementary vocabulary analysis selects up to `TOP_DRIFT_FEATURES` terms using the classifier's coefficients or feature importances. It monitors document presence rather than term-value quantiles and does not influence the primary breach decision. Set `ENABLE_TFIDF_TERM_MONITORING=false` to disable it.

## Persistent PSI baseline

Training builds a baseline only after the final selected Pipeline is refit on the training-plus-validation split. The held-out test data is excluded. The baseline is saved as `nlp_psi_baseline.joblib` alongside that model's artifacts and logged to MLflow at `monitoring/psi_baseline/`.

Each baseline records its schema, model and dataset versions, timestamp, feature configurations, reference counts, fixed quantile boundaries or category sets, immutable samples for KS comparisons, smoothing configuration, and optional selected vocabulary. Sparse features use zero/nonzero categories; constant features use below/equal/above categories; continuous features use reference-only quantile bins.

Drift evaluation loads and validates the active model's saved baseline. It never recomputes a baseline or uses current data to determine bin edges. Training a new model creates a new versioned baseline. Injection, reset, and evaluation leave existing baselines untouched. Models created before baseline support must be retrained before they can be evaluated; historical baselines are not silently generated.

PSI compares feature-distribution changes; it does not directly measure classification accuracy or semantic drift. `PSI_THRESHOLD` defaults to `0.20`. A primary drift breach occurs when any primary feature exceeds that value. Vocabulary-only breaches are reported separately.

## API workflow

1. `POST /api/datasets/version` with `{"name":"sst2"}`.
2. `POST /api/models/train` with `{"model_type":"logistic_regression"}`, `{"model_type":"sgd_classifier"}`, or `{"model_type":"lightgbm"}` to train, register, and save the model and baseline. An empty request continues to default to Logistic Regression.
3. Optionally `POST /api/drift/inject` with `{"feature":"sentence","drift_text":"excellent wonderful","drift_ratio":0.3}`.
4. `POST /api/drift/evaluate` to compare the current dataset with the persisted baseline.
5. `POST /api/drift/reset` to restore `current.csv` from `reference.csv`.

`GET /api/state`, `POST /api/predict`, `POST /api/grafana/dashboard`, `/metrics`, and `/report` remain available. Evaluation results include per-feature PSI, KS and p-values where numerical, Jensen-Shannon divergence, the model/baseline versions, primary breach counts, and supplementary vocabulary drift.

The Step 2 model selector uses the active dataset's immutable `reference.csv` snapshot. Each algorithm is tuned using a stratified five-fold search over a shared train/validation/test split and its fitted TF-IDF vectorizer is saved separately. Model versions are registered as `sentiment_logistic_regression`, `sentiment_sgd_classifier`, or `sentiment_lightgbm`. Test accuracy, weighted precision/recall/F1, confusion matrix, and ROC-AUC when available are recorded in the model manifest and MLflow run.

## Prometheus and Grafana

The application preserves `ml_drift_psi`, `ml_drift_ks_statistic`, `ml_drift_js_divergence`, `ml_drift_breach`, `ml_quality_pass`, `ml_model_accuracy`, and `ml_model_f1`. Labeled drift metrics use the `feature` label and are refreshed for each evaluation. The categorical sentiment feature intentionally has no KS series.

Additional metrics are `ml_nlp_baseline_version` and `ml_nlp_breached_features`. The Grafana dashboard contains separate PSI (with the configured threshold), KS, Jensen-Shannon, and breach panels, alongside dataset/model version, accuracy, quality, and F1 panels.

## Development and tests

Install the development requirements and run the unit tests from the repository root:

```powershell
pip install -r requirements-dev.txt
pytest
```

To rebuild and start the complete Docker Compose stack:

```powershell
docker compose up --build -d
docker compose logs -f control
```

To restart the already-built stack without rebuilding:

```powershell
docker compose restart
```

To train from the dashboard, create a dataset version in Step 1, select an algorithm in Step 2, and click **Train and register model**. The default is Logistic Regression; the same algorithms can be selected through `POST /api/models/train` using the `model_type` request property.

The control service mounts `./workspace` persistently; MLflow, Prometheus, and Grafana retain their existing named volumes.
