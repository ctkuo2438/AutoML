# AutoTabular

AutoTabular is a full-stack, no-code AutoML platform for tabular data. Users upload a CSV, preprocess it, train a tree-based model, inspect evaluation metrics, and run inference — all through a browser UI backed by a FastAPI service.

## Features

- **Authentication** — JWT-based register / login with per-user file ownership
- **CSV Upload** — Upload structured CSV files; metadata stored in SQLite
- **Data Preview** — Inspect column names, types, and row samples before preprocessing
- **Preprocessing** — Handle missing values (mean / median / mode / drop rows), drop specific columns, and apply min-max normalisation; changes are persisted back to the file so training sees them
- **Model Training** — Train LightGBM, XGBoost, or Random Forest for classification or regression; non-numeric columns are label-encoded automatically
- **Evaluation Metrics**
  - Classification: Accuracy, F1, Precision, Recall, Confusion Matrix with TP / FP / FN / TN labels (binary) and Actual / Predicted axes (multi-class)
  - Regression: RMSE, MAE, R²
- **Training History** — List and inspect past training jobs per user
- **Inference** — Upload a new CSV with the same column structure and get predictions back

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18 + TypeScript + Vite + Tailwind CSS |
| Backend | FastAPI (Python 3.10) |
| Database | SQLite via SQLAlchemy |
| ML | LightGBM · XGBoost · scikit-learn (Random Forest) |
| Auth | JWT (python-jose) + bcrypt |
| Model persistence | joblib |

## Project Structure

```
AutoML/
├── app/
│   ├── api/endpoints/      # auth, csv_upload, data_preprocessing, training, inference
│   ├── core/               # settings, config
│   ├── db/                 # SQLAlchemy models & session
│   ├── schemas/            # Pydantic validators
│   └── services/
│       ├── data/           # load_csv, preprocess
│       └── training/       # trainer (ModelTrainer)
├── frontend/
│   └── src/
│       ├── api/            # Axios client
│       ├── context/        # AuthContext
│       └── pages/          # Login, Upload, Preview, Preprocess, Train, Metrics, Inference, Dashboard
├── models/                 # Saved .joblib model artifacts
├── uploads/                # Uploaded CSV files
├── tests/                  # pytest test suite + fixtures
└── requirements.txt
```

## Setup

### Prerequisites

- Python 3.10
- Node.js 18+
- conda (recommended)

### Backend

```bash
conda create --name AutoTabular python=3.10
conda activate AutoTabular
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The backend runs on `http://localhost:8000`. The SQLite database (`automl.db`) is created automatically on first run.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend runs on `http://localhost:5173`.

## Default Model Hyperparameters

| Algorithm | Parameters |
|-----------|-----------|
| LightGBM | n_estimators=100, learning_rate=0.1, max_depth=-1, num_leaves=31 |
| XGBoost | n_estimators=100, learning_rate=0.1, max_depth=6 |
| Random Forest | n_estimators=100, max_depth=None, min_samples_split=2 |

Custom hyperparameters can be passed via the `hyperparameters` field in the training request.

## API Overview

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/auth/register` | Register a new user |
| POST | `/api/auth/login` | Login and receive JWT |
| POST | `/api/data/upload` | Upload a CSV file |
| GET | `/api/data/summary/{file_id}` | Get column info and data summary |
| POST | `/api/data/preprocess/{file_id}` | Apply preprocessing steps |
| POST | `/api/training/train` | Start a training job |
| GET | `/api/training/jobs` | List training jobs |
| GET | `/api/training/jobs/{job_id}` | Get job metrics |
| POST | `/api/inference/predict/{job_id}` | Run inference on new CSV |

## Running Tests

```bash
pytest tests/
```
