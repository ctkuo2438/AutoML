# AutoTabular — Implementation Plan

## Phase 1 ✅ COMPLETED
- User authentication (register, login, JWT tokens) with bcrypt password hashing
- File upload endpoint (`/api/files/upload`) — stores CSV metadata in DB
- File ownership verification middleware
- Data preprocessing endpoints: summary, handle missing values, handle outliers, encode categorical, scale features, remove duplicates
- 20/20 tests passing

---

## Phase 2 — Model Training

### Requirements
- User selects a file (already uploaded/preprocessed), target column, task type, algorithm, and optional hyperparameters
- Data split into train/test sets (default 80/20)
- Three algorithms: **LightGBM**, **XGBoost**, **Random Forest** for both classification and regression
- Model artifact saved to disk via joblib
- Evaluation metrics returned and persisted in DB
- All endpoints require auth and file ownership checks

### New / Modified Files

| Action | File |
|--------|------|
| New | `app/db/models/training_job_model.py` |
| New | `app/schemas/training_validator.py` |
| New | `app/services/training/trainer.py` |
| New | `app/services/training/__init__.py` |
| New | `app/api/endpoints/training.py` |
| New | `tests/test_training.py` |
| Modified | `app/main.py` |
| Modified | `requirements.txt` |

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/training/train` | Start a training job |
| GET | `/api/training/jobs` | List user's training jobs |
| GET | `/api/training/jobs/{job_id}` | Get job details and metrics |
| DELETE | `/api/training/jobs/{job_id}` | Delete job and model artifact |

### Database: `training_jobs` Table

| Column | Type | Notes |
|--------|------|-------|
| id | String (UUID) | PK, indexed |
| file_id | String | FK -> files.id, NOT NULL |
| user_id | Integer | FK -> users.id, NOT NULL |
| task_type | String | "classification" or "regression" |
| target_column | String | NOT NULL |
| algorithm | String | "lightgbm", "xgboost", "random_forest" |
| hyperparameters | Text | Nullable, JSON string |
| test_size | Float | Default 0.2 |
| random_state | Integer | Default 42 |
| metrics | Text | Nullable, JSON string |
| model_filepath | String | Nullable |
| status | String | "pending", "training", "completed", "failed" |
| error_message | String | Nullable |
| training_duration_seconds | Float | Nullable |
| created_at | DateTime | Default utcnow |
| completed_at | DateTime | Nullable |

### Implementation Steps

#### Phase 2A — DB & Schema
1. Create `TrainingJob` SQLAlchemy model (`app/db/models/training_job_model.py`)
2. Register model in `main.py` for table creation
3. Create Pydantic schemas (`TrainingRequest`, `TrainingResponse`, `ClassificationMetrics`, `RegressionMetrics`) in `app/schemas/training_validator.py`

#### Phase 2B — Training Service
4. Create `ModelTrainer` class (`app/services/training/trainer.py`) with:
   - `_load_and_prepare_data()` — loads CSV, separates X/y, train/test split
   - `_validate_data()` — checks for NaN, non-numeric columns, minimum row count
   - `_get_model()` — factory: returns the right sklearn/lgbm/xgb model instance
   - `_evaluate_classification()` — accuracy, precision, recall, F1, confusion matrix
   - `_evaluate_regression()` — MSE, RMSE, MAE, R2
   - `_save_model()` — saves artifact to `{MODEL_DIR}/{job_id}.joblib`
   - `train()` — orchestrates full pipeline, updates DB record
5. Default hyperparameters:
   - LightGBM: `n_estimators=100, learning_rate=0.1, max_depth=-1, num_leaves=31`
   - XGBoost: `n_estimators=100, learning_rate=0.1, max_depth=6`
   - Random Forest: `n_estimators=100, max_depth=None, min_samples_split=2`
6. Ensure `MODEL_DIR` is created at startup in `main.py`

#### Phase 2C — API Layer
7. Create training router (`app/api/endpoints/training.py`) with 4 endpoints
8. Register router in `main.py` at prefix `/api/training`

#### Phase 2D — Tests (15+ cases)
- Happy path: classification + regression for all 3 algorithms
- Custom hyperparameters
- Validation errors: missing target column, non-numeric features, bad hyperparameters
- File ownership enforcement (403)
- Job list, retrieve, delete

### Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Invalid hyperparameters crash the library | Wrap `fit()` in try/except, store error in DB, return 400 |
| numpy types not JSON-serializable | Cast all metric values to native Python types before storing |
| Non-numeric / NaN data reaches model | `_validate_data()` checks explicitly, returns 400 with guidance to preprocess first |
| Large datasets block the server | Synchronous is acceptable for Phase 2; background tasks deferred to Phase 3 |
| Concurrent training requests | Each job uses unique job_id filename — no shared mutable state |

### Success Criteria
- [ ] LightGBM, XGBoost, Random Forest work for both classification and regression
- [ ] Custom hyperparameters accepted and applied
- [ ] Invalid inputs return clear 400 errors
- [ ] Training job metadata and metrics persisted in DB
- [ ] User can list, retrieve, and delete training jobs
- [ ] Model artifact (.joblib) saved to MODEL_DIR
- [ ] File ownership enforced on all endpoints
- [ ] 15+ new tests passing
- [ ] Existing 20 tests continue to pass

### Not in Phase 2 (deferred)
- Async/background training
- Cross-validation
- Feature importance
- Inference on new data (Phase 3)
- Hyperparameter tuning / grid search
- Model download endpoint

---

## Phase 3 — Inference (Planned)
- Load a saved model artifact by job_id
- Accept a new CSV file for prediction
- Return predictions alongside the original data
- Validate that the new CSV has matching column structure

## Phase 4 — Frontend / Visualization (Planned)
- Interactive interface for CSV upload and task selection
- Data table preview and distribution visualization
- Training configuration form
- Evaluation metrics display (confusion matrix, charts)
- Inference results display
