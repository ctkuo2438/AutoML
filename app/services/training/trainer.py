import json
import logging
import os
import pathlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from fastapi import HTTPException
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models.training_job_model import TrainingJob
from app.services.data.load_csv import load_csv

logger = logging.getLogger(__name__)

MIN_TRAINING_ROWS = 10

DEFAULT_HYPERPARAMETERS = {
    "lightgbm": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": -1, "num_leaves": 31},
    "xgboost": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
    "random_forest": {"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
}

ALLOWED_HYPERPARAMETERS: Dict[str, set] = {
    "lightgbm": {
        "n_estimators", "learning_rate", "max_depth", "num_leaves",
        "min_child_samples", "subsample", "colsample_bytree",
        "reg_alpha", "reg_lambda", "random_state", "n_jobs",
    },
    "xgboost": {
        "n_estimators", "learning_rate", "max_depth", "min_child_weight",
        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
        "gamma", "random_state", "n_jobs",
    },
    "random_forest": {
        "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
        "max_features", "bootstrap", "random_state", "n_jobs", "class_weight",
    },
}


class ModelTrainer:
    def __init__(self, user_id: int, config: Dict[str, Any], db: Session):
        self.user_id = user_id
        self.file_id = config["file_id"]
        self.target_column = config["target_column"]
        self.task_type = config["task_type"]
        self.algorithm = config["algorithm"]
        self.test_size = config.get("test_size", 0.2)
        self.random_state = config.get("random_state", 42)
        self.hyperparameters = config.get("hyperparameters") or {}
        self.experiment_name = config.get("experiment_name")
        self.db = db
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def _load_and_prepare_data(self):
        df = load_csv(self.file_id, self.db)

        if self.target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{self.target_column}' not found. "
                       f"Available columns: {list(df.columns)}",
            )

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Auto label-encode non-numeric feature columns
        for col in X.select_dtypes(exclude=["number"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        return X, y

    def _validate_hyperparameters(self) -> None:
        allowed = ALLOWED_HYPERPARAMETERS.get(self.algorithm, set())
        unknown = set(self.hyperparameters) - allowed
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown hyperparameters for {self.algorithm}: {sorted(unknown)}. "
                       f"Allowed: {sorted(allowed)}",
            )

    def _validate_data(self, X, y):
        if len(X) < MIN_TRAINING_ROWS:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset too small: {len(X)} rows. Minimum {MIN_TRAINING_ROWS} rows required.",
            )

        if X.isnull().any().any():
            cols_with_nan = X.columns[X.isnull().any()].tolist()
            raise HTTPException(
                status_code=400,
                detail=f"Missing values found in columns: {cols_with_nan}. "
                       "Please handle missing values before training.",
            )

        if y.isnull().any():
            raise HTTPException(
                status_code=400,
                detail=f"Missing values found in target column '{self.target_column}'.",
            )

    def _get_model(self):
        params = {**DEFAULT_HYPERPARAMETERS[self.algorithm], **self.hyperparameters}
        try:
            if self.algorithm == "lightgbm":
                import lightgbm as lgb
                if self.task_type == "classification":
                    return lgb.LGBMClassifier(**params)
                return lgb.LGBMRegressor(**params)
            elif self.algorithm == "xgboost":
                import xgboost as xgb
                if self.task_type == "classification":
                    return xgb.XGBClassifier(**params, eval_metric="logloss", verbosity=0)
                return xgb.XGBRegressor(**params, verbosity=0)
            else:  # random_forest
                if self.task_type == "classification":
                    return RandomForestClassifier(**params, random_state=self.random_state)
                return RandomForestRegressor(**params, random_state=self.random_state)
        except TypeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid hyperparameter for {self.algorithm}: {e}",
            )

    def _evaluate_classification(self, y_true, y_pred) -> Dict[str, Any]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    def _evaluate_regression(self, y_true, y_pred) -> Dict[str, Any]:
        mse = float(mean_squared_error(y_true, y_pred))
        return {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2_score": float(r2_score(y_true, y_pred)),
        }

    def _save_model(self, model, job_id: str, feature_columns: List[str]) -> str:
        os.makedirs(settings.MODEL_DIR, exist_ok=True)
        filepath = os.path.join(settings.MODEL_DIR, f"{job_id}.joblib")
        joblib.dump({"model": model, "feature_columns": feature_columns}, filepath)
        return filepath

    @staticmethod
    def load_model_artifact(filepath: str):
        """Load a model artifact. Returns (model, feature_columns).
        feature_columns is None for legacy artifacts saved before Phase 3."""
        model_root = pathlib.Path(settings.MODEL_DIR).resolve()
        safe_path = pathlib.Path(filepath).resolve()
        try:
            safe_path.relative_to(model_root)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid model path.")
        artifact = joblib.load(safe_path)
        if isinstance(artifact, dict) and "model" in artifact:
            return artifact["model"], artifact.get("feature_columns")
        # Legacy format: raw model object
        return artifact, None

    def train(self) -> Dict[str, Any]:
        job = TrainingJob(
            file_id=self.file_id,
            user_id=self.user_id,
            experiment_name=self.experiment_name,
            task_type=self.task_type,
            target_column=self.target_column,
            algorithm=self.algorithm,
            hyperparameters=json.dumps(self.hyperparameters),
            test_size=self.test_size,
            random_state=self.random_state,
            status="training",
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)

        logger.info(
            "training start: job_id=%s file_id=%s algorithm=%s task_type=%s",
            job.id, self.file_id, self.algorithm, self.task_type,
        )
        start_time = time.time()
        try:
            self._validate_hyperparameters()
            X, y = self._load_and_prepare_data()
            self._validate_data(X, y)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

            model = self._get_model()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if self.task_type == "classification":
                metrics = self._evaluate_classification(y_test, y_pred)
            else:
                metrics = self._evaluate_regression(y_test, y_pred)

            model_filepath = self._save_model(model, job.id, X_train.columns.tolist())
            duration = time.time() - start_time

            job.metrics = json.dumps(metrics)
            job.model_filepath = model_filepath
            job.status = "completed"
            job.training_duration_seconds = duration
            job.completed_at = datetime.now(timezone.utc)
            self.db.commit()

            logger.info(
                "training complete: job_id=%s algorithm=%s duration=%.2fs metrics=%s",
                job.id, self.algorithm, duration,
                {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)},
            )
            return {
                "job_id": job.id,
                "file_id": self.file_id,
                "task_type": self.task_type,
                "algorithm": self.algorithm,
                "status": "completed",
                "metrics": metrics,
                "model_available": True,
                "training_duration_seconds": duration,
                "message": "Model trained successfully",
            }

        except HTTPException as e:
            duration = time.time() - start_time
            job.status = "failed"
            job.error_message = e.detail
            job.training_duration_seconds = duration
            self.db.commit()
            logger.warning(
                "training failed: job_id=%s status=%d detail=%s duration=%.2fs",
                job.id, e.status_code, e.detail, duration,
            )
            raise e
        except Exception as e:
            duration = time.time() - start_time
            job.status = "failed"
            job.error_message = str(e)
            job.training_duration_seconds = duration
            self.db.commit()
            logger.exception("training error: job_id=%s duration=%.2fs", job.id, duration)
            raise HTTPException(status_code=500, detail="An unexpected error occurred during training.")
