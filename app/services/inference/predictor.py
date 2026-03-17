import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.db.models.training_job_model import TrainingJob
from app.services.data.load_csv import load_csv
from app.services.training.trainer import ModelTrainer

logger = logging.getLogger(__name__)


class ModelPredictor:
    def __init__(self, job_id: str, file_id: str, user_id: int, db: Session):
        self.job_id = job_id
        self.file_id = file_id
        self.user_id = user_id
        self.db = db

    def _load_training_job(self) -> TrainingJob:
        job = self.db.query(TrainingJob).filter(TrainingJob.id == self.job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found.")
        if job.user_id != self.user_id:
            raise HTTPException(status_code=403, detail="Access denied.")
        if job.status != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Model is not available. Job status: '{job.status}'.",
            )
        if not job.model_filepath or not os.path.exists(job.model_filepath):
            raise HTTPException(
                status_code=404,
                detail="Model artifact not found on disk. Please retrain.",
            )
        return job

    def _resolve_feature_columns(self, job: TrainingJob, model) -> List[str]:
        """Try to get feature columns from the artifact; fall back to deriving from original CSV."""
        _, feature_columns = ModelTrainer.load_model_artifact(job.model_filepath)
        if feature_columns is not None:
            return feature_columns

        # Legacy artifact: re-derive from original training CSV
        logger.warning(
            "Legacy model artifact for job %s has no feature_columns. "
            "Falling back to original training CSV.",
            job.id,
        )
        try:
            df = load_csv(job.file_id, self.db)
            return [c for c in df.columns if c != job.target_column]
        except HTTPException:
            raise HTTPException(
                status_code=400,
                detail="Cannot determine feature columns for this model. Please retrain.",
            )

    def _load_and_validate_input(self, feature_columns: List[str]):
        df = load_csv(self.file_id, self.db)

        missing = set(feature_columns) - set(df.columns)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Input CSV is missing required columns: {sorted(missing)}. "
                       f"Expected columns: {feature_columns}",
            )

        # Select only the required columns in training order (extra columns ignored)
        df = df[feature_columns]

        non_numeric = df.select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric:
            raise HTTPException(
                status_code=400,
                detail=f"Non-numeric columns in input: {non_numeric}. "
                       "Please encode categorical variables before inference.",
            )

        if df.isnull().any().any():
            cols_with_nan = df.columns[df.isnull().any()].tolist()
            raise HTTPException(
                status_code=400,
                detail=f"Missing values found in columns: {cols_with_nan}. "
                       "Please handle missing values before inference.",
            )

        return df

    def predict(self) -> Dict[str, Any]:
        job = self._load_training_job()
        model, _ = ModelTrainer.load_model_artifact(job.model_filepath)
        feature_columns = self._resolve_feature_columns(job, model)
        input_df = self._load_and_validate_input(feature_columns)

        raw_preds = model.predict(input_df)

        if job.task_type == "classification":
            predictions = [int(p) for p in raw_preds]
        else:
            predictions = [float(p) for p in raw_preds]

        original_rows = load_csv(self.file_id, self.db).to_dict(orient="records")
        data_with_predictions = [
            {**row, "prediction": pred}
            for row, pred in zip(original_rows, predictions)
        ]

        return {
            "job_id": self.job_id,
            "task_type": job.task_type,
            "algorithm": job.algorithm,
            "num_rows": len(predictions),
            "predictions": predictions,
            "data_with_predictions": data_with_predictions,
            "message": "Predictions generated successfully.",
        }
