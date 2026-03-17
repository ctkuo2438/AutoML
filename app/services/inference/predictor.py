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
            logger.warning("predict: job_id=%s not found", self.job_id)
            raise HTTPException(status_code=404, detail="Training job not found.")
        if job.user_id != self.user_id:
            logger.warning(
                "predict: user_id=%d denied access to job_id=%s", self.user_id, self.job_id
            )
            raise HTTPException(status_code=403, detail="Access denied.")
        if job.status != "completed":
            logger.warning(
                "predict: job_id=%s not usable, status=%s", self.job_id, job.status
            )
            raise HTTPException(
                status_code=400,
                detail=f"Model is not available. Job status: '{job.status}'.",
            )
        if not job.model_filepath or not os.path.exists(job.model_filepath):
            logger.error(
                "predict: job_id=%s model artifact missing at path=%s",
                self.job_id, job.model_filepath,
            )
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
        """Returns (feature_df, original_df) — original_df for attaching predictions."""
        original_df = load_csv(self.file_id, self.db)

        missing = set(feature_columns) - set(original_df.columns)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Input CSV is missing required columns: {sorted(missing)}. "
                       f"Expected columns: {feature_columns}",
            )

        # Select only the required columns in training order (extra columns ignored)
        feature_df = original_df[feature_columns]

        non_numeric = feature_df.select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric:
            raise HTTPException(
                status_code=400,
                detail=f"Non-numeric columns in input: {non_numeric}. "
                       "Please encode categorical variables before inference.",
            )

        if feature_df.isnull().any().any():
            cols_with_nan = feature_df.columns[feature_df.isnull().any()].tolist()
            raise HTTPException(
                status_code=400,
                detail=f"Missing values found in columns: {cols_with_nan}. "
                       "Please handle missing values before inference.",
            )

        return feature_df, original_df

    def predict(self) -> Dict[str, Any]:
        logger.info(
            "predict start: job_id=%s file_id=%s user_id=%d",
            self.job_id, self.file_id, self.user_id,
        )
        job = self._load_training_job()
        model, _ = ModelTrainer.load_model_artifact(job.model_filepath)
        feature_columns = self._resolve_feature_columns(job, model)
        input_df, original_df = self._load_and_validate_input(feature_columns)

        raw_preds = model.predict(input_df)

        if job.task_type == "classification":
            predictions = [int(p) for p in raw_preds]
        else:
            predictions = [float(p) for p in raw_preds]

        original_rows = original_df.to_dict(orient="records")
        data_with_predictions = [
            {**row, "prediction": pred}
            for row, pred in zip(original_rows, predictions)
        ]

        logger.info(
            "predict complete: job_id=%s algorithm=%s task_type=%s num_rows=%d",
            self.job_id, job.algorithm, job.task_type, len(predictions),
        )
        return {
            "job_id": self.job_id,
            "task_type": job.task_type,
            "algorithm": job.algorithm,
            "num_rows": len(predictions),
            "predictions": predictions,
            "data_with_predictions": data_with_predictions,
            "message": "Predictions generated successfully.",
        }
