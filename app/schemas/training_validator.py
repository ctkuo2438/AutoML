from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class TrainingRequest(BaseModel):
    file_id: str
    target_column: str
    task_type: Literal["classification", "regression"]
    algorithm: Literal["lightgbm", "xgboost", "random_forest"]
    hyperparameters: Optional[Dict[str, Any]] = None
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    random_state: int = 42


class ClassificationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]


class RegressionMetrics(BaseModel):
    mse: float
    rmse: float
    mae: float
    r2_score: float


class TrainingResponse(BaseModel):
    job_id: str
    file_id: str
    task_type: str
    algorithm: str
    status: str
    metrics: Optional[Dict[str, Any]] = None
    model_filepath: Optional[str] = None
    training_duration_seconds: Optional[float] = None
    message: str

    class Config:
        from_attributes = True


class TrainingJobListResponse(BaseModel):
    jobs: List[TrainingResponse]
