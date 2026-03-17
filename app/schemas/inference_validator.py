from typing import Any, Dict, List, Union
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    job_id: str
    file_id: str


class PredictionResponse(BaseModel):
    job_id: str
    task_type: str
    algorithm: str
    num_rows: int
    predictions: List[Union[int, float]]
    data_with_predictions: List[Dict[str, Any]]
    message: str
