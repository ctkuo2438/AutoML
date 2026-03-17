from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class PreprocessingConfig(BaseModel):
    handle_missing: bool = False
    missing_config: Optional[Dict[str, Any]] = None
    handle_outliers: bool = False
    outlier_config: Optional[Dict[str, Any]] = None
    encode_categorical: bool = False
    encoding_config: Optional[Dict[str, Any]] = None
    scale_features: bool = False
    scaling_config: Optional[Dict[str, Any]] = None
    remove_duplicates: bool = False
    save_processed: bool = False
    output_filename: Optional[str] = None


class DataSummaryResponse(BaseModel):
    success: bool
    file_id: str
    summary: Dict[str, Any]
    message: str


class PreprocessingResponse(BaseModel):
    success: bool
    file_id: str
    initial_summary: Dict[str, Any]
    final_summary: Dict[str, Any]
    preprocessing_steps: List[str]
    processed_filepath: Optional[str]
    message: str
