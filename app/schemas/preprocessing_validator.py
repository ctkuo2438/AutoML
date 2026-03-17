from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class MissingConfig(BaseModel):
    strategy: Literal[
        "remove_column", "remove_rows", "fill_custom", "mean", "median", "mode", "drop"
    ] = "remove_column"
    missing_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    columns: Optional[List[str]] = None
    custom_value: Optional[str] = None
    drop_columns: Optional[List[str]] = None


class OutlierConfig(BaseModel):
    method: Literal["iqr", "zscore", "clip", "valid_classes"] = "iqr"
    columns: Optional[List[str]] = None
    threshold: float = Field(default=1.5, ge=0.0)
    valid_classes: Optional[List[str]] = None


class EncodingConfig(BaseModel):
    method: Literal["label", "onehot"] = "label"
    columns: Optional[List[str]] = None


class ScalingConfig(BaseModel):
    method: Literal["standard", "minmax"] = "standard"
    columns: Optional[List[str]] = None


class PreprocessingConfig(BaseModel):
    handle_missing: bool = False
    missing_config: Optional[MissingConfig] = None
    handle_outliers: bool = False
    outlier_config: Optional[OutlierConfig] = None
    encode_categorical: bool = False
    encoding_config: Optional[EncodingConfig] = None
    scale_features: bool = False
    scaling_config: Optional[ScalingConfig] = None
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
