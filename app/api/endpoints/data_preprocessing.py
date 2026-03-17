from typing import Annotated, List, Literal, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import Field

from app.api.deps import get_preprocessor
from app.schemas.preprocessing_validator import (
    DataSummaryResponse,
    PreprocessingConfig,
    PreprocessingResponse,
)
from app.services.data.preprocess import DataPreprocessor, preprocess_data

router = APIRouter()


@router.get("/summary/{file_id}", response_model=DataSummaryResponse)
async def get_data_summary(preprocessor: DataPreprocessor = Depends(get_preprocessor)):
    return DataSummaryResponse(
        success=True,
        file_id=preprocessor.file_id,
        summary=preprocessor.get_data_summary(),
        message="Data summary retrieved successfully",
    )


@router.post("/preprocess/{file_id}", response_model=PreprocessingResponse)
async def preprocess_csv_data(
    config: PreprocessingConfig,
    preprocessor: DataPreprocessor = Depends(get_preprocessor),
):
    result = preprocess_data(preprocessor.file_id, preprocessor.db, config.model_dump())
    return PreprocessingResponse(**result)


@router.post("/missing-values/{file_id}")
async def handle_missing_values(
    strategy: Literal[
        "remove_column", "remove_rows", "fill_custom", "mean", "median", "mode", "drop"
    ] = "remove_column",
    missing_threshold: Annotated[float, Query(ge=0.0, le=1.0)] = 0.8,
    columns: Optional[List[str]] = None,
    custom_value: Optional[str] = None,
    preprocessor: DataPreprocessor = Depends(get_preprocessor),
):
    preprocessor.handle_missing_values(
        strategy=strategy,
        missing_threshold=missing_threshold,
        columns=columns,
        custom_value=custom_value,
    )
    return {
        "success": True,
        "file_id": preprocessor.file_id,
        "strategy": strategy,
        "missing_threshold": missing_threshold,
        "custom_value": custom_value,
        "columns_processed": columns or "all",
        "summary": preprocessor.get_data_summary(),
        "preprocessing_steps": preprocessor.preprocessing_steps,
        "message": "Missing values handled successfully",
    }


@router.post("/outliers/{file_id}")
async def handle_outliers(
    method: Literal["iqr", "zscore", "clip", "valid_classes"] = "iqr",
    columns: Optional[List[str]] = Query(None),
    threshold: Annotated[float, Query(ge=0.0)] = 1.5,
    valid_classes: Optional[List[str]] = Query(None),
    preprocessor: DataPreprocessor = Depends(get_preprocessor),
):
    parsed_classes = None
    if valid_classes is not None:
        parsed_classes = []
        for val in valid_classes:
            try:
                parsed_classes.append(float(val) if "." in val else int(val))
            except ValueError:
                parsed_classes.append(val)

    preprocessor.handle_outliers(method=method, columns=columns, threshold=threshold, valid_classes=parsed_classes)
    return {
        "success": True,
        "file_id": preprocessor.file_id,
        "method": method,
        "threshold": threshold,
        "valid_classes": valid_classes,
        "columns_processed": columns or "all",
        "summary": preprocessor.get_data_summary(),
        "preprocessing_steps": preprocessor.preprocessing_steps,
        "message": "Outliers handled successfully",
    }


@router.post("/encode/{file_id}")
async def encode_categorical_variables(
    method: Literal["label", "onehot"] = "label",
    columns: Optional[List[str]] = None,
    preprocessor: DataPreprocessor = Depends(get_preprocessor),
):
    preprocessor.encode_categorical_variables(columns=columns, method=method)
    return {
        "success": True,
        "file_id": preprocessor.file_id,
        "method": method,
        "columns_processed": columns or "all categorical",
        "summary": preprocessor.get_data_summary(),
        "preprocessing_steps": preprocessor.preprocessing_steps,
        "message": "Categorical variables encoded successfully",
    }


@router.post("/scale/{file_id}")
async def scale_features(
    method: Literal["standard", "minmax"] = "standard",
    columns: Optional[List[str]] = None,
    preprocessor: DataPreprocessor = Depends(get_preprocessor),
):
    preprocessor.scale_features(columns=columns, method=method)
    return {
        "success": True,
        "file_id": preprocessor.file_id,
        "method": method,
        "columns_processed": columns or "all numeric",
        "summary": preprocessor.get_data_summary(),
        "preprocessing_steps": preprocessor.preprocessing_steps,
        "message": "Features scaled successfully",
    }


@router.delete("/duplicates/{file_id}")
async def remove_duplicates(preprocessor: DataPreprocessor = Depends(get_preprocessor)):
    preprocessor.remove_duplicates()
    return {
        "success": True,
        "file_id": preprocessor.file_id,
        "summary": preprocessor.get_data_summary(),
        "preprocessing_steps": preprocessor.preprocessing_steps,
        "message": "Duplicate rows removed successfully",
    }


@router.post("/save/{file_id}")
async def save_processed_data(
    filename: Optional[str] = None,
    preprocessor: DataPreprocessor = Depends(get_preprocessor),
):
    filepath = preprocessor.save_processed_data(filename=filename)
    return {
        "success": True,
        "file_id": preprocessor.file_id,
        "processed_filepath": filepath,
        "message": "Processed data saved successfully",
    }


@router.post("/reset/{file_id}")
async def reset_data(preprocessor: DataPreprocessor = Depends(get_preprocessor)):
    preprocessor.reset_data()
    return {
        "success": True,
        "file_id": preprocessor.file_id,
        "summary": preprocessor.get_data_summary(),
        "message": "Data reset to original state successfully",
    }
