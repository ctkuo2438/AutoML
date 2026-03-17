from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.preprocessing_validator import (
    DataSummaryResponse,
    PreprocessingConfig,
    PreprocessingResponse,
)
from app.services.auth_service import verify_token
from app.services.data.load_csv import verify_file_ownership
from app.services.data.preprocess import DataPreprocessor, preprocess_data

router = APIRouter()


@router.get("/summary/{file_id}", response_model=DataSummaryResponse)
async def get_data_summary(
    file_id: str,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db),
):
    try:
        verify_file_ownership(file_id, user_id, db)
        preprocessor = DataPreprocessor(file_id, db)
        preprocessor.load_data()
        summary = preprocessor.get_data_summary()
        return DataSummaryResponse(
            success=True,
            file_id=file_id,
            summary=summary,
            message="Data summary retrieved successfully",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting data summary: {e}")


@router.post("/preprocess/{file_id}", response_model=PreprocessingResponse)
async def preprocess_csv_data(
    file_id: str,
    config: PreprocessingConfig,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db),
):
    try:
        verify_file_ownership(file_id, user_id, db)
        result = preprocess_data(file_id, db, config.model_dump())
        return PreprocessingResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in data preprocessing: {e}")


@router.post("/missing-values/{file_id}")
async def handle_missing_values(
    file_id: str,
    strategy: str = "remove_column",
    missing_threshold: float = 0.8,
    columns: Optional[List[str]] = None,
    custom_value: Optional[str] = None,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db),
):
    try:
        verify_file_ownership(file_id, user_id, db)
        preprocessor = DataPreprocessor(file_id, db)
        preprocessor.load_data()
        preprocessor.handle_missing_values(
            strategy=strategy,
            missing_threshold=missing_threshold,
            columns=columns,
            custom_value=custom_value,
        )
        return {
            "success": True,
            "file_id": file_id,
            "strategy": strategy,
            "missing_threshold": missing_threshold,
            "custom_value": custom_value,
            "columns_processed": columns or "all",
            "summary": preprocessor.get_data_summary(),
            "preprocessing_steps": preprocessor.preprocessing_steps,
            "message": "Missing values handled successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error handling missing values: {e}")


@router.post("/outliers/{file_id}")
async def handle_outliers(
    file_id: str,
    method: str = "iqr",
    columns: Optional[List[str]] = Query(None),
    threshold: float = 1.5,
    valid_classes: Optional[List[str]] = Query(None),
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db),
):
    try:
        verify_file_ownership(file_id, user_id, db)
        preprocessor = DataPreprocessor(file_id, db)
        preprocessor.load_data()

        processed_valid_classes = None
        if valid_classes is not None:
            processed_valid_classes = []
            for val in valid_classes:
                try:
                    processed_valid_classes.append(float(val) if "." in val else int(val))
                except ValueError:
                    processed_valid_classes.append(val)

        preprocessor.handle_outliers(
            method=method,
            columns=columns,
            threshold=threshold,
            valid_classes=processed_valid_classes,
        )
        return {
            "success": True,
            "file_id": file_id,
            "method": method,
            "threshold": threshold,
            "valid_classes": valid_classes,
            "columns_processed": columns or "all",
            "summary": preprocessor.get_data_summary(),
            "preprocessing_steps": preprocessor.preprocessing_steps,
            "message": "Outliers handled successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error handling outliers: {e}")


@router.post("/encode/{file_id}")
async def encode_categorical_variables(
    file_id: str,
    method: str = "label",
    columns: Optional[List[str]] = None,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db),
):
    try:
        verify_file_ownership(file_id, user_id, db)
        preprocessor = DataPreprocessor(file_id, db)
        preprocessor.load_data()
        preprocessor.encode_categorical_variables(columns=columns, method=method)
        return {
            "success": True,
            "file_id": file_id,
            "method": method,
            "columns_processed": columns or "all categorical",
            "summary": preprocessor.get_data_summary(),
            "preprocessing_steps": preprocessor.preprocessing_steps,
            "message": "Categorical variables encoded successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error encoding categorical variables: {e}")


@router.post("/scale/{file_id}")
async def scale_features(
    file_id: str,
    method: str = "standard",
    columns: Optional[List[str]] = None,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db),
):
    try:
        verify_file_ownership(file_id, user_id, db)
        preprocessor = DataPreprocessor(file_id, db)
        preprocessor.load_data()
        preprocessor.scale_features(columns=columns, method=method)
        return {
            "success": True,
            "file_id": file_id,
            "method": method,
            "columns_processed": columns or "all numeric",
            "summary": preprocessor.get_data_summary(),
            "preprocessing_steps": preprocessor.preprocessing_steps,
            "message": "Features scaled successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scaling features: {e}")


@router.delete("/duplicates/{file_id}")
async def remove_duplicates(
    file_id: str,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db),
):
    try:
        verify_file_ownership(file_id, user_id, db)
        preprocessor = DataPreprocessor(file_id, db)
        preprocessor.load_data()
        preprocessor.remove_duplicates()
        return {
            "success": True,
            "file_id": file_id,
            "summary": preprocessor.get_data_summary(),
            "preprocessing_steps": preprocessor.preprocessing_steps,
            "message": "Duplicate rows removed successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing duplicates: {e}")


@router.post("/save/{file_id}")
async def save_processed_data(
    file_id: str,
    filename: Optional[str] = None,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db),
):
    try:
        verify_file_ownership(file_id, user_id, db)
        preprocessor = DataPreprocessor(file_id, db)
        preprocessor.load_data()
        filepath = preprocessor.save_processed_data(filename=filename)
        return {
            "success": True,
            "file_id": file_id,
            "processed_filepath": filepath,
            "message": "Processed data saved successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving processed data: {e}")


@router.post("/reset/{file_id}")
async def reset_data(
    file_id: str,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db),
):
    try:
        verify_file_ownership(file_id, user_id, db)
        preprocessor = DataPreprocessor(file_id, db)
        preprocessor.load_data()
        preprocessor.reset_data()
        return {
            "success": True,
            "file_id": file_id,
            "summary": preprocessor.get_data_summary(),
            "message": "Data reset to original state successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting data: {e}")
