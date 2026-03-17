from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

from app.api.deps import get_db
from app.services.auth_service import verify_token
from app.services.data.load_csv import verify_file_ownership
from app.services.data.preprocess import DataPreprocessor, preprocess_data

router = APIRouter()

# Pydantic models for request/response
class PreprocessingConfig(BaseModel):
    """Configuration for data preprocessing operations."""
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
    """Response model for data summary."""
    success: bool
    file_id: str
    summary: Dict[str, Any]
    message: str

class PreprocessingResponse(BaseModel):
    """Response model for preprocessing operations."""
    success: bool
    file_id: str
    initial_summary: Dict[str, Any]
    final_summary: Dict[str, Any]
    preprocessing_steps: List[str]
    processed_filepath: Optional[str]
    message: str

@router.get("/summary/{file_id}", response_model=DataSummaryResponse)
async def get_data_summary(
    file_id: str,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Get a comprehensive summary of the uploaded CSV file.
    
    Args:
        file_id (str): The unique ID of the file in the database
        user_id (int): The ID of the authenticated user
        db (Session): The database session
        
    Returns:
        DataSummaryResponse: Summary information about the dataset
    """
    try:
        verify_file_ownership(file_id, user_id, db)
        # Initialize preprocessor and load data
        preprocessor = DataPreprocessor(file_id, db)
        preprocessor.load_data()

        # Get data summary
        summary = preprocessor.get_data_summary()
        
        return DataSummaryResponse(
            success=True,
            file_id=file_id,
            summary=summary,
            message="Data summary retrieved successfully"
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting data summary: {str(e)}")

@router.post("/preprocess/{file_id}", response_model=PreprocessingResponse)
async def preprocess_csv_data(
    file_id: str,
    config: PreprocessingConfig,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Preprocess CSV data based on the provided configuration.
    
    Args:
        file_id (str): The unique ID of the file in the database
        config (PreprocessingConfig): Configuration for preprocessing steps
        user_id (int): The ID of the authenticated user
        db (Session): The database session
        
    Returns:
        PreprocessingResponse: Results of preprocessing operations
    """
    try:
        verify_file_ownership(file_id, user_id, db)
        # Convert Pydantic model to dict
        preprocessing_config = config.dict()

        # Perform preprocessing
        result = preprocess_data(file_id, db, preprocessing_config)
        
        return PreprocessingResponse(**result)
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in data preprocessing: {str(e)}")

@router.post("/missing-values/{file_id}")
async def handle_missing_values(
    file_id: str,
    strategy: str = "remove_column",
    missing_threshold: float = 0.8,
    columns: Optional[List[str]] = None,
    custom_value: Optional[str] = None,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Handle missing values in the dataset with three specific strategies.
    
    Args:
        file_id (str): The unique ID of the file in the database
        strategy (str): Strategy for handling missing values
                       - 'remove_column': Remove columns with missing ratio >= threshold
                       - 'remove_rows': Remove rows with missing values for specified columns
                       - 'fill_custom': Fill missing values with custom value
        missing_threshold (float): Missing ratio threshold (0.0 to 1.0)
        columns (List[str], optional): Specific columns to process
        custom_value (str, optional): Custom value to fill missing values (required for 'fill_custom')
        user_id (int): The ID of the authenticated user
        db (Session): The database session
        
    Returns:
        Dict: Results of missing value handling
    """
    try:
        verify_file_ownership(file_id, user_id, db)
        preprocessor = DataPreprocessor(file_id, db)
        preprocessor.load_data()

        # Handle missing values
        preprocessor.handle_missing_values(
            strategy=strategy, 
            missing_threshold=missing_threshold,
            columns=columns,
            custom_value=custom_value
        )
        
        # Get updated summary
        summary = preprocessor.get_data_summary()
        
        return {
            "success": True,
            "file_id": file_id,
            "strategy": strategy,
            "missing_threshold": missing_threshold,
            "custom_value": custom_value,
            "columns_processed": columns or "all",
            "summary": summary,
            "preprocessing_steps": preprocessor.preprocessing_steps,
            "message": "Missing values handled successfully"
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error handling missing values: {str(e)}")

@router.post("/outliers/{file_id}")
async def handle_outliers(
    file_id: str,
    method: str = "iqr",
    columns: Optional[List[str]] = Query(None),
    threshold: float = 1.5,
    valid_classes: Optional[List[str]] = Query(None),
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Handle outliers in both numeric and categorical columns.
    
    Args:
        file_id (str): The unique ID of the file in the database
        method (str): Method for outlier detection
                     For numeric: 'iqr', 'zscore', 'clip'
                     For categorical: 'valid_classes'
        columns (List[str], optional): Specific columns to process
        threshold (float): Threshold for numeric outlier detection
        valid_classes (List[str], optional): Valid class values for categorical outlier detection
        user_id (int): The ID of the authenticated user
        db (Session): The database session
        
    Returns:
        Dict: Results of outlier handling
    """
    try:
        verify_file_ownership(file_id, user_id, db)
        preprocessor = DataPreprocessor(file_id, db)
        preprocessor.load_data()

        # Convert valid_classes from strings to appropriate types if needed
        processed_valid_classes = None
        if valid_classes is not None:
            processed_valid_classes = []
            for val in valid_classes:
                # Try to convert to int/float if possible, otherwise keep as string
                try:
                    if '.' in val:
                        processed_valid_classes.append(float(val))
                    else:
                        processed_valid_classes.append(int(val))
                except ValueError:
                    processed_valid_classes.append(val)
        
        # Handle outliers
        preprocessor.handle_outliers(
            method=method, 
            columns=columns, 
            threshold=threshold,
            valid_classes=processed_valid_classes
        )
        
        # Get updated summary
        summary = preprocessor.get_data_summary()
        
        return {
            "success": True,
            "file_id": file_id,
            "method": method,
            "threshold": threshold,
            "valid_classes": valid_classes,
            "columns_processed": columns or "all",
            "summary": summary,
            "preprocessing_steps": preprocessor.preprocessing_steps,
            "message": "Outliers handled successfully"
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error handling outliers: {str(e)}")

@router.post("/encode/{file_id}")
async def encode_categorical_variables(
    file_id: str,
    method: str = "label",
    columns: Optional[List[str]] = None,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Encode categorical variables.
    
    Args:
        file_id (str): The unique ID of the file in the database
        method (str): Encoding method ('label' or 'onehot')
        columns (List[str], optional): Specific columns to encode
        user_id (int): The ID of the authenticated user
        db (Session): The database session
        
    Returns:
        Dict: Results of categorical encoding
    """
    try:
        verify_file_ownership(file_id, user_id, db)
        preprocessor = DataPreprocessor(file_id, db)
        preprocessor.load_data()

        # Encode categorical variables
        preprocessor.encode_categorical_variables(columns=columns, method=method)
        
        # Get updated summary
        summary = preprocessor.get_data_summary()
        
        return {
            "success": True,
            "file_id": file_id,
            "method": method,
            "columns_processed": columns or "all categorical",
            "summary": summary,
            "preprocessing_steps": preprocessor.preprocessing_steps,
            "message": "Categorical variables encoded successfully"
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error encoding categorical variables: {str(e)}")

@router.post("/scale/{file_id}")
async def scale_features(
    file_id: str,
    method: str = "standard",
    columns: Optional[List[str]] = None,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Scale numeric features.
    
    Args:
        file_id (str): The unique ID of the file in the database
        method (str): Scaling method ('standard' or 'minmax')
        columns (List[str], optional): Specific columns to scale
        user_id (int): The ID of the authenticated user
        db (Session): The database session
        
    Returns:
        Dict: Results of feature scaling
    """
    try:
        verify_file_ownership(file_id, user_id, db)
        preprocessor = DataPreprocessor(file_id, db)
        preprocessor.load_data()

        # Scale features
        preprocessor.scale_features(columns=columns, method=method)
        
        # Get updated summary
        summary = preprocessor.get_data_summary()
        
        return {
            "success": True,
            "file_id": file_id,
            "method": method,
            "columns_processed": columns or "all numeric",
            "summary": summary,
            "preprocessing_steps": preprocessor.preprocessing_steps,
            "message": "Features scaled successfully"
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scaling features: {str(e)}")

@router.delete("/duplicates/{file_id}")
async def remove_duplicates(
    file_id: str,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Remove duplicate rows from the dataset.
    
    Args:
        file_id (str): The unique ID of the file in the database
        user_id (int): The ID of the authenticated user
        db (Session): The database session
        
    Returns:
        Dict: Results of duplicate removal
    """
    try:
        verify_file_ownership(file_id, user_id, db)
        preprocessor = DataPreprocessor(file_id, db)
        preprocessor.load_data()

        # Remove duplicates
        preprocessor.remove_duplicates()
        
        # Get updated summary
        summary = preprocessor.get_data_summary()
        
        return {
            "success": True,
            "file_id": file_id,
            "summary": summary,
            "preprocessing_steps": preprocessor.preprocessing_steps,
            "message": "Duplicate rows removed successfully"
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing duplicates: {str(e)}")

@router.post("/save/{file_id}")
async def save_processed_data(
    file_id: str,
    filename: Optional[str] = None,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Save the processed data to a new CSV file.
    
    Args:
        file_id (str): The unique ID of the file in the database
        filename (str, optional): Custom filename for the processed data
        user_id (int): The ID of the authenticated user
        db (Session): The database session
        
    Returns:
        Dict: Results of saving processed data
    """
    try:
        verify_file_ownership(file_id, user_id, db)
        preprocessor = DataPreprocessor(file_id, db)
        preprocessor.load_data()

        # Save processed data
        filepath = preprocessor.save_processed_data(filename=filename)
        
        return {
            "success": True,
            "file_id": file_id,
            "processed_filepath": filepath,
            "message": "Processed data saved successfully"
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving processed data: {str(e)}")

@router.post("/reset/{file_id}")
async def reset_data(
    file_id: str,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Reset the data to its original state.
    
    Args:
        file_id (str): The unique ID of the file in the database
        user_id (int): The ID of the authenticated user
        db (Session): The database session
        
    Returns:
        Dict: Results of data reset
    """
    try:
        verify_file_ownership(file_id, user_id, db)
        preprocessor = DataPreprocessor(file_id, db)
        preprocessor.load_data()

        # Reset data
        preprocessor.reset_data()
        
        # Get original summary
        summary = preprocessor.get_data_summary()
        
        return {
            "success": True,
            "file_id": file_id,
            "summary": summary,
            "message": "Data reset to original state successfully"
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting data: {str(e)}")
