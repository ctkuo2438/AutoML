import warnings
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.services.data.handlers import (
    DuplicateRemover,
    EncoderHandler,
    MissingValueHandler,
    OutlierHandler,
    PreprocessingPipeline,
    ScalerHandler,
)
from app.services.data.load_csv import load_csv

warnings.filterwarnings("ignore")


class DataPreprocessor:
    def __init__(self, file_id: str, db: Session):
        self.file_id = file_id
        self.db = db
        self.df: Optional[pd.DataFrame] = None
        self.original_df: Optional[pd.DataFrame] = None
        self.preprocessing_steps: list[str] = []
        self.column_info: dict = {}

    # ------------------------------------------------------------------
    # Data loading / introspection
    # ------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        try:
            self.df = load_csv(self.file_id, self.db)
            self.original_df = self.df.copy()
            self._analyze_columns()
            return self.df
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

    def _analyze_columns(self) -> None:
        if self.df is None:
            return
        self.column_info = {}
        for column in self.df.columns:
            col_info: dict[str, Any] = {
                "dtype": str(self.df[column].dtype),
                "null_count": int(self.df[column].isnull().sum()),
                "null_percentage": float(self.df[column].isnull().sum() / len(self.df) * 100),
                "unique_count": int(self.df[column].nunique()),
                "is_numeric": pd.api.types.is_numeric_dtype(self.df[column]),
                "is_categorical": pd.api.types.is_object_dtype(self.df[column]),
            }
            if col_info["is_numeric"] and not self.df[column].isnull().all():
                col_info.update({
                    "mean": float(self.df[column].mean()),
                    "std": float(self.df[column].std()),
                    "min": float(self.df[column].min()),
                    "max": float(self.df[column].max()),
                    "median": float(self.df[column].median()),
                })
            self.column_info[column] = col_info

    def get_data_summary(self) -> Dict[str, Any]:
        if self.df is None:
            raise HTTPException(status_code=400, detail="Data not loaded. Call load_data() first.")
        return {
            "shape": {"rows": int(self.df.shape[0]), "columns": int(self.df.shape[1])},
            "columns": list(self.df.columns),
            "column_info": self.column_info,
            "missing_values_total": int(self.df.isnull().sum().sum()),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "memory_usage": f"{self.df.memory_usage(deep=True).sum() / 1024:.2f} KB",
            "preprocessing_steps": self.preprocessing_steps,
        }

    def get_processed_data(self) -> pd.DataFrame:
        if self.df is None:
            raise HTTPException(status_code=400, detail="Data not loaded. Call load_data() first.")
        return self.df

    def save_processed_data(self, filename: Optional[str] = None) -> str:
        if self.df is None:
            raise HTTPException(status_code=400, detail="Data not loaded. Call load_data() first.")
        filename = filename or f"processed_{self.file_id}.csv"
        filepath = f"uploads/{filename}"
        try:
            self.df.to_csv(filepath, index=False)
            self.preprocessing_steps.append(f"Saved processed data to: {filepath}")
            return filepath
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving processed data: {str(e)}")

    def reset_data(self) -> pd.DataFrame:
        if self.original_df is None:
            raise HTTPException(status_code=400, detail="Original data not available.")
        self.df = self.original_df.copy()
        self.preprocessing_steps = []
        self._analyze_columns()
        return self.df

    # ------------------------------------------------------------------
    # Strategy delegation — each method creates a handler and runs it
    # ------------------------------------------------------------------

    def _run(self, handler) -> pd.DataFrame:
        """Run a single handler, extend the step log, refresh column info."""
        assert self.df is not None
        self.df, steps = handler.apply(self.df)
        self.preprocessing_steps.extend(steps)
        self._analyze_columns()
        return self.df

    def handle_missing_values(
        self,
        strategy: str = "remove_column",
        missing_threshold: float = 0.8,
        columns: Optional[List[str]] = None,
        custom_value: Optional[Any] = None,
    ) -> pd.DataFrame:
        if self.df is None:
            raise HTTPException(status_code=400, detail="Data not loaded. Call load_data() first.")
        return self._run(MissingValueHandler(strategy, missing_threshold, columns, custom_value))

    def handle_outliers(
        self,
        method: str = "iqr",
        columns: Optional[List[str]] = None,
        threshold: float = 1.5,
        valid_classes: Optional[List] = None,
    ) -> pd.DataFrame:
        if self.df is None:
            raise HTTPException(status_code=400, detail="Data not loaded. Call load_data() first.")
        return self._run(OutlierHandler(method, columns, threshold, valid_classes))

    def encode_categorical_variables(
        self,
        columns: Optional[List[str]] = None,
        method: str = "label",
    ) -> pd.DataFrame:
        if self.df is None:
            raise HTTPException(status_code=400, detail="Data not loaded. Call load_data() first.")
        return self._run(EncoderHandler(method, columns))

    def scale_features(
        self,
        columns: Optional[List[str]] = None,
        method: str = "standard",
    ) -> pd.DataFrame:
        if self.df is None:
            raise HTTPException(status_code=400, detail="Data not loaded. Call load_data() first.")
        return self._run(ScalerHandler(method, columns))

    def remove_duplicates(self) -> pd.DataFrame:
        if self.df is None:
            raise HTTPException(status_code=400, detail="Data not loaded. Call load_data() first.")
        return self._run(DuplicateRemover())


def preprocess_data(file_id: str, db: Session, preprocessing_config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        preprocessor = DataPreprocessor(file_id, db)
        preprocessor.load_data()
        initial_summary = preprocessor.get_data_summary()

        pipeline = PreprocessingPipeline()

        if preprocessing_config.get("handle_missing"):
            cfg = preprocessing_config.get("missing_config", {})
            # Drop specific user-selected columns before the missing-value pass
            drop_columns = [c for c in (cfg.get("drop_columns") or []) if c in preprocessor.df.columns]
            if drop_columns:
                preprocessor.df = preprocessor.df.drop(columns=drop_columns)
                preprocessor.preprocessing_steps.append(f"Dropped columns: {drop_columns}")
                preprocessor._analyze_columns()
            pipeline.add(MissingValueHandler(
                strategy=cfg.get("strategy", "remove_column"),
                missing_threshold=cfg.get("missing_threshold", 0.8),
                columns=cfg.get("columns"),
                custom_value=cfg.get("custom_value"),
            ))

        if preprocessing_config.get("handle_outliers"):
            cfg = preprocessing_config.get("outlier_config", {})
            pipeline.add(OutlierHandler(
                method=cfg.get("method", "iqr"),
                columns=cfg.get("columns"),
                threshold=cfg.get("threshold", 1.5),
            ))

        if preprocessing_config.get("encode_categorical"):
            cfg = preprocessing_config.get("encoding_config", {})
            pipeline.add(EncoderHandler(
                method=cfg.get("method", "label"),
                columns=cfg.get("columns"),
            ))

        if preprocessing_config.get("scale_features"):
            cfg = preprocessing_config.get("scaling_config", {})
            pipeline.add(ScalerHandler(
                method=cfg.get("method", "standard"),
                columns=cfg.get("columns"),
            ))

        if preprocessing_config.get("remove_duplicates"):
            pipeline.add(DuplicateRemover())

        preprocessor.df, pipeline_steps = pipeline.run(preprocessor.df)
        preprocessor.preprocessing_steps.extend(pipeline_steps)
        preprocessor._analyze_columns()

        # Persist preprocessed data back to the original file so training sees it
        from app.db.models.file_model import File
        db_file = db.query(File).filter(File.id == file_id).first()
        if db_file:
            preprocessor.df.to_csv(str(db_file.filepath), index=False)
            preprocessor.preprocessing_steps.append(f"Saved preprocessed data to: {db_file.filepath}")

        return {
            "success": True,
            "file_id": file_id,
            "initial_summary": initial_summary,
            "final_summary": preprocessor.get_data_summary(),
            "preprocessing_steps": preprocessor.preprocessing_steps,
            "processed_filepath": str(db_file.filepath) if db_file else None,
            "message": "Data preprocessing completed successfully",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in data preprocessing: {str(e)}")
