import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from fastapi import HTTPException
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


class BaseHandler(ABC):
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        ...


class MissingValueHandler(BaseHandler):
    def __init__(
        self,
        strategy: str = "remove_column",
        missing_threshold: float = 0.8,
        columns: Optional[List[str]] = None,
        custom_value: Optional[Any] = None,
    ):
        self.strategy = strategy
        self.missing_threshold = missing_threshold
        self.columns = columns
        self.custom_value = custom_value

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        steps: list[str] = []
        cols = self.columns if self.columns is not None else df.columns.tolist()
        original_missing = df.isnull().sum().sum()
        original_shape = df.shape

        if self.strategy == "remove_column":
            to_remove = [
                c for c in cols
                if c in df.columns and df[c].isnull().sum() / len(df) >= self.missing_threshold
            ]
            if to_remove:
                df = df.drop(columns=to_remove)
                steps.append(f"Removed {len(to_remove)} columns with missing ratio >= {self.missing_threshold}: {to_remove}")
            else:
                steps.append(f"No columns found with missing ratio >= {self.missing_threshold}")

        elif self.strategy == "remove_rows":
            to_check = [
                c for c in cols
                if c in df.columns and df[c].isnull().sum() / len(df) >= self.missing_threshold
            ]
            if to_check:
                original_rows = len(df)
                df = df.dropna(subset=to_check)
                steps.append(
                    f"Removed {original_rows - len(df)} rows with missing values in columns "
                    f"with missing ratio >= {self.missing_threshold}: {to_check}"
                )
            else:
                steps.append(f"No columns found with missing ratio >= {self.missing_threshold} for row removal")

        elif self.strategy == "fill_custom":
            if self.custom_value is None:
                raise HTTPException(status_code=400, detail="custom_value is required for 'fill_custom' strategy")
            filled = []
            for c in cols:
                if c not in df.columns:
                    continue
                if df[c].isnull().sum() / len(df) >= self.missing_threshold:
                    missing_count = int(df[c].isnull().sum())
                    if missing_count > 0:
                        df[c] = df[c].fillna(self.custom_value)
                        filled.append(f"{c} ({missing_count} values)")
            if filled:
                steps.append(f"Filled missing values with '{self.custom_value}' in: {filled}")
            else:
                steps.append(f"No columns found with missing ratio >= {self.missing_threshold} for custom value filling")

        elif self.strategy in ("mean", "median", "mode"):
            filled = []
            for c in cols:
                if c not in df.columns:
                    continue
                missing_count = int(df[c].isnull().sum())
                if missing_count == 0:
                    continue
                fill_val = None
                if self.strategy == "mean" and pd.api.types.is_numeric_dtype(df[c]):
                    fill_val = df[c].mean()
                elif self.strategy == "median" and pd.api.types.is_numeric_dtype(df[c]):
                    fill_val = df[c].median()
                elif self.strategy == "mode":
                    mode_vals = df[c].mode()
                    fill_val = mode_vals.iloc[0] if not mode_vals.empty else None
                if fill_val is not None:
                    df[c] = df[c].fillna(fill_val)
                    filled.append(f"{c} ({missing_count} values → {fill_val})")
            steps.append(f"Filled missing values with {self.strategy} in columns: {filled or 'none needed'}")

        elif self.strategy == "drop":
            original_rows = len(df)
            df = df.dropna(subset=[c for c in cols if c in df.columns])
            steps.append(f"Dropped {original_rows - len(df)} rows with missing values")

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy '{self.strategy}'. Use 'remove_column', 'remove_rows', "
                       f"'fill_custom', 'mean', 'median', 'mode', or 'drop'",
            )

        steps.append(
            f"Missing values reduced from {original_missing} to {df.isnull().sum().sum()}. "
            f"Shape changed from {original_shape} to {df.shape}"
        )
        return df, steps


class OutlierHandler(BaseHandler):
    def __init__(
        self,
        method: str = "iqr",
        columns: Optional[List[str]] = None,
        threshold: float = 1.5,
        valid_classes: Optional[List] = None,
    ):
        self.method = method
        self.columns = columns
        self.threshold = threshold
        self.valid_classes = valid_classes

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        steps: list[str] = []
        cols = self.columns if self.columns is not None else df.columns.tolist()
        total_removed = 0

        for col in cols:
            if col not in df.columns:
                continue
            count_before = len(df)

            if self.method == "valid_classes" and self.valid_classes is not None:
                if pd.api.types.is_object_dtype(df[col]) or str(df[col].dtype) == "category":
                    mask = ~df[col].isin(self.valid_classes)
                    if mask.sum() > 0:
                        invalid = df[col][mask].unique()
                        df = df[~mask]
                        steps.append(f"Removed {mask.sum()} categorical outliers from '{col}': {list(invalid)}")
                elif pd.api.types.is_numeric_dtype(df[col]):
                    mask = ~df[col].isin(self.valid_classes)
                    if mask.sum() > 0:
                        invalid = df[col][mask].unique()
                        df = df[~mask]
                        steps.append(f"Removed {mask.sum()} invalid class values from '{col}': {list(invalid)}")

            elif pd.api.types.is_numeric_dtype(df[col]):
                if self.method == "iqr":
                    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                    lo, hi = Q1 - self.threshold * (Q3 - Q1), Q3 + self.threshold * (Q3 - Q1)
                    mask = (df[col] < lo) | (df[col] > hi)
                    if mask.sum() > 0:
                        df = df[~mask]
                        steps.append(f"Removed {mask.sum()} outliers from '{col}' using IQR (bounds: {lo:.2f} to {hi:.2f})")

                elif self.method == "zscore":
                    z = np.abs((df[col] - df[col].mean()) / df[col].std())
                    mask = z > self.threshold
                    if mask.sum() > 0:
                        df = df[~mask]
                        steps.append(f"Removed {mask.sum()} outliers from '{col}' using Z-score (threshold: {self.threshold})")

                elif self.method == "clip":
                    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                    lo, hi = Q1 - self.threshold * (Q3 - Q1), Q3 + self.threshold * (Q3 - Q1)
                    original = df[col].copy()
                    df[col] = df[col].clip(lower=lo, upper=hi)
                    clipped = int((original != df[col]).sum())
                    if clipped > 0:
                        steps.append(f"Clipped {clipped} values in '{col}' to bounds: {lo:.2f} to {hi:.2f}")

            total_removed += count_before - len(df)

        if total_removed > 0:
            steps.append(f"Total outliers removed: {total_removed} rows")
        return df, steps


class EncoderHandler(BaseHandler):
    def __init__(self, method: str = "label", columns: Optional[List[str]] = None):
        self.method = method
        self.columns = columns

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        steps: list[str] = []
        cols = self.columns if self.columns is not None else [
            c for c in df.columns if pd.api.types.is_object_dtype(df[c])
        ]
        for col in cols:
            if col not in df.columns or not pd.api.types.is_object_dtype(df[col]):
                continue
            if self.method == "label":
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
                steps.append(f"Label encoded column: {col}")
            elif self.method == "onehot":
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
                steps.append(f"One-hot encoded column: {col}")
        return df, steps


class ScalerHandler(BaseHandler):
    def __init__(self, method: str = "standard", columns: Optional[List[str]] = None):
        self.method = method
        self.columns = columns

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        steps: list[str] = []
        cols = self.columns if self.columns is not None else [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ]
        scaler_cls = StandardScaler if self.method == "standard" else MinMaxScaler
        label = "Standard" if self.method == "standard" else "MinMax"
        for col in cols:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            df[col] = scaler_cls().fit_transform(df[[col]])
            steps.append(f"{label} scaled column: {col}")
        return df, steps


class DuplicateRemover(BaseHandler):
    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        original = len(df)
        df = df.drop_duplicates()
        removed = original - len(df)
        return df, ([f"Removed {removed} duplicate rows"] if removed > 0 else [])


class PreprocessingPipeline:
    def __init__(self) -> None:
        self._steps: list[BaseHandler] = []

    def add(self, handler: BaseHandler) -> "PreprocessingPipeline":
        self._steps.append(handler)
        return self

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        all_steps: list[str] = []
        for handler in self._steps:
            df, steps = handler.apply(df)
            all_steps.extend(steps)
        return df, all_steps
