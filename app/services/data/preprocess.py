import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from fastapi import HTTPException
from sqlalchemy.orm import Session
from app.services.data.load_csv import load_csv
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    A comprehensive data preprocessing class for handling CSV files.
    Provides methods for handling missing values, outliers, encoding, scaling, and data analysis.
    """
    
    def __init__(self, file_id: str, db: Session):
        """
        Initialize the DataPreprocessor with a file ID and database session.
        
        Args:
            file_id (str): The unique ID of the file in the database
            db (Session): The database session
        """
        self.file_id = file_id
        self.db = db
        self.df = None
        self.original_df = None
        self.preprocessing_steps = []
        self.column_info = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the CSV file data using the file ID.
        
        Returns:
            pd.DataFrame: The loaded DataFrame
            
        Raises:
            HTTPException: If file loading fails
        """
        try:
            self.df = load_csv(self.file_id, self.db)
            self.original_df = self.df.copy()  # Keep original for comparison
            self._analyze_columns()
            return self.df
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")
    
    def _analyze_columns(self) -> None:
        """Analyze column types and characteristics."""
        if self.df is None:
            return
            
        for column in self.df.columns:
            col_info = {
                'dtype': str(self.df[column].dtype),
                'null_count': int(self.df[column].isnull().sum()),
                'null_percentage': float(self.df[column].isnull().sum() / len(self.df) * 100),
                'unique_count': int(self.df[column].nunique()),
                'is_numeric': pd.api.types.is_numeric_dtype(self.df[column]),
                'is_categorical': pd.api.types.is_object_dtype(self.df[column])
            }
            
            if col_info['is_numeric']:
                col_info.update({
                    'mean': float(self.df[column].mean()) if not self.df[column].isnull().all() else None,
                    'std': float(self.df[column].std()) if not self.df[column].isnull().all() else None,
                    'min': float(self.df[column].min()) if not self.df[column].isnull().all() else None,
                    'max': float(self.df[column].max()) if not self.df[column].isnull().all() else None,
                    'median': float(self.df[column].median()) if not self.df[column].isnull().all() else None
                })
            
            self.column_info[column] = col_info
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the dataset.
        
        Returns:
            Dict: Summary information about the dataset
        """
        if self.df is None:
            raise HTTPException(status_code=400, detail="Data not loaded. Call load_data() first.")
        
        return {
            'shape': {
                'rows': int(self.df.shape[0]),
                'columns': int(self.df.shape[1])
            },
            'columns': list(self.df.columns),
            'column_info': self.column_info,
            'missing_values_total': int(self.df.isnull().sum().sum()),
            'duplicate_rows': int(self.df.duplicated().sum()),
            'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024:.2f} KB",
            'preprocessing_steps': self.preprocessing_steps
        }
    
    def handle_missing_values(self, strategy: str = 'remove_column', 
                            missing_threshold: float = 0.8, 
                            columns: Optional[List[str]] = None,
                            custom_value: Optional[Any] = None) -> pd.DataFrame:
        """
        Handle missing values in the dataset with three specific strategies.
        
        Args:
            strategy (str): Strategy for handling missing values
                          - 'remove_column': Remove columns with missing ratio >= threshold
                          - 'remove_rows': Remove rows with missing values for specified columns
                          - 'fill_custom': Fill missing values with custom value
            missing_threshold (float): Missing ratio threshold (0.0 to 1.0). 
                                     For 'remove_column': columns with missing ratio >= threshold will be removed
                                     For 'remove_rows': only applies to columns with missing ratio >= threshold
            columns (List[str], optional): Specific columns to process (if None, process all columns)
            custom_value (Any, optional): Custom value to fill missing values (required for 'fill_custom' strategy)
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        if self.df is None:
            raise HTTPException(status_code=400, detail="Data not loaded. Call load_data() first.")
        
        if columns is None:
            columns = self.df.columns.tolist()
        
        original_missing = self.df.isnull().sum().sum()
        original_shape = self.df.shape
        
        if strategy == 'remove_column':
            # Remove columns with missing ratio >= threshold
            columns_to_remove = []
            for col in columns:
                if col not in self.df.columns:
                    continue
                
                missing_ratio = self.df[col].isnull().sum() / len(self.df)
                if missing_ratio >= missing_threshold:
                    columns_to_remove.append(col)
            
            if columns_to_remove:
                self.df = self.df.drop(columns=columns_to_remove)
                self.preprocessing_steps.append(
                    f"Removed {len(columns_to_remove)} columns with missing ratio >= {missing_threshold}: {columns_to_remove}"
                )
            else:
                self.preprocessing_steps.append(
                    f"No columns found with missing ratio >= {missing_threshold}"
                )
                
        elif strategy == 'remove_rows':
            # Remove rows with missing values for columns that meet the threshold
            columns_to_check = []
            for col in columns:
                if col not in self.df.columns:
                    continue
                
                missing_ratio = self.df[col].isnull().sum() / len(self.df)
                if missing_ratio >= missing_threshold:
                    columns_to_check.append(col)
            
            if columns_to_check:
                original_rows = len(self.df)
                self.df = self.df.dropna(subset=columns_to_check)
                rows_removed = original_rows - len(self.df)
                self.preprocessing_steps.append(
                    f"Removed {rows_removed} rows with missing values in columns with missing ratio >= {missing_threshold}: {columns_to_check}"
                )
            else:
                self.preprocessing_steps.append(
                    f"No columns found with missing ratio >= {missing_threshold} for row removal"
                )
                
        elif strategy == 'fill_custom':
            if custom_value is None:
                raise HTTPException(status_code=400, detail="custom_value is required for 'fill_custom' strategy")
            
            # Fill missing values with custom value for columns that meet the threshold
            columns_filled = []
            for col in columns:
                if col not in self.df.columns:
                    continue
                
                missing_ratio = self.df[col].isnull().sum() / len(self.df)
                if missing_ratio >= missing_threshold:
                    missing_count = self.df[col].isnull().sum()
                    if missing_count > 0:
                        self.df[col].fillna(custom_value, inplace=True)
                        columns_filled.append(f"{col} ({missing_count} values)")
            
            if columns_filled:
                self.preprocessing_steps.append(
                    f"Filled missing values with '{custom_value}' in columns with missing ratio >= {missing_threshold}: {columns_filled}"
                )
            else:
                self.preprocessing_steps.append(
                    f"No columns found with missing ratio >= {missing_threshold} for custom value filling"
                )
        
        else:
            raise HTTPException(status_code=400, detail=f"Invalid strategy '{strategy}'. Use 'remove_column', 'remove_rows', or 'fill_custom'")
        
        new_missing = self.df.isnull().sum().sum()
        new_shape = self.df.shape
        
        self.preprocessing_steps.append(
            f"Missing values reduced from {original_missing} to {new_missing}. "
            f"Shape changed from {original_shape} to {new_shape}"
        )
        
        # Update column info
        self._analyze_columns()
        
        return self.df
    
    def handle_outliers(self, method: str = 'iqr', columns: Optional[List[str]] = None, 
                       threshold: float = 1.5, valid_classes: Optional[List] = None) -> pd.DataFrame:
        """
        Handle outliers in both numeric and categorical columns.
        
        Args:
            method (str): Method for outlier detection
                         For numeric columns:
                         - 'iqr': Interquartile Range method
                         - 'zscore': Z-score method  
                         - 'clip': Clip outliers to threshold values
                         For categorical columns:
                         - 'valid_classes': Remove rows with classes not in valid_classes list
            columns (List[str], optional): Specific columns to process
            threshold (float): Threshold for numeric outlier detection
            valid_classes (List, optional): List of valid class values for categorical outlier detection
            
        Returns:
            pd.DataFrame: DataFrame with outliers handled
        """
        if self.df is None:
            raise HTTPException(status_code=400, detail="Data not loaded. Call load_data() first.")
        
        if columns is None:
            # Process all columns by default
            columns = self.df.columns.tolist()
        
        outliers_removed = 0
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            original_count = len(self.df)
            
            # Handle categorical outliers (e.g., invalid class labels)
            if method == 'valid_classes' and valid_classes is not None:
                if pd.api.types.is_object_dtype(self.df[col]) or str(self.df[col].dtype) == 'category':
                    # For categorical columns, remove rows with invalid classes
                    outlier_mask = ~self.df[col].isin(valid_classes)
                    outlier_count = outlier_mask.sum()
                    if outlier_count > 0:
                        invalid_values = self.df[col][outlier_mask].unique()
                        self.df = self.df[~outlier_mask]
                        self.preprocessing_steps.append(
                            f"Removed {outlier_count} categorical outliers from '{col}': {list(invalid_values)}"
                        )
                elif pd.api.types.is_numeric_dtype(self.df[col]):
                    # For numeric columns that represent classes, also check valid classes
                    outlier_mask = ~self.df[col].isin(valid_classes)
                    outlier_count = outlier_mask.sum()
                    if outlier_count > 0:
                        invalid_values = self.df[col][outlier_mask].unique()
                        self.df = self.df[~outlier_mask]
                        self.preprocessing_steps.append(
                            f"Removed {outlier_count} invalid class values from '{col}': {list(invalid_values)}"
                        )
            
            # Handle numeric outliers
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                if method == 'iqr':
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                    outlier_count = outlier_mask.sum()
                    if outlier_count > 0:
                        self.df = self.df[~outlier_mask]
                        self.preprocessing_steps.append(
                            f"Removed {outlier_count} numeric outliers from '{col}' using IQR method (bounds: {lower_bound:.2f} to {upper_bound:.2f})"
                        )
                    
                elif method == 'zscore':
                    z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                    outlier_mask = z_scores > threshold
                    outlier_count = outlier_mask.sum()
                    if outlier_count > 0:
                        self.df = self.df[~outlier_mask]
                        self.preprocessing_steps.append(
                            f"Removed {outlier_count} numeric outliers from '{col}' using Z-score method (threshold: {threshold})"
                        )
                    
                elif method == 'clip':
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    original_values = self.df[col].copy()
                    self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
                    clipped_count = (original_values != self.df[col]).sum()
                    if clipped_count > 0:
                        self.preprocessing_steps.append(
                            f"Clipped {clipped_count} values in '{col}' to bounds: {lower_bound:.2f} to {upper_bound:.2f}"
                        )
            
            new_count = len(self.df)
            outliers_removed += (original_count - new_count)
        
        if outliers_removed > 0:
            self.preprocessing_steps.append(f"Total outliers removed: {outliers_removed} rows")
        
        # Update column info
        self._analyze_columns()
        
        return self.df
    
    def encode_categorical_variables(self, columns: Optional[List[str]] = None, 
                                   method: str = 'label') -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            columns (List[str], optional): Specific columns to encode
            method (str): Encoding method ('label' or 'onehot')
            
        Returns:
            pd.DataFrame: DataFrame with encoded categorical variables
        """
        if self.df is None:
            raise HTTPException(status_code=400, detail="Data not loaded. Call load_data() first.")
        
        if columns is None:
            columns = [col for col in self.df.columns if pd.api.types.is_object_dtype(self.df[col])]
        
        for col in columns:
            if col not in self.df.columns or not pd.api.types.is_object_dtype(self.df[col]):
                continue
            
            if method == 'label':
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.preprocessing_steps.append(f"Label encoded column: {col}")
                
            elif method == 'onehot':
                # Create dummy variables
                dummies = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df.drop(col, axis=1), dummies], axis=1)
                self.preprocessing_steps.append(f"One-hot encoded column: {col}")
        
        # Update column info
        self._analyze_columns()
        
        return self.df
    
    def scale_features(self, columns: Optional[List[str]] = None, 
                      method: str = 'standard') -> pd.DataFrame:
        """
        Scale numeric features.
        
        Args:
            columns (List[str], optional): Specific columns to scale
            method (str): Scaling method ('standard' or 'minmax')
            
        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        if self.df is None:
            raise HTTPException(status_code=400, detail="Data not loaded. Call load_data() first.")
        
        if columns is None:
            columns = [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]
        
        for col in columns:
            if col not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            
            if method == 'standard':
                scaler = StandardScaler()
                self.df[col] = scaler.fit_transform(self.df[[col]])
                self.preprocessing_steps.append(f"Standard scaled column: {col}")
                
            elif method == 'minmax':
                scaler = MinMaxScaler()
                self.df[col] = scaler.fit_transform(self.df[[col]])
                self.preprocessing_steps.append(f"MinMax scaled column: {col}")
        
        # Update column info
        self._analyze_columns()
        
        return self.df
    
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Returns:
            pd.DataFrame: DataFrame with duplicates removed
        """
        if self.df is None:
            raise HTTPException(status_code=400, detail="Data not loaded. Call load_data() first.")
        
        original_count = len(self.df)
        self.df = self.df.drop_duplicates()
        new_count = len(self.df)
        
        duplicates_removed = original_count - new_count
        if duplicates_removed > 0:
            self.preprocessing_steps.append(f"Removed {duplicates_removed} duplicate rows")
        
        # Update column info
        self._analyze_columns()
        
        return self.df
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Get the current processed DataFrame.
        
        Returns:
            pd.DataFrame: The processed DataFrame
        """
        if self.df is None:
            raise HTTPException(status_code=400, detail="Data not loaded. Call load_data() first.")
        
        return self.df
    
    def save_processed_data(self, filename: Optional[str] = None) -> str:
        """
        Save the processed data to a new CSV file.
        
        Args:
            filename (str, optional): Custom filename for the processed data
            
        Returns:
            str: Path to the saved file
        """
        if self.df is None:
            raise HTTPException(status_code=400, detail="Data not loaded. Call load_data() first.")
        
        if filename is None:
            filename = f"processed_{self.file_id}.csv"
        
        filepath = f"uploads/{filename}"
        
        try:
            self.df.to_csv(filepath, index=False)
            self.preprocessing_steps.append(f"Saved processed data to: {filepath}")
            return filepath
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving processed data: {str(e)}")
    
    def reset_data(self) -> pd.DataFrame:
        """
        Reset the DataFrame to its original state.
        
        Returns:
            pd.DataFrame: The original DataFrame
        """
        if self.original_df is None:
            raise HTTPException(status_code=400, detail="Original data not available.")
        
        self.df = self.original_df.copy()
        self.preprocessing_steps = []
        self._analyze_columns()
        
        return self.df


def preprocess_data(file_id: str, db: Session, 
                   preprocessing_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to preprocess data based on configuration.
    
    Args:
        file_id (str): The unique ID of the file in the database
        db (Session): The database session
        preprocessing_config (Dict): Configuration for preprocessing steps
        
    Returns:
        Dict: Results of preprocessing including summary and processed data info
    """
    try:
        preprocessor = DataPreprocessor(file_id, db)
        preprocessor.load_data()
        
        # Get initial summary
        initial_summary = preprocessor.get_data_summary()
        
        # Apply preprocessing steps based on configuration
        if preprocessing_config.get('handle_missing', False):
            missing_config = preprocessing_config.get('missing_config', {})
            preprocessor.handle_missing_values(
                strategy=missing_config.get('strategy', 'remove_column'),
                missing_threshold=missing_config.get('missing_threshold', 0.8),
                columns=missing_config.get('columns'),
                custom_value=missing_config.get('custom_value')
            )
        
        if preprocessing_config.get('handle_outliers', False):
            outlier_config = preprocessing_config.get('outlier_config', {})
            preprocessor.handle_outliers(
                method=outlier_config.get('method', 'iqr'),
                columns=outlier_config.get('columns'),
                threshold=outlier_config.get('threshold', 1.5)
            )
        
        if preprocessing_config.get('encode_categorical', False):
            encoding_config = preprocessing_config.get('encoding_config', {})
            preprocessor.encode_categorical_variables(
                columns=encoding_config.get('columns'),
                method=encoding_config.get('method', 'label')
            )
        
        if preprocessing_config.get('scale_features', False):
            scaling_config = preprocessing_config.get('scaling_config', {})
            preprocessor.scale_features(
                columns=scaling_config.get('columns'),
                method=scaling_config.get('method', 'standard')
            )
        
        if preprocessing_config.get('remove_duplicates', False):
            preprocessor.remove_duplicates()
        
        # Get final summary
        final_summary = preprocessor.get_data_summary()
        
        # Save processed data if requested
        processed_filepath = None
        if preprocessing_config.get('save_processed', False):
            processed_filepath = preprocessor.save_processed_data(
                preprocessing_config.get('output_filename')
            )
        
        return {
            'success': True,
            'file_id': file_id,
            'initial_summary': initial_summary,
            'final_summary': final_summary,
            'preprocessing_steps': preprocessor.preprocessing_steps,
            'processed_filepath': processed_filepath,
            'message': 'Data preprocessing completed successfully'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in data preprocessing: {str(e)}")
