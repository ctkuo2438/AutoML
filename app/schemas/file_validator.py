from pydantic import BaseModel  # Base class for creating data validation and serialization schemas
from datetime import datetime  # Used for handling date and time fields
from typing import Optional  # Used for defining optional fields in schemas

'''Base schema for file data, shared across different use cases.'''
class FileBase(BaseModel):
    filename: str  # Name of the file, like 'Titanic.csv'
    filepath: str  # Path where the file is stored
    task_type: Optional[str] = None  # Type of task associated with the file (e.g., classification, regression)
    target_column: Optional[str] = None  # Target column in the file, if applicable
    description: Optional[str] = None  # Additional description or metadata about the file

'''Schema for creating a new file entry. Inherits all fields from FileBase.'''
class FileCreate(FileBase): 
    filename: str  # Name of the file (required for creation)
    user_id: int  # ID of the user who is uploading the file (required for creation)
    task_type: Optional[str] = None  # Type of task associated with the file (optional)
    target_column: Optional[str] = None  # Target column in the file, if applicable (optional)
    description: Optional[str] = None  # Additional description or metadata about the file (optional)

'''Schema representing a file entry as stored in the database, not input by users.'''
class File(FileBase): 
    id: str  # uuid of the file, unique identifier in the database, automatically generated
    upload_time: datetime  # Timestamp of when the file was uploaded
    user_id: int  # ID of the user who uploaded the file

    # Configuration for Pydantic model
    class Config:
        from_attributes = True  # Allows populating the model from ORM objects or attributes

'''Schema for file upload response, used to return information after a file is uploaded.'''
class FileUploadResponse(BaseModel):
    file_id: str  # Unique identifier for the uploaded file
    file_path : str  # Path where the file is stored
    filename: str  # Name of the uploaded file
    message: str  # Response message indicating success or additional information