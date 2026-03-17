from sqlalchemy import Column, Integer, String, DateTime, ForeignKey # SQLAlchemy column types
from datetime import datetime, timezone  # Used for handling timestamps
from app.db.base import Base  # Base class for ORM models
import uuid  # Used for generating unique identifiers

# Define the File model, representing the 'files' table in the psql database
# This files table will store metadata about uploaded files
class File(Base):
    __tablename__ = 'files'  # Name of the database table

    # Define the columns of the 'files' table
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)  # Primary key, unique identifier for each file
    filename = Column(String, nullable=False)  # Name of the file (required)
    filepath = Column(String, nullable=False)  # Path where the file is stored (required)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)  # ID of the user who uploaded the file (required)
    upload_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))  # Timestamp of file upload, defaults to current UTC time
    task_type = Column(String, nullable=True)  # Type of task associated with the file (e.g., classification, regression) (optional)
    target_column = Column(String, nullable=True)  # Target column in the file, if applicable (optional)
    description = Column(String, nullable=True)  # Additional description or metadata about the file (optional)

