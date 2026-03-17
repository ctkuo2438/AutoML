import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text

from app.db.base import Base


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    file_id = Column(String, ForeignKey("files.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    task_type = Column(String, nullable=False)
    target_column = Column(String, nullable=False)
    algorithm = Column(String, nullable=False)
    hyperparameters = Column(Text, nullable=True)
    test_size = Column(Float, default=0.2)
    random_state = Column(Integer, default=42)
    metrics = Column(Text, nullable=True)
    model_filepath = Column(String, nullable=True)
    status = Column(String, default="pending")
    error_message = Column(String, nullable=True)
    training_duration_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
