from typing import Generator

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.services.auth_service import verify_token
from app.services.data.load_csv import verify_file_ownership
from app.services.data.preprocess import DataPreprocessor


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_preprocessor(
    file_id: str,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db),
) -> DataPreprocessor:
    try:
        verify_file_ownership(file_id, user_id, db)
        preprocessor = DataPreprocessor(file_id, db)
        preprocessor.load_data()
        return preprocessor
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
