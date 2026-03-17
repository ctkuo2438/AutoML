import logging

import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session
from app.db.models.file_model import File

logger = logging.getLogger(__name__)


def load_csv(file_id: str, db: Session) -> pd.DataFrame:
    """
    Load a CSV file from the database using its ID and return it as a pandas DataFrame.

    Args:
        file_id (str): The unique id of the file in the database.
        db (Session): The database session.

    Returns:
        pd.DataFrame: The loaded CSV file as a pandas DataFrame.

    Raises:
        HTTPException: If the file is not found or if there is an error reading the file.
    """
    db_file = db.query(File).filter(File.id == file_id).first()
    if not db_file:
        logger.warning("load_csv: file_id=%s not found in database", file_id)
        raise HTTPException(status_code=404, detail="File not found.")

    try:
        df = pd.read_csv(str(db_file.filepath))
    except pd.errors.EmptyDataError:
        logger.warning("load_csv: file_id=%s is empty", file_id)
        raise HTTPException(status_code=400, detail="CSV file is empty.")
    except FileNotFoundError:
        logger.error("load_csv: file_id=%s path=%s not found on disk", file_id, db_file.filepath)
        raise HTTPException(status_code=404, detail="CSV file not found.")
    except Exception as e:
        logger.exception("load_csv: unexpected error reading file_id=%s", file_id)
        raise HTTPException(status_code=500, detail=f"Error loading CSV file: {str(e)}")

    if df.empty:
        logger.warning("load_csv: file_id=%s loaded but DataFrame is empty", file_id)
        raise HTTPException(status_code=400, detail="CSV file is empty.")

    logger.debug("load_csv: file_id=%s rows=%d cols=%d", file_id, df.shape[0], df.shape[1])
    return df


def verify_file_ownership(file_id: str, user_id: int, db: Session) -> None:
    """
    Verify that a file belongs to the given user.

    Raises:
        HTTPException 404 if file not found.
        HTTPException 403 if file does not belong to the user.
    """
    db_file = db.query(File).filter(File.id == file_id).first()
    if not db_file:
        logger.warning("verify_file_ownership: file_id=%s not found", file_id)
        raise HTTPException(status_code=404, detail="File not found.")
    if db_file.user_id != user_id:
        logger.warning(
            "verify_file_ownership: user_id=%d denied access to file_id=%s", user_id, file_id
        )
        raise HTTPException(status_code=403, detail="Access denied.")
