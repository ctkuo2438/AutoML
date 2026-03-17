import logging
import os
import pathlib
import re
import uuid

from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models.file_model import File

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB


async def save_uploaded_file(file: UploadFile, user_id: int, db: Session) -> dict:
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    # Sanitize filename: strip directory components, allow only safe characters
    safe_name = pathlib.Path(file.filename).name
    safe_name = re.sub(r"[^\w.\-]", "_", safe_name)

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds maximum allowed size of 100 MB.")

    file_id = str(uuid.uuid4())
    upload_root = pathlib.Path(settings.UPLOAD_DIR).resolve()
    upload_root.mkdir(parents=True, exist_ok=True)
    file_path = upload_root / f"{file_id}_{safe_name}"

    try:
        file_path.write_bytes(contents)
    except Exception:
        logger.exception("save_uploaded_file: failed to write file_id=%s", file_id)
        raise HTTPException(status_code=500, detail="Error saving file.")

    try:
        db_file = File(
            id=file_id,
            filename=safe_name,
            filepath=str(file_path),
            user_id=user_id,
        )
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
    except Exception:
        logger.exception("save_uploaded_file: failed to save metadata for file_id=%s", file_id)
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to save file metadata.")

    logger.info("save_uploaded_file: file_id=%s user_id=%d size=%d bytes", file_id, user_id, len(contents))
    return {
        "file_id": file_id,
        "file_path": str(file_path),
        "filename": safe_name,
        "message": "File uploaded successfully.",
    }
