from typing import List

from app.utils.file_upload_utils import save_uploaded_file
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from app.api.deps import get_db
from app.db.models.file_model import File as FileModel
from app.schemas.file_validator import File as FileSchema, FileUploadResponse
from app.services.auth_service import verify_token

router = APIRouter()


@router.get("/", response_model=List[FileSchema])
async def list_files(
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db),
):
    return db.query(FileModel).filter(FileModel.user_id == user_id).all()


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Upload a CSV file to the server, save it to the uploads/ directory, and store its metadata in the database.

    Args:
        file (UploadFile): The CSV file to upload.
        user_id (int): The ID of the authenticated user uploading the file.
        db (Session): The database session, injected via dependency.

    Returns:
        FileUploadResponse: A response containing the file id, file path, filename, and success message.

    Raises:
        HTTPException: If the file is invalid or an error occurs during processing.
    """
    try:
        result = save_uploaded_file(file, user_id, db)
        return FileUploadResponse(**result) # ** means unpacking the dictionary into keyword arguments
    except HTTPException as e:
        raise e
