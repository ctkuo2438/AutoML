import os
import uuid
from fastapi import HTTPException, UploadFile
from app.db.models.file_model import File
from sqlalchemy.orm import Session

def save_uploaded_file(file: UploadFile, user_id: int, db: Session) -> dict:
    """
    Save an uploaded file to the uploads/ folder and store its metadata to the SQLite database.

    Args:
        file (UploadFile): The uploaded file object.
        user_id (int): The ID of the user uploading the file.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the file ID, file path, filename, and a success message.

    Raises:
        HTTPException: If there is an error saving the file or storing its metadata.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    
    file_id = str(uuid.uuid4())  # Generate a unique ID for the file
    file_path = f"uploads/{file_id}_{file.filename}"  # Create a unique file path

    os.makedirs("uploads", exist_ok=True)  # Ensure the uploads directory exists

    try:
        with open(file_path, "wb") as f:
            f.write(file.file.read())  # save the uploaded file to uploads directory
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    # Store file metadata in the database
    try:
        db_file = File(
            id=file_id,
            filename=file.filename,
            filepath=file_path,
            user_id=user_id
        )
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save file metadata: {str(e)}")

    # If everything is successful, return the file ID and file path
    return {"file_id": file_id, 
            "file_path": file_path,
            "filename": file.filename,
            "message": "File uploaded successfully."}


