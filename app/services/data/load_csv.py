import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session
from app.db.models.file_model import File

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

    # Query the database for the file using its ID, first() returns the first result or None
    db_file = db.query(File).filter(File.id == file_id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found.")
    
    # Load the CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(str(db_file.filepath))
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty.")
        return df
    
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty.")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="CSV file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading CSV file: {str(e)}")
