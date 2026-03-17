from sqlalchemy.orm import Session
from app.db.session import SessionLocal  # Importing the session factory for database interactions
from typing import Generator  # Importing Generator for type hinting

# Generator[yield_type, send_type, return_type]
def get_db() -> Generator[Session, None, None]:
    """
    Provide a database session for use in FastAPI endpoints.

    Yields:
        Session: A SQLAlchemy database session.

    Ensures:
        The session is closed after use, even if an error occurs.
    """
    db = SessionLocal()  # Create a new database session
    try:
        yield db  # Provide the session to the caller
    except Exception as e:
        raise Exception(f"An error occurred while using the database session: {str(e)}")
    finally:
        if db:
            db.close()  # Ensure the session is closed after use