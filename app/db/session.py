from sqlalchemy import create_engine  # Used to create a connection to the database
from sqlalchemy.orm import sessionmaker  # Used to create a session factory for database interactions
from app.core.config import settings  # Importing configuration settings, including the database URL

'''
Create a database engine using the DATABASE_URL from the settings.
The engine manages the connection pool and database connections.
'''
engine = create_engine(settings.DATABASE_URL)

'''
Create a session factory bound to the engine.
SessionLocal is a factory for creating new Session objects.
- autocommit=False: Transactions are not automatically committed
- autoflush=False: Changes are not automatically flushed to the database
'''
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
