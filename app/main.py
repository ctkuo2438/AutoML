from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import csv_upload, auth, data_preprocessing, training
from app.db.base import Base
from app.core.config import settings
from sqlalchemy import create_engine

app = FastAPI(
    title = "AutoML API",
    description = "An API for uploading CSV files and performing automated machine learning tasks.",
    version = "1.0.0",
)

# Add CORS middleware to allow frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

'''
Import all models to ensure they are registered with SQLAlchemy's metadata.
This is necessary for SQLAlchemy to create the corresponding tables in the database.
'''
import os
from app.db.models.file_model import File
from app.db.models.user_model import User
from app.db.models.training_job_model import TrainingJob
# Create the database engine using the connection string from settings
engine = create_engine(settings.DATABASE_URL)
# Create all tables in the database (if they don't exist)
Base.metadata.create_all(bind=engine)
# Ensure required directories exist
os.makedirs(settings.MODEL_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

app.include_router(
    auth.router,
    prefix="/api/auth",
    tags=["authentication"],
)

app.include_router(
    csv_upload.router, # file is the endpoint for uploading files
    prefix="/api/files",
    tags=["files"], # POST /api/files/upload is the endpoint for uploading files
)

app.include_router(
    data_preprocessing.router,
    prefix="/api/data",
    tags=["data-preprocessing"],
)

app.include_router(
    training.router,
    prefix="/api/training",
    tags=["model-training"],
)

@app.get("/") # This is the root endpoint
async def read_root():
    return {"message": "Welcome to the AutoML API!"}
