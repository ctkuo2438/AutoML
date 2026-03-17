import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import auth, csv_upload, data_preprocessing, training
from app.core.config import settings
from app.db.base import Base
from app.db.session import engine

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AutoML API",
    description="An API for uploading CSV files and performing automated machine learning tasks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import all models so SQLAlchemy registers them before create_all
from app.db.models.file_model import File  # noqa: F401, E402
from app.db.models.training_job_model import TrainingJob  # noqa: F401, E402
from app.db.models.user_model import User  # noqa: F401, E402

Base.metadata.create_all(bind=engine)

os.makedirs(settings.MODEL_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

logger.info("Database tables created. Directories ready.")

app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(csv_upload.router, prefix="/api/files", tags=["files"])
app.include_router(data_preprocessing.router, prefix="/api/data", tags=["data-preprocessing"])
app.include_router(training.router, prefix="/api/training", tags=["model-training"])


@app.get("/")
async def read_root():
    return {"message": "Welcome to the AutoML API!"}
