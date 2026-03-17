import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.endpoints import auth, csv_upload, data_preprocessing, inference, training
from app.core.config import settings
from app.db.base import Base
from app.db.session import engine

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="AutoML API",
    description="An API for uploading CSV files and performing automated machine learning tasks.",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# Import all models so SQLAlchemy registers them before create_all
from app.db.models.file_model import File  # noqa: F401, E402
from app.db.models.training_job_model import TrainingJob  # noqa: F401, E402
from app.db.models.user_model import User  # noqa: F401, E402

Base.metadata.create_all(bind=engine)

# Migrate existing databases: add experiment_name column if absent
with engine.connect() as _conn:
    try:
        _conn.execute(__import__("sqlalchemy").text(
            "ALTER TABLE training_jobs ADD COLUMN experiment_name VARCHAR"
        ))
        _conn.commit()
        logger.info("Migration: added experiment_name column to training_jobs")
    except Exception:
        pass  # Column already exists

os.makedirs(settings.MODEL_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

logger.info("Database tables created. Directories ready.")

app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(csv_upload.router, prefix="/api/files", tags=["files"])
app.include_router(data_preprocessing.router, prefix="/api/data", tags=["data-preprocessing"])
app.include_router(training.router, prefix="/api/training", tags=["model-training"])
app.include_router(inference.router, prefix="/api/inference", tags=["inference"])


@app.get("/")
async def read_root():
    return {"message": "Welcome to the AutoML API!"}
