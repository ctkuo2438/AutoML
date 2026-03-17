import json
import os

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.db.models.training_job_model import TrainingJob
from app.schemas.training_validator import (
    TrainingJobListResponse,
    TrainingRequest,
    TrainingResponse,
)
from app.services.auth_service import verify_token
from app.services.data.load_csv import verify_file_ownership
from app.services.training.trainer import ModelTrainer

router = APIRouter()


def _job_to_response(job: TrainingJob) -> TrainingResponse:
    metrics = json.loads(job.metrics) if job.metrics else None
    message = job.error_message or ("Model trained successfully" if job.status == "completed" else job.status)
    return TrainingResponse(
        job_id=job.id,
        file_id=job.file_id,
        task_type=job.task_type,
        algorithm=job.algorithm,
        status=job.status,
        metrics=metrics,
        model_filepath=job.model_filepath,
        training_duration_seconds=job.training_duration_seconds,
        message=message,
    )


@router.post("/train", response_model=TrainingResponse)
async def train_model(
    request: TrainingRequest,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db),
):
    verify_file_ownership(request.file_id, user_id, db)
    trainer = ModelTrainer(
        user_id=user_id,
        config=request.model_dump(),
        db=db,
    )
    result = trainer.train()
    return TrainingResponse(**result)


@router.get("/jobs", response_model=TrainingJobListResponse)
async def list_training_jobs(
    file_id: str = None,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db),
):
    query = db.query(TrainingJob).filter(TrainingJob.user_id == user_id)
    if file_id:
        query = query.filter(TrainingJob.file_id == file_id)
    jobs = query.order_by(TrainingJob.created_at.desc()).all()
    return TrainingJobListResponse(jobs=[_job_to_response(j) for j in jobs])


@router.get("/jobs/{job_id}", response_model=TrainingResponse)
async def get_training_job(
    job_id: str,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db),
):
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found.")
    if job.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied.")
    return _job_to_response(job)


@router.delete("/jobs/{job_id}", response_model=TrainingResponse)
async def delete_training_job(
    job_id: str,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db),
):
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found.")
    if job.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied.")

    response = _job_to_response(job)

    if job.model_filepath and os.path.exists(job.model_filepath):
        os.remove(job.model_filepath)

    db.delete(job)
    db.commit()
    return response
