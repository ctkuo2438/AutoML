from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.inference_validator import PredictionRequest, PredictionResponse
from app.services.auth_service import verify_token
from app.services.data.load_csv import verify_file_ownership
from app.services.inference.predictor import ModelPredictor

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db),
):
    verify_file_ownership(request.file_id, user_id, db)
    predictor = ModelPredictor(
        job_id=request.job_id,
        file_id=request.file_id,
        user_id=user_id,
        db=db,
    )
    result = predictor.predict()
    return PredictionResponse(**result)
