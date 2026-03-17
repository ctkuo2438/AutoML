import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.db.models.user_model import User
from app.schemas.user_validator import UserCreate, UserResponse, UserLogin
from app.services.auth_service import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)

logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)
router = APIRouter()


@router.post("/register", response_model=UserResponse)
@limiter.limit("20/minute")
async def register_user(request: Request, user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        logger.warning("register: username=%s already taken", user_data.username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    existing_email = db.query(User).filter(User.email == user_data.email).first()
    if existing_email:
        logger.warning("register: email already registered (username=%s)", user_data.username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    logger.info("register: new user created user_id=%d username=%s", new_user.id, new_user.username)
    return UserResponse.model_validate(new_user)


@router.post("/login")
@limiter.limit("10/minute")
async def login_user(request: Request, user_data: UserLogin, db: Session = Depends(get_db)):
    """Login user and return JWT token"""
    user = db.query(User).filter(User.username == user_data.username).first()
    if not user or not verify_password(user_data.password, user.hashed_password):
        logger.warning("login: failed attempt username=%s", user_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    access_token = create_access_token(data={"sub": str(user.id)})
    logger.info("login: success user_id=%d username=%s", user.id, user.username)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.id,
    }


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse.model_validate(current_user)
