from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.api.deps import get_db
from app.db.models.user_model import User
from app.schemas.user_validator import UserCreate, UserResponse, UserLogin
from app.services.auth_service import create_access_token, get_current_user

router = APIRouter()

@router.post("/register", response_model=UserResponse)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # Check if username already exists
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    existing_email = db.query(User).filter(User.email == user_data.email).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    new_user = User(
        username=user_data.username,
        email=user_data.email
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return UserResponse.model_validate(new_user)

@router.post("/login")
async def login_user(username: str, db: Session = Depends(get_db)):
    """Login user and return JWT token"""
    # Find user by username
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username"
        )
    
    # Create access token
    access_token = create_access_token(data={"sub": str(user.id)})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.id
    }

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse.model_validate(current_user)
