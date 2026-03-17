from pydantic import BaseModel
from typing import Optional

class UserCreate(BaseModel):
    """Schema for creating a new user"""
    username: str
    email: str

class UserResponse(BaseModel):
    """Schema for user response"""
    id: int
    username: str
    email: str
    
    class Config:
        from_attributes = True  # This allows the model to work with SQLAlchemy objects

class UserLogin(BaseModel):
    """Schema for user login"""
    username: str
