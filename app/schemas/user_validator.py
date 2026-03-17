from pydantic import BaseModel


class UserCreate(BaseModel):
    """Schema for creating a new user"""
    username: str
    email: str
    password: str


class UserResponse(BaseModel):
    """Schema for user response"""
    id: int
    username: str
    email: str

    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    """Schema for user login"""
    username: str
    password: str
