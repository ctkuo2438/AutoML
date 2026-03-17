import logging
from typing import List
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./automl.db"
    UPLOAD_DIR: str = "uploads"
    MODEL_DIR: str = "models"
    SECRET_KEY: str = "changeme"
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    LOG_LEVEL: str = "INFO"

    @field_validator("SECRET_KEY")
    @classmethod
    def secret_key_must_be_set(cls, v: str) -> str:
        if v == "changeme":
            logging.warning(
                "SECRET_KEY is using the default value. "
                "Set a strong SECRET_KEY environment variable in production."
            )
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
