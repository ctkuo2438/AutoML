import logging
from typing import List
from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ENV: str = "development"
    DATABASE_URL: str = "sqlite:///./automl.db"
    UPLOAD_DIR: str = "uploads"
    MODEL_DIR: str = "models"
    SECRET_KEY: str = "changeme"
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    LOG_LEVEL: str = "INFO"

    @model_validator(mode="after")
    def validate_secret_key(self) -> "Settings":
        weak = self.SECRET_KEY == "changeme" or len(self.SECRET_KEY) < 32
        if weak:
            if self.ENV == "production":
                raise ValueError(
                    "SECRET_KEY must be at least 32 characters and not the default value. "
                    "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
                )
            logging.warning(
                "SECRET_KEY is weak or default. Set a strong SECRET_KEY before deploying to production."
            )
        return self

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
