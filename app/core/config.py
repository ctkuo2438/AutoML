# BaseSettings is used to manage application configuration using environment variables
from pydantic_settings import BaseSettings

# Define the Settings class, inheriting from BaseSettings
# This class is used to load and validate configuration values from environment variables
class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./automl.db"
    UPLOAD_DIR: str = "uploads"
    MODEL_DIR: str = "models"
    SECRET_KEY: str = "your-secret-key-here-change-in-production-please-use-a-secure-random-key"

    # Configuration for the Settings class
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Instantiate the Settings class to load the configuration
settings = Settings()
