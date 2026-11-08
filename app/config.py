"""Task 4.4 - every setting comes from an environment variable (safe local defaults)."""
import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "models/champion")
    THRESHOLD = os.getenv("THRESHOLD")                       # empty -> threshold stored with the model
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///visionguard.db")
    MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "5"))
    INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local")    # local | remote (Task 8)
    MODEL_API_URL = os.getenv("MODEL_API_URL", "")           # remote mode: base URL of the cloud API
    API_KEY = os.getenv("API_KEY", "")                       # protects /api/predict
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-me")
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "gif", "bmp"}
