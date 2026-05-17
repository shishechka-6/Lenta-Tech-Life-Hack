from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LENTA_", extra="ignore")

    model_path: Path = PROJECT_ROOT / "runs" / "tags_v1" / "weights" / "best.pt"
    storage_dir: Path = Path("/tmp/lenta-tech-backend")
    device: str = "auto"
    conf: float = 0.25
    iou: float = 0.5
    imgsz: int = 1280
    max_frames: int | None = None
    max_upload_mb: int = 2048


settings = Settings()
