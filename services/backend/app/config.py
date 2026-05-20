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
    k_best_crops: int = 1
    max_crop_side: int = 768
    min_track_detections: int = 2
    decode_codes_on_crops: int = 0
    max_upload_mb: int = 2048
    job_workers: int = 1


settings = Settings()
