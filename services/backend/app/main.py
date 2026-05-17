from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .config import settings
from .ml_pipeline import PipelineError, PriceTagProcessor


app = FastAPI(
    title="Lenta Tech ML Backend",
    version="0.1.0",
    description="Video-to-submission service for price-tag recognition.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_processor: PriceTagProcessor | None = None


def get_processor() -> PriceTagProcessor:
    global _processor
    if _processor is None:
        _processor = PriceTagProcessor(
            model_path=settings.model_path,
            device=settings.device,
            conf=settings.conf,
            iou=settings.iou,
            imgsz=settings.imgsz,
            max_frames=settings.max_frames,
        )
    return _processor


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_path": str(settings.model_path),
        "model_exists": settings.model_path.exists(),
        "storage_dir": str(settings.storage_dir),
        "device": settings.device,
    }


@app.post("/api/v1/process-video")
def process_video(file: UploadFile = File(...)) -> dict[str, object]:
    upload_path = _save_upload(file)
    try:
        result = get_processor().process(upload_path, original_filename=file.filename)
    except PipelineError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        shutil.rmtree(upload_path.parent, ignore_errors=True)

    return {
        "columns": result.columns,
        "rows": result.rows,
        "csv": result.csv_text,
        "row_count": len(result.rows),
        "tracks_detected": result.tracks_detected,
        "frames_seen": result.frames_seen,
        "processing_seconds": result.processing_seconds,
        "model_path": result.model_path,
        "device": result.device,
    }


@app.post("/api/v1/process-video.csv")
def process_video_csv(file: UploadFile = File(...)) -> Response:
    upload_path = _save_upload(file)
    try:
        result = get_processor().process(upload_path, original_filename=file.filename)
    except PipelineError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        shutil.rmtree(upload_path.parent, ignore_errors=True)

    filename = f"{Path(file.filename or 'submission').stem}_submission.csv"
    return Response(
        content=result.csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _save_upload(file: UploadFile) -> Path:
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename is required")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".avi"}:
        raise HTTPException(status_code=400, detail="supported video formats: mp4, mov, mkv, avi")

    job_dir = settings.storage_dir / str(uuid4())
    job_dir.mkdir(parents=True, exist_ok=True)
    upload_path = job_dir / f"input{suffix}"

    max_bytes = settings.max_upload_mb * 1024 * 1024
    written = 0
    with upload_path.open("wb") as dst:
        while chunk := file.file.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"file is larger than {settings.max_upload_mb} MB",
                )
            dst.write(chunk)

    return upload_path
