from __future__ import annotations

import csv
import io
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .config import settings
from .ml_pipeline import PipelineError, PriceTagProcessor, ProcessingResult


logger = logging.getLogger("uvicorn.error")


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
_executor = ThreadPoolExecutor(
    max_workers=max(1, settings.job_workers),
    thread_name_prefix="lenta-job",
)
_jobs_lock = Lock()
_jobs: dict[str, "JobState"] = {}


@dataclass
class JobState:
    job_id: str
    filename: str
    status: str = "queued"
    stage: str = "queued"
    message: str = "Ожидает запуска"
    progress: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error: str | None = None
    result: ProcessingResult | None = None
    result_path: Path | None = None
    debug_path: Path | None = None


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
    storage_writable = _storage_writable()
    return {
        "status": "ok" if storage_writable else "degraded",
        "model_path": str(settings.model_path),
        "model_exists": settings.model_path.exists(),
        "storage_dir": str(settings.storage_dir),
        "storage_writable": storage_writable,
        "device": settings.device,
    }


@app.post("/api/v1/jobs")
def create_job(file: UploadFile = File(...)) -> dict[str, object]:
    job_id = str(uuid4())
    filename = file.filename or "input"
    _put_job(
        JobState(
            job_id=job_id,
            filename=filename,
            status="running",
            stage="upload",
            message="Загрузка видео на backend",
            progress=5,
        )
    )
    logger.info("job=%s event=start stage=upload filename=%s", job_id, filename)

    try:
        upload_path = _save_upload(file, job_id=job_id)
    except HTTPException as exc:
        _update_job(
            job_id,
            status="failed",
            stage="upload",
            message="Ошибка загрузки видео",
            progress=100,
            error=str(exc.detail),
        )
        raise

    _update_job(
        job_id,
        status="queued",
        stage="queued",
        message="Видео загружено, задача ожидает запуска",
        progress=8,
    )
    _executor.submit(_run_job, job_id, upload_path, filename)
    return _job_response(job_id)


@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    return _job_response(job_id)


@app.get("/api/v1/jobs/{job_id}/result")
def get_job_result(job_id: str) -> dict[str, object]:
    job = _get_job(job_id)
    if job.status == "failed":
        raise HTTPException(status_code=422, detail=job.error or "job failed")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="job is not completed yet")
    if job.result is not None:
        return _result_payload(job.result)
    if job.result_path is not None and job.result_path.exists():
        return _csv_result_payload(job.result_path.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="result csv not found")


@app.get("/api/v1/jobs/{job_id}/result.csv")
def get_job_result_csv(job_id: str) -> Response:
    job = _get_job(job_id)
    if job.status == "failed":
        raise HTTPException(status_code=422, detail=job.error or "job failed")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="job is not completed yet")

    filename = f"{job_id}.csv"
    if job.result_path is not None and job.result_path.exists():
        return Response(
            content=job.result_path.read_text(encoding="utf-8"),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    if job.result is None:
        raise HTTPException(status_code=404, detail="result csv not found")
    return Response(
        content=job.result.csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/v1/process-video")
def process_video(file: UploadFile = File(...)) -> dict[str, object]:
    upload_path = _save_upload(file)
    try:
        result = get_processor().process(upload_path, original_filename=file.filename)
    except PipelineError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        shutil.rmtree(upload_path.parent, ignore_errors=True)

    return _result_payload(result)


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


def _save_upload(file: UploadFile, job_id: str | None = None) -> Path:
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename is required")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".avi"}:
        raise HTTPException(status_code=400, detail="supported video formats: mp4, mov, mkv, avi")

    job_dir = settings.storage_dir / (job_id or str(uuid4()))
    job_dir.mkdir(parents=True, exist_ok=True)
    upload_path = job_dir / f"input{suffix}"

    max_bytes = settings.max_upload_mb * 1024 * 1024
    written = 0
    logger.info("job=%s event=upload_start filename=%s", job_id or "sync", file.filename)
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

    logger.info(
        "job=%s event=upload_done path=%s bytes=%s",
        job_id or "sync",
        upload_path,
        written,
    )
    return upload_path


def _run_job(job_id: str, upload_path: Path, original_filename: str) -> None:
    started = time.perf_counter()
    debug_path = upload_path.parent / "debug"
    logger.info("job=%s event=processing_start path=%s", job_id, upload_path)

    def progress_callback(stage: str, message: str, progress: int) -> None:
        _update_job(
            job_id,
            status="running",
            stage=stage,
            message=message,
            progress=progress,
        )

    try:
        result = get_processor().process(
            upload_path,
            original_filename=original_filename,
            progress_callback=progress_callback,
            debug_dir=debug_path,
        )
        result_path = _write_job_csv(job_id, result.csv_text)
    except PipelineError as exc:
        _update_job(
            job_id,
            status="failed",
            stage="failed",
            message="Ошибка обработки",
            progress=100,
            error=str(exc),
        )
        logger.error("job=%s event=processing_failed error=%s", job_id, exc)
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            stage="failed",
            message="Неожиданная ошибка обработки",
            progress=100,
            error=str(exc),
        )
        logger.exception("job=%s event=processing_failed_unexpected", job_id)
    else:
        elapsed = time.perf_counter() - started
        _update_job(
            job_id,
            status="completed",
            stage="completed",
            message="Готово",
            progress=100,
            result=result,
            result_path=result_path,
            debug_path=debug_path,
        )
        logger.info(
            "job=%s event=processing_done seconds=%.3f rows=%s tracks=%s frames=%s",
            job_id,
            elapsed,
            len(result.rows),
            result.tracks_detected,
            result.frames_seen,
        )
    finally:
        _cleanup_job_upload(upload_path)


def _put_job(job: JobState) -> None:
    with _jobs_lock:
        _jobs[job.job_id] = job


def _get_job(job_id: str) -> JobState:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        result_path = _job_csv_path(job_id)
        if result_path.exists():
            debug_path = settings.storage_dir / job_id / "debug"
            return JobState(
                job_id=job_id,
                filename=result_path.name,
                status="completed",
                stage="completed",
                message="Готово",
                progress=100,
                result_path=result_path,
                debug_path=debug_path if debug_path.exists() else None,
            )
        raise HTTPException(status_code=404, detail="job not found")
    return job


def _update_job(job_id: str, **changes: object) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        for key, value in changes.items():
            setattr(job, key, value)
        job.updated_at = time.time()
        snapshot = _job_snapshot(job)

    logger.info(
        "job=%s status=%s stage=%s progress=%s message=%s",
        snapshot["job_id"],
        snapshot["status"],
        snapshot["stage"],
        snapshot["progress"],
        snapshot["message"],
    )


def _job_response(job_id: str) -> dict[str, object]:
    return _job_snapshot(_get_job(job_id))


def _job_snapshot(job: JobState) -> dict[str, object]:
    has_result_file = job.result_path is not None and job.result_path.exists()
    has_debug = job.debug_path is not None and job.debug_path.exists()
    return {
        "job_id": job.job_id,
        "filename": job.filename,
        "status": job.status,
        "stage": job.stage,
        "message": job.message,
        "progress": job.progress,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "error": job.error,
        "has_result": job.result is not None or has_result_file,
        "result_filename": f"{job.job_id}.csv" if has_result_file else None,
        "has_debug": has_debug,
        "debug_path": str(job.debug_path) if has_debug else None,
    }


def _result_payload(result: ProcessingResult) -> dict[str, object]:
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


def _csv_result_payload(csv_text: str) -> dict[str, object]:
    reader = csv.DictReader(io.StringIO(csv_text), delimiter=";")
    rows = list(reader)
    columns = reader.fieldnames or []
    return {
        "columns": columns,
        "rows": rows,
        "csv": csv_text,
        "row_count": len(rows),
        "tracks_detected": len(rows),
        "frames_seen": None,
        "processing_seconds": None,
        "model_path": str(settings.model_path),
        "device": settings.device,
    }


def _job_csv_path(job_id: str) -> Path:
    return settings.storage_dir / f"{job_id}.csv"


def _write_job_csv(job_id: str, csv_text: str) -> Path:
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    result_path = _job_csv_path(job_id)
    result_path.write_text(csv_text, encoding="utf-8")
    logger.info("job=%s event=result_saved path=%s", job_id, result_path)
    return result_path


def _cleanup_job_upload(upload_path: Path) -> None:
    try:
        upload_path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("upload_cleanup_failed path=%s error=%s", upload_path, exc)
    try:
        next(upload_path.parent.iterdir())
    except StopIteration:
        shutil.rmtree(upload_path.parent, ignore_errors=True)
    except OSError:
        pass


def _storage_writable() -> bool:
    try:
        settings.storage_dir.mkdir(parents=True, exist_ok=True)
        probe_path = settings.storage_dir / ".healthcheck-write-test"
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink(missing_ok=True)
        return True
    except OSError as exc:
        logger.warning("storage_dir_not_writable path=%s error=%s", settings.storage_dir, exc)
        return False
