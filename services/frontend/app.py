import os
import time

import pandas as pd
import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "3600"))
STATUS_TIMEOUT_SECONDS = int(os.getenv("STATUS_TIMEOUT_SECONDS", "30"))
POLL_INTERVAL_SECONDS = float(os.getenv("POLL_INTERVAL_SECONDS", "2"))

STAGE_LABELS = {
    "upload": "Загрузка",
    "queued": "В очереди",
    "preprocess": "Предпроцесс",
    "yolo": "YOLO прогон",
    "ocr": "OCR",
    "postprocess": "Постобработка",
    "completed": "Готово",
    "failed": "Ошибка",
}
STAGE_ORDER = ("upload", "queued", "preprocess", "yolo", "ocr", "postprocess", "completed")
TERMINAL_STATUSES = {"completed", "failed"}


def current_job_id() -> str | None:
    value = st.query_params.get("job_id")
    if isinstance(value, list):
        value = value[0] if value else None
    return str(value).strip() if value else None


def open_job(job_id: str) -> None:
    st.query_params["job_id"] = job_id
    st.rerun()


def open_upload_view() -> None:
    st.query_params.clear()
    st.rerun()


def fetch_job(job_id: str) -> dict[str, object]:
    response = requests.get(
        f"{BACKEND_URL}/api/v1/jobs/{job_id}",
        timeout=STATUS_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def fetch_result(job_id: str) -> dict[str, object]:
    response = requests.get(
        f"{BACKEND_URL}/api/v1/jobs/{job_id}/result",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def render_sidebar() -> None:
    with st.sidebar:
        st.caption("Backend")
        st.code(BACKEND_URL, language="text")

        if st.button("Healthcheck", use_container_width=True):
            try:
                response = requests.get(f"{BACKEND_URL}/health", timeout=10)
                response.raise_for_status()
                st.json(response.json())
            except requests.RequestException as exc:
                st.error(str(exc))

        st.divider()
        reattach_job_id = st.text_input("Job ID", placeholder="Вставьте job_id")
        if st.button("Открыть job", use_container_width=True, disabled=not reattach_job_id.strip()):
            open_job(reattach_job_id.strip())


def render_upload_view() -> None:
    st.title("Lenta Tech: ценники")

    uploaded_file = st.file_uploader("Видео", type=["mp4", "mov", "mkv", "avi"])
    process_clicked = st.button(
        "Обработать",
        type="primary",
        disabled=uploaded_file is None,
        use_container_width=False,
    )

    if not process_clicked or uploaded_file is None:
        return

    try:
        with st.spinner("Загрузка видео на backend"):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "video/mp4",
                )
            }
            response = requests.post(
                f"{BACKEND_URL}/api/v1/jobs",
                files=files,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            job = response.json()
    except requests.RequestException as exc:
        st.error(f"Backend error: {exc}")
        if getattr(exc, "response", None) is not None:
            st.code(exc.response.text, language="json")
        return

    job_id = str(job["job_id"])
    st.session_state["last_job_id"] = job_id
    open_job(job_id)


def render_stage_table(job: dict[str, object]) -> None:
    current_stage = str(job.get("stage") or "queued")
    status = str(job.get("status") or "queued")
    current_index = STAGE_ORDER.index(current_stage) if current_stage in STAGE_ORDER else -1

    rows = []
    for index, stage in enumerate(STAGE_ORDER):
        if status == "completed":
            stage_status = "готово"
        elif status == "failed":
            stage_status = "ошибка" if stage == current_stage else "не завершено"
        elif index < current_index:
            stage_status = "готово"
        elif stage == current_stage:
            stage_status = "сейчас"
        else:
            stage_status = "ожидает"
        rows.append({"Этап": STAGE_LABELS.get(stage, stage), "Статус": stage_status})

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_result(job_id: str) -> None:
    try:
        result = fetch_result(job_id)
    except requests.RequestException as exc:
        st.warning(f"Результат готов, но frontend не смог получить CSV: {exc}")
        if getattr(exc, "response", None) is not None:
            st.code(exc.response.text, language="json")
        return

    rows = result.get("rows", [])
    columns = result.get("columns", [])
    df = pd.DataFrame(rows, columns=columns)

    left, middle, right = st.columns(3)
    left.metric("Строк", result.get("row_count", len(df)))
    middle.metric("Треков", result.get("tracks_detected", 0))
    right.metric("Секунд", result.get("processing_seconds") or "-")

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Скачать CSV",
        data=str(result.get("csv", "")).encode("utf-8"),
        file_name=f"{job_id}.csv",
        mime="text/csv",
    )


def render_job_view(job_id: str) -> None:
    st.title("Обработка видео")

    left, middle, right = st.columns([1, 1, 4])
    if left.button("Refresh", type="primary", use_container_width=True):
        st.rerun()
    if middle.button("Новая загрузка", use_container_width=True):
        open_upload_view()

    st.caption(f"job_id: {job_id}")

    auto_refresh = st.checkbox("Автообновлять", value=True)

    try:
        job = fetch_job(job_id)
    except requests.Timeout as exc:
        st.warning(
            f"Backend не ответил за {STATUS_TIMEOUT_SECONDS} секунд. "
            "Job не отменен; нажмите Refresh, чтобы переподключиться."
        )
        st.caption(str(exc))
        return
    except requests.RequestException as exc:
        st.error(f"Backend error: {exc}")
        if getattr(exc, "response", None) is not None:
            st.code(exc.response.text, language="json")
        return

    progress = int(job.get("progress") or 0)
    progress = max(0, min(progress, 100))
    st.progress(progress)

    stage = str(job.get("stage") or "queued")
    stage_label = STAGE_LABELS.get(stage, stage)
    message = str(job.get("message") or stage_label)
    status = str(job.get("status") or "queued")

    if status == "completed":
        st.success(f"{stage_label}: {message}")
    elif status == "failed":
        st.error(str(job.get("error") or message))
    else:
        st.info(f"{stage_label}: {message}")

    render_stage_table(job)

    if status == "completed":
        render_result(job_id)
        return

    if status == "failed":
        return

    if auto_refresh and status not in TERMINAL_STATUSES:
        time.sleep(POLL_INTERVAL_SECONDS)
        st.rerun()


st.set_page_config(page_title="Lenta Tech", layout="wide")
render_sidebar()

job_id = current_job_id()
if job_id:
    render_job_view(job_id)
else:
    render_upload_view()
