import os

import pandas as pd
import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "3600"))


st.set_page_config(page_title="Lenta Tech", layout="wide")

st.title("Lenta Tech: ценники")

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


uploaded_file = st.file_uploader("Видео", type=["mp4", "mov", "mkv", "avi"])
process_clicked = st.button(
    "Обработать",
    type="primary",
    disabled=uploaded_file is None,
    use_container_width=False,
)

if process_clicked and uploaded_file is not None:
    with st.spinner("Видео обрабатывается"):
        try:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "video/mp4",
                )
            }
            response = requests.post(
                f"{BACKEND_URL}/api/v1/process-video",
                files=files,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            st.error(f"Backend error: {exc}")
            if getattr(exc, "response", None) is not None:
                st.code(exc.response.text, language="json")
        else:
            rows = result.get("rows", [])
            columns = result.get("columns", [])
            df = pd.DataFrame(rows, columns=columns)

            left, middle, right = st.columns(3)
            left.metric("Строк", result.get("row_count", len(df)))
            middle.metric("Треков", result.get("tracks_detected", 0))
            right.metric("Секунд", result.get("processing_seconds", 0))

            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "Скачать CSV",
                data=result.get("csv", "").encode("utf-8"),
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_submission.csv",
                mime="text/csv",
            )
