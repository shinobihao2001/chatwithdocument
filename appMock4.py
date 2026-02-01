import os
import subprocess
import tempfile
import requests
from pathlib import Path

import streamlit as st
from bs4 import BeautifulSoup

# ============================
# CẤU HÌNH
# ============================

OLLAMA_URL = "http://14.241.244.57:11434/api/chat"
MODEL_NAME = "llama3.1:8b"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MAIN_PANEL_HEIGHT = 680


# ============================
# GỌI OLLAMA
# ============================

def chat_with_ollama(context: str, question: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an assistant that answers questions strictly "
                    "based on the following document content:\n\n"
                    f"{context}"
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "stream": False
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"]


# ============================
# CHẠY CHANDRA
# ============================

def run_chandra_cli(input_file: Path, output_dir: Path):
    cmd = [
        "chandra",
        str(input_file),
        str(output_dir),
        "--method",
        "hf"
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)


# ============================
# ĐỌC OCR
# ============================

def read_ocr_text_and_tables(output_dir: Path):
    text_blocks = []
    html_tables = []

    for file in sorted(output_dir.glob("**/*")):
        if file.suffix.lower() in [".md", ".txt"]:
            text_blocks.append(
                file.read_text(encoding="utf-8", errors="ignore")
            )

        if file.suffix.lower() in [".html", ".htm"]:
            html = file.read_text(encoding="utf-8", errors="ignore")
            if "<table" in html.lower():
                html_tables.append(html)

    return "\n\n".join(text_blocks), html_tables


def table_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    lines = []
    for row in soup.find_all("tr"):
        cells = row.find_all(["th", "td"])
        values = [c.get_text(" ", strip=True) for c in cells]
        if any(values):
            lines.append(" | ".join(values))

    return "\n".join(lines)


def read_ocr_images(output_dir: Path):
    images = []
    for file in sorted(output_dir.glob("**/*")):
        if file.suffix.lower() in {".webp", ".png", ".jpg", ".jpeg"}:
            images.append(
                {
                    "name": file.name,
                    "bytes": file.read_bytes()
                }
            )
    return images


# ============================
# UI
# ============================

st.set_page_config(
    page_title="OCR + Chat LLM",
    layout="wide"
)

st.markdown("### 📄 OCR tài liệu → 💬 Hỏi LLM")
st.markdown(
    "<small>Chandra CLI • PDF / Image • Text-only LLM</small>",
    unsafe_allow_html=True,
)


# ============================
# SESSION
# ============================

for k, v in {
    "ocr_text": "",
    "ocr_tables_html": [],
    "ocr_images": [],
    "chat_answer": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================
# KHUNG CHÍNH FIX HEIGHT
# ============================

with st.container():

    left_col, right_col = st.columns([1.1, 1.4])

    # ========================
    # CỘT TRÁI
    # ========================

    with left_col:

        st.markdown("#### 📤 Tài liệu")

        upload_col, btn_col = st.columns([2.6, 1])

        with upload_col:
            uploaded_file = st.file_uploader(
                "Upload",
                type=["pdf", "jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed"
            )

        with btn_col:
            run_btn = st.button("🚀 OCR", use_container_width=True)

        # preview + status FIX HEIGHT
        with st.container(height=MAIN_PANEL_HEIGHT):

            if uploaded_file:
                suffix = Path(uploaded_file.name).suffix.lower()

                if suffix == ".pdf":
                    st.pdf(uploaded_file)
                else:
                    st.image(uploaded_file, use_container_width=True)

            if run_btn and uploaded_file:

                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)

                    input_file = tmp / f"input{suffix}"
                    output_dir = tmp / "ocr_output"

                    input_file.write_bytes(uploaded_file.read())
                    output_dir.mkdir(exist_ok=True)

                    with st.spinner("OCR đang chạy..."):
                        try:
                            run_chandra_cli(input_file, output_dir)

                            text, tables = read_ocr_text_and_tables(output_dir)

                            st.session_state.ocr_text = text
                            st.session_state.ocr_tables_html = tables
                            st.session_state.ocr_images = read_ocr_images(output_dir)

                            st.success("Hoàn tất")

                        except Exception as e:
                            st.error("OCR lỗi")
                            st.exception(e)

    # ========================
    # CỘT PHẢI
    # ========================

    with right_col:

        tab_ocr, tab_chat = st.tabs(
            ["📄 Kết quả OCR", "💬 Chat LLM"]
        )

        # ----- OCR TAB -----
        with tab_ocr:
            with st.container(height=MAIN_PANEL_HEIGHT):

                if st.session_state.ocr_text:
                    st.markdown("#### Văn bản")
                    st.markdown(st.session_state.ocr_text)

                if st.session_state.ocr_tables_html:
                    st.markdown("#### Bảng")

                    for i, html in enumerate(
                        st.session_state.ocr_tables_html, 1
                    ):
                        st.markdown(f"**Bảng {i}**")
                        st.markdown(html, unsafe_allow_html=True)

                if st.session_state.ocr_images:
                    st.markdown("#### Dấu / chữ ký")

                    cols = st.columns(3)
                    for i, img in enumerate(
                        st.session_state.ocr_images
                    ):
                        with cols[i % 3]:
                            st.image(
                                img["bytes"],
                                caption=img["name"],
                                use_container_width=True
                            )

        # ----- CHAT TAB -----
        with tab_chat:

            with st.container(height=MAIN_PANEL_HEIGHT):

                st.markdown("#### Trả lời")

                if st.session_state.chat_answer:
                    st.markdown(st.session_state.chat_answer)

            question = st.text_area(
                "Câu hỏi",
                height=90,
                placeholder="Ví dụ: Văn bản ban hành ngày nào?",
                label_visibility="collapsed"
            )

            if st.button("📨 Hỏi LLM", use_container_width=True) and question:

                with st.spinner("LLM đang xử lý..."):

                    table_text = "\n\n".join(
                        table_html_to_text(t)
                        for t in st.session_state.ocr_tables_html
                    )

                    llm_context = (
                        st.session_state.ocr_text
                        + "\n\n"
                        + table_text
                    )

                    try:
                        answer = chat_with_ollama(
                            llm_context,
                            question
                        )
                        st.session_state.chat_answer = answer

                    except Exception as e:
                        st.error("LLM lỗi")
                        st.exception(e)
