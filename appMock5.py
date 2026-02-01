import os
import subprocess
import tempfile
import requests
from pathlib import Path

import streamlit as st
from bs4 import BeautifulSoup

# ============================
# CONFIG
# ============================

OLLAMA_URL = "http://14.241.244.57:11434/api/chat"
MODEL_NAME = "llama3.1:8b"

MAIN_COL_HEIGHT = 92  # % viewport height

# ============================
# CSS LAYOUT
# ============================

st.set_page_config(layout="wide")

st.markdown(f"""
<style>

/* ===== COLUMN HEIGHT ===== */

.main-col {{
    height: {MAIN_COL_HEIGHT}vh;
    display: flex;
    flex-direction: column;
}}

.fixed-panel {{
    flex: 1;
    overflow-y: auto;
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 12px;
}}

.no-border {{
    border: none !important;
    box-shadow: none !important;
}}

/* ===== BUTTON & UPLOAD COMPACT ===== */

section[data-testid="stFileUploader"] {{
    padding: 0.2rem !important;
}}

div[data-testid="stButton"] button {{
    padding: 0.3rem 0.7rem !important;
    font-size: 0.85rem !important;
}}

.app-title {{
    font-size: 1.6rem;
    font-weight: 700;
    padding-bottom: 0.5rem;
}}

</style>
""", unsafe_allow_html=True)


# ============================
# LLM
# ============================

def chat_with_ollama(context: str, question: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an assistant that answers strictly "
                    "based on the following document:\n\n"
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
# OCR CLI
# ============================

def run_chandra_cli(input_file: Path, output_dir: Path):
    cmd = ["chandra", str(input_file), str(output_dir), "--method", "hf"]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)


# ============================
# READ OCR
# ============================

def read_ocr_text_and_tables(output_dir: Path):
    text_blocks = []
    html_tables = []

    for file in output_dir.glob("**/*"):
        if file.suffix.lower() in [".md", ".txt"]:
            text_blocks.append(file.read_text(errors="ignore"))

        if file.suffix.lower() in [".html", ".htm"]:
            html = file.read_text(errors="ignore")
            if "<table" in html.lower():
                html_tables.append(html)

    return "\n\n".join(text_blocks), html_tables


def table_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for r in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in r.find_all(["td", "th"])]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


# ============================
# SESSION
# ============================

for k in ["ocr_text", "ocr_tables_html", "chat_answer"]:
    st.session_state.setdefault(k, "" if "text" in k or "answer" in k else [])


# ============================
# LAYOUT
# ============================

col1, col2, col3 = st.columns([0.8, 1.6, 1.6])

# ============================
# COLUMN 1
# ============================

with col1:

    st.markdown('<div class="main-col">', unsafe_allow_html=True)

    st.markdown('<div class="app-title">📄 OCR + LLM Dashboard</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload",
        type=["pdf", "png", "jpg", "jpeg", "webp"]
    )

    run_btn = st.button("🚀 Chạy OCR", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================
# COLUMN 2
# ============================

with col2:

    st.markdown('<div class="main-col">', unsafe_allow_html=True)

    st.markdown("### 📂 Tài liệu")

    st.markdown('<div class="fixed-panel">', unsafe_allow_html=True)

    if uploaded_file:
        ext = Path(uploaded_file.name).suffix.lower()
        if ext == ".pdf":
            st.pdf(uploaded_file)
        else:
            st.image(uploaded_file, use_container_width=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

# ============================
# COLUMN 3
# ============================

with col3:

    st.markdown('<div class="main-col">', unsafe_allow_html=True)

    tab_ocr, tab_chat = st.tabs(["📄 OCR", "💬 Chat LLM"])

    # OCR TAB
    with tab_ocr:

        st.markdown('<div class="fixed-panel no-border">', unsafe_allow_html=True)

        if st.session_state.ocr_text:
            st.markdown(st.session_state.ocr_text)

        for html in st.session_state.get("ocr_tables_html", []):
            st.markdown(html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # CHAT TAB
    with tab_chat:

        st.markdown('<div class="fixed-panel">', unsafe_allow_html=True)

        if st.session_state.chat_answer:
            st.markdown(st.session_state.chat_answer)

        question = st.text_area(
            "Câu hỏi",
            height=100,
            placeholder="Ví dụ: Văn bản ban hành ngày nào?"
        )

        if st.button("📨 Hỏi LLM", use_container_width=True) and question:

            ctx = st.session_state.ocr_text

            try:
                st.session_state.chat_answer = chat_with_ollama(ctx, question)
            except Exception as e:
                st.error("LLM lỗi")
                st.exception(e)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================
# OCR RUN
# ============================

if run_btn and uploaded_file:

    with tempfile.TemporaryDirectory() as tmp:

        tmp = Path(tmp)

        input_file = tmp / uploaded_file.name
        output_dir = tmp / "ocr"

        input_file.write_bytes(uploaded_file.read())
        output_dir.mkdir()

        with st.spinner("OCR đang chạy..."):

            try:
                run_chandra_cli(input_file, output_dir)

                text, tables = read_ocr_text_and_tables(output_dir)

                st.session_state.ocr_text = text
                st.session_state.ocr_tables_html = tables

                st.success("OCR xong")

            except Exception as e:
                st.error("OCR lỗi")
                st.exception(e)
