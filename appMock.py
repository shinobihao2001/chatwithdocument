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


# ============================
# GỌI OLLAMA (TEXT ONLY)
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
# CHẠY CHANDRA CLI
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
# ĐỌC OCR TEXT & HTML
# ============================
def read_ocr_text_and_tables(output_dir: Path):
    text_blocks = []
    html_tables = []

    for file in sorted(output_dir.glob("**/*")):
        if file.suffix.lower() in [".md", ".txt"]:
            text_blocks.append(file.read_text(encoding="utf-8", errors="ignore"))

        if file.suffix.lower() in [".html", ".htm"]:
            html = file.read_text(encoding="utf-8", errors="ignore")
            if "<table" in html.lower():
                html_tables.append(html)

    return "\n\n".join(text_blocks), html_tables


# ============================
# HTML TABLE → TEXT CHO LLM
# ============================
def table_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.find_all("tr")

    lines = []
    for row in rows:
        cells = row.find_all(["th", "td"])
        values = [c.get_text(" ", strip=True) for c in cells]
        if any(values):
            lines.append(" | ".join(values))

    return "\n".join(lines)


# ============================
# ĐỌC ẢNH OCR → BYTES
# ============================
def read_ocr_images(output_dir: Path):
    images = []
    for file in sorted(output_dir.glob("**/*")):
        if file.suffix.lower() in {".webp", ".png", ".jpg", ".jpeg"}:
            images.append({
                "name": file.name,
                "bytes": file.read_bytes()
            })
    return images


# ============================
# STREAMLIT UI
# ============================

st.set_page_config(
    page_title="OCR PDF / Image → Chat LLM (Chandra CLI)",
    layout="wide"
)

st.title("📄 OCR PDF / Ảnh → 💬 Chat với LLM")
st.caption("Chandra CLI | PDF + JPG/JPEG/PNG/WEBP | LLM chỉ nhận text")

# --- session state
for k, v in {
    "ocr_text": "",
    "ocr_tables_html": [],
    "ocr_images": [],
    "uploaded_preview": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================
# CHIA 2 CỘT CHÍNH
# ============================

left_col, right_col = st.columns([1.1, 1.3])

# ============================
# CỘT TRÁI: UPLOAD + PREVIEW
# ============================

with left_col:
    st.subheader("📤 Upload tài liệu")

    uploaded_file = st.file_uploader(
        "Upload PDF hoặc ảnh",
        type=["pdf", "jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:
        st.session_state.uploaded_preview = uploaded_file

        st.divider()
        st.subheader("👁️ Xem tài liệu")

        suffix = Path(uploaded_file.name).suffix.lower()

        if suffix == ".pdf":
            st.pdf(uploaded_file)
        else:
            st.image(uploaded_file, use_container_width=True)

        st.divider()

        if st.button("🚀 Chạy OCR (Chandra CLI)", use_container_width=True):
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)

                input_file = tmp / f"input{suffix}"
                output_dir = tmp / "ocr_output"

                input_file.write_bytes(uploaded_file.read())
                output_dir.mkdir(exist_ok=True)

                with st.spinner("Chandra đang OCR..."):
                    try:
                        run_chandra_cli(input_file, output_dir)

                        text, tables = read_ocr_text_and_tables(output_dir)

                        st.session_state.ocr_text = text
                        st.session_state.ocr_tables_html = tables
                        st.session_state.ocr_images = read_ocr_images(output_dir)

                        st.success("OCR hoàn tất")

                    except Exception as e:
                        st.error("OCR thất bại")
                        st.exception(e)


# ============================
# CỘT PHẢI: TAB OCR + CHAT
# ============================

with right_col:

    tab_ocr, tab_chat = st.tabs(["📄 Kết quả OCR", "💬 Chat với LLM"])

    # ============================
    # TAB OCR
    # ============================
    with tab_ocr:

        if st.session_state.ocr_text:
            st.markdown("### 📄 OCR Text")
            st.markdown(st.session_state.ocr_text)

        if st.session_state.ocr_tables_html:
            st.divider()
            st.markdown("### 📊 Bảng OCR")

            for i, html in enumerate(st.session_state.ocr_tables_html, 1):
                st.markdown(f"**Bảng {i}**")
                st.markdown(html, unsafe_allow_html=True)

        if st.session_state.ocr_images:
            st.divider()
            st.markdown("### 🖼️ Con dấu / chữ ký")

            cols = st.columns(3)
            for i, img in enumerate(st.session_state.ocr_images):
                with cols[i % 3]:
                    st.image(
                        img["bytes"],
                        caption=img["name"],
                        use_container_width=True
                    )

        if not st.session_state.ocr_text:
            st.info("Chưa có dữ liệu OCR.")


    # ============================
    # TAB CHAT
    # ============================
    with tab_chat:

        if "chat_answer" not in st.session_state:
            st.session_state.chat_answer = ""

        st.markdown("### 🤖 Trả lời")

        if st.session_state.chat_answer:
            st.markdown(st.session_state.chat_answer)
        else:
            st.info("Chưa có câu trả lời.")

        st.divider()

        question = st.text_area(
            "Đặt câu hỏi về tài liệu",
            height=120,
            placeholder="Ví dụ: Văn bản này ban hành ngày nào?"
        )

        if st.button("📨 Gửi câu hỏi", use_container_width=True) and question:

            with st.spinner("LLM đang suy nghĩ..."):

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
                    answer = chat_with_ollama(llm_context, question)
                    st.session_state.chat_answer = answer
                except Exception as e:
                    st.error("LLM lỗi")
                    st.exception(e)
