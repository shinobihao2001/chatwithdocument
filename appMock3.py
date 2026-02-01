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
# STREAMLIT STYLE COMPACT
# ============================

st.set_page_config(
    page_title="OCR + Chat LLM",
    layout="wide"
)

st.markdown(
    """
<style>
/* toàn trang */
.block-container {
    padding-top: 0.6rem;
    padding-bottom: 0.5rem;
    padding-left: 1.1rem;
    padding-right: 1.1rem;
}

/* header */
h2 {
    margin-top: 0.1rem !important;
    margin-bottom: 0.3rem !important;
}

/* caption */
[data-testid="stCaptionContainer"] {
    margin-top: -0.4rem;
    margin-bottom: 0.4rem;
}

/* tabs */
[data-testid="stTabs"] {
    margin-top: 0.2rem;
}

/* uploader */
[data-testid="stFileUploader"] {
    padding-bottom: 0.2rem;
}

/* divider */
hr {
    margin: 0.4rem 0 !important;
}

/* buttons */
button {
    padding-top: 0.35rem !important;
    padding-bottom: 0.35rem !important;
}

/* text area */
textarea {
    margin-top: 0.2rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================
# HEADER GỌN
# ============================

st.markdown("## 📄 OCR tài liệu → 💬 Hỏi LLM")
st.caption("Chandra CLI • PDF / Image • Text-only LLM")


# ============================
# SESSION STATE
# ============================

for k, v in {
    "ocr_text": "",
    "ocr_tables_html": [],
    "ocr_images": [],
    "uploaded_preview": None,
    "chat_answer": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================
# CHIA 2 CỘT
# ============================

left_col, right_col = st.columns([1.1, 1.4])


# ============================
# CỘT TRÁI
# ============================

with left_col:

    st.markdown("### 📤 Tài liệu")

    upload_col, btn_col = st.columns([2.6, 1])

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload",
            type=["pdf", "jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )

    with btn_col:
        run_btn = st.button("🚀 OCR", use_container_width=True)

    if uploaded_file:
        st.session_state.uploaded_preview = uploaded_file

        suffix = Path(uploaded_file.name).suffix.lower()

        st.divider()

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
                    run_chandra_cli = lambda i, o: subprocess.run(
                        [
                            "chandra",
                            str(i),
                            str(o),
                            "--method",
                            "hf",
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True,
                    )

                    run_chandra_cli(input_file, output_dir)

                    text, tables = read_ocr_text_and_tables(output_dir)

                    st.session_state.ocr_text = text
                    st.session_state.ocr_tables_html = tables
                    st.session_state.ocr_images = read_ocr_images(output_dir)

                    st.success("Hoàn tất")

                except Exception as e:
                    st.error("OCR lỗi")
                    st.exception(e)


# ============================
# CỘT PHẢI
# ============================

with right_col:

    tab_ocr, tab_chat = st.tabs(
        ["📄 Kết quả OCR", "💬 Chat LLM"]
    )

    # -------- TAB OCR --------
    with tab_ocr:

        if st.session_state.ocr_text:
            st.markdown("#### Văn bản")
            st.markdown(st.session_state.ocr_text)

        if st.session_state.ocr_tables_html:
            st.divider()
            st.markdown("#### Bảng")

            for i, html in enumerate(
                st.session_state.ocr_tables_html, 1
            ):
                st.markdown(f"**Bảng {i}**")
                st.markdown(html, unsafe_allow_html=True)

        if st.session_state.ocr_images:
            st.divider()
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

    # -------- TAB CHAT --------
    with tab_chat:

        st.markdown("#### Trả lời")

        if st.session_state.chat_answer:
            st.markdown(st.session_state.chat_answer)

        st.divider()

        question = st.text_area(
            "Câu hỏi",
            height=95,
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


# ============================
# HÀM PHỤ
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
