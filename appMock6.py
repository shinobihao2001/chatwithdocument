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
            text_blocks.append(
                file.read_text(encoding="utf-8", errors="ignore")
            )

        if file.suffix.lower() in [".html", ".htm"]:
            html = file.read_text(encoding="utf-8", errors="ignore")
            if "<table" in html.lower():
                html_tables.append(html)

    return "\n\n".join(text_blocks), html_tables


# ============================
# HTML TABLE → TEXT
# ============================

def table_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    lines = []
    for row in soup.find_all("tr"):
        cells = row.find_all(["th", "td"])
        values = [c.get_text(" ", strip=True) for c in cells]
        if any(values):
            lines.append(" | ".join(values))

    return "\n".join(lines)


# ============================
# ĐỌC ẢNH OCR
# ============================

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
# STREAMLIT UI
# ============================

st.set_page_config(
    page_title="OCR + Chat LLM",
    layout="wide"
)

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
# LAYOUT CHÍNH - 3 CỘT
# ============================

col1, col2, col3 = st.columns([2, 4, 4])

# ============================
# CỘT 1 - ĐIỀU KHIỂN
# ============================

with col1:
    st.markdown("### 📄 OCR + Chat LLM")
    st.markdown("<small>Chandra CLI • PDF / Image • Text-only LLM</small>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("#### 📤 Tải tài liệu")
    
    uploaded_file = st.file_uploader(
        "Chọn file",
        type=["pdf", "jpg", "jpeg", "png", "webp"],
        label_visibility="visible"
    )
    
    run_btn = st.button("🚀 Chạy OCR", use_container_width=True, type="primary")
    
    if uploaded_file:
        st.success(f"✅ Đã tải: {uploaded_file.name}")
    
    # Xử lý OCR
    if run_btn and uploaded_file:
        suffix = Path(uploaded_file.name).suffix.lower()
        
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
                    
                    st.success("✅ OCR hoàn tất")
                    
                except Exception as e:
                    st.error("❌ OCR lỗi")
                    st.exception(e)


# ============================
# CỘT 2 - HIỂN THỊ TÀI LIỆU
# ============================

with col2:
    st.markdown("#### 👁️ Xem trước tài liệu")
    
    # Container với scroll riêng
    with st.container():
        if uploaded_file:
            suffix = Path(uploaded_file.name).suffix.lower()
            
            if suffix == ".pdf":
                # CSS để tạo scroll riêng cho PDF
                st.markdown("""
                <style>
                .pdf-container {
                    height: 600px;
                    overflow-y: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 10px;
                }
                </style>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="pdf-container">', unsafe_allow_html=True)
                st.pdf(uploaded_file)
                st.markdown('</div>', unsafe_allow_html=True)
                
            else:
                # Container với scroll cho ảnh
                st.markdown("""
                <style>
                .image-container {
                    height: 600px;
                    overflow-y: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 10px;
                    text-align: center;
                }
                </style>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(uploaded_file, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("📁 Chưa có tài liệu nào được tải lên")


# ============================
# CỘT 3 - KẾT QUẢ OCR VÀ CHAT
# ============================

with col3:
    st.markdown("#### 🔍 Kết quả & Chat")
    
    tab_ocr, tab_chat = st.tabs(["📄 Kết quả OCR", "💬 Chat LLM"])
    
    # -------- TAB OCR --------
    with tab_ocr:
        # Container với scroll riêng cho OCR
        st.markdown("""
        <style>
        .ocr-container {
            height: 500px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            background-color: #fafafa;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if st.session_state.ocr_text or st.session_state.ocr_tables_html or st.session_state.ocr_images:
            st.markdown('<div class="ocr-container">', unsafe_allow_html=True)
            
            if st.session_state.ocr_text:
                st.markdown("**📝 Văn bản:**")
                st.markdown(st.session_state.ocr_text)
                st.markdown("---")
            
            if st.session_state.ocr_tables_html:
                st.markdown("**📊 Bảng:**")
                for i, html in enumerate(st.session_state.ocr_tables_html, 1):
                    st.markdown(f"*Bảng {i}:*")
                    st.markdown(html, unsafe_allow_html=True)
                    if i < len(st.session_state.ocr_tables_html):
                        st.markdown("---")
            
            if st.session_state.ocr_images:
                st.markdown("**🖼️ Hình ảnh/Chữ ký:**")
                cols = st.columns(2)
                for i, img in enumerate(st.session_state.ocr_images):
                    with cols[i % 2]:
                        st.image(
                            img["bytes"],
                            caption=img["name"],
                            use_container_width=True
                        )
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("📋 Chưa có kết quả OCR")
    
    # -------- TAB CHAT --------
    with tab_chat:
        # Container cho chat
        st.markdown("""
        <style>
        .chat-container {
            height: 350px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            background-color: #f8f9fa;
            margin-bottom: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("**💬 Trả lời:**")
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        if st.session_state.chat_answer:
            st.markdown(st.session_state.chat_answer)
        else:
            st.info("🤖 Hãy đặt câu hỏi về tài liệu...")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Input câu hỏi
        question = st.text_area(
            "Câu hỏi về tài liệu:",
            height=80,
            placeholder="Ví dụ: Văn bản này ban hành ngày nào?",
            key="question_input"
        )
        
        ask_btn = st.button("📨 Hỏi LLM", use_container_width=True, type="primary")
        
        if ask_btn and question:
            if not st.session_state.ocr_text and not st.session_state.ocr_tables_html:
                st.warning("⚠️ Vui lòng chạy OCR trước khi đặt câu hỏi!")
            else:
                with st.spinner("🤖 LLM đang suy nghĩ..."):
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
                        st.rerun()
                        
                    except Exception as e:
                        st.error("❌ LLM gặp lỗi")
                        st.exception(e)
