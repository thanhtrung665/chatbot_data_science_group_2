# chain.py (Dùng cho Streamlit)
import streamlit as st
from rag_engine import RAGPipeline # Import class vừa viết

@st.cache_resource
def get_rag_pipeline():
    # Hàm này chỉ chạy 1 lần duy nhất lúc khởi động app
    return RAGPipeline()

def get_rag_response(query_text):
    rag = get_rag_pipeline()
    return rag.generate_answer(query_text)