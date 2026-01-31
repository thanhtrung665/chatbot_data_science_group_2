import streamlit as st
import requests
from datetime import datetime
import uuid
from chain import get_rag_response

# =========================
# 1. CẤU HÌNH & CSS
st.set_page_config(
    page_title="Học Data Science cùng AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme CSS
CUSTOM_CSS = """
<style>
    /* Nền tổng thể: Dark Gradient */
    .stApp {
        background: linear-gradient(180deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: #ffffff;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(17, 25, 40, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Input Box */
    .stChatInput textarea {
        background-color: #1e293b !important;
        color: white !important;
        border: 1px solid #334155 !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e2e8f0 !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Custom Button Style */
    div.stButton > button {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(75, 108, 183, 0.5);
    }

    /* Ẩn Header mặc định của Streamlit */
    header[data-testid="stHeader"] {background: transparent;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================
# 2. QUẢN LÝ SESSION STATE
def init_session():
    defaults = {
        "authenticated": False,
        "username": "",
        "chats": {},  # Dùng Dict thay vì List để truy xuất nhanh hơn theo ID
        "current_chat_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session()

# =========================
# 3. HÀM XỬ LÝ CHAT & RAG API

def create_new_chat():
    """Tạo một phiên chat mới"""
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = {
        "title": f"New Chat {datetime.now().strftime('%H:%M')}",
        "messages": [],
        "created_at": datetime.now()
    }
    st.session_state.current_chat_id = new_id
    return new_id

def delete_chat(chat_id):
    """Xóa phiên chat"""
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
    if st.session_state.current_chat_id == chat_id:
        st.session_state.current_chat_id = None

def call_rag_api(query):
    try:
        response_text = get_rag_response(query)
        return response_text
        
    except Exception as e:
        return f"Lỗi xử lý RAG: {str(e)}"

# =========================
# 4. GIAO DIỆN ĐĂNG NHẬP
def login_ui():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.title("🔐 ACCESS CONTROL")
        
        with st.form("login_form"):
            user = st.text_input("Username", placeholder="admin")
            pwd = st.text_input("Password", type="password", placeholder="123")
            submitted = st.form_submit_button("Authenticate System")
            
            if submitted:
                if user == "admin" and pwd == "123":
                    st.session_state.authenticated = True
                    st.session_state.username = user
                    if not st.session_state.chats:
                        create_new_chat()
                    st.rerun()
                else:
                    st.error("Access Denied.")

# =========================
# 5. GIAO DIỆN CHÍNH
def main_ui():
    # --- SIDEBAR (LỊCH SỬ CHAT) ---
    with st.sidebar:
        st.title(f"👤 {st.session_state.username}")
        st.divider()
        
        if st.button("+ New Thread", use_container_width=True):
            create_new_chat()
            st.rerun()
            
        st.subheader("History")
        
        # Sắp xếp chat mới nhất lên đầu
        sorted_chats = sorted(
            st.session_state.chats.items(), 
            key=lambda x: x[1]['created_at'], 
            reverse=True
        )

        for chat_id, chat_data in sorted_chats:
            col_btn, col_del = st.columns([5, 1])
            
            # Highlight chat đang chọn
            is_active = (chat_id == st.session_state.current_chat_id)
            btn_style = "primary" if is_active else "secondary"
            
            with col_btn:
                if st.button(f"💬 {chat_data['title']}", key=f"btn_{chat_id}", type=btn_style, use_container_width=True):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()
            
            with col_del:
                if st.button("✕", key=f"del_{chat_id}", help="Delete"):
                    delete_chat(chat_id)
                    st.rerun()

        st.divider()
        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

    # --- MAIN CHAT AREA ---
    
    # Kiểm tra nếu chưa có chat nào được chọn
    if not st.session_state.current_chat_id:
        create_new_chat() # Tự động tạo nếu trống
    
    current_id = st.session_state.current_chat_id
    current_messages = st.session_state.chats[current_id]["messages"]

    st.header("⚡ LLMOps RAG Assistant")
    
    # Hiển thị tin nhắn (Sử dụng native st.chat_message cho đẹp và chuẩn)
    for msg in current_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input Area
    if prompt := st.chat_input("Nhập câu hỏi chuyên sâu về dữ liệu..."):
        # 1. Hiển thị user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 2. Lưu user message
        st.session_state.chats[current_id]["messages"].append({"role": "user", "content": prompt})
        
        # 3. Cập nhật title nếu là tin nhắn đầu tiên
        if len(current_messages) == 1:
             # Lấy 30 ký tự đầu làm title
            st.session_state.chats[current_id]["title"] = prompt[:30] + "..."

        # 4. Xử lý AI Response
        with st.chat_message("assistant"):
            with st.spinner("Processing Logic..."):
                response_text = call_rag_api(prompt)
                st.markdown(response_text)
        
        # 5. Lưu bot message
        st.session_state.chats[current_id]["messages"].append({"role": "assistant", "content": response_text})
        
        # Rerun để cập nhật title bên sidebar nếu cần (optional)
        # st.rerun() 

# =========================
# 6. APP ENTRY POINT
if __name__ == "__main__":
    if st.session_state.authenticated:
        main_ui()
    else:
        login_ui()