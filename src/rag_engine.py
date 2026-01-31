import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1. Load biến môi trường
load_dotenv()

class RAGPipeline:
    def __init__(self):
        """Khởi tạo các thành phần của hệ thống RAG"""
        print("Đang khởi tạo RAG Pipeline...")
        self._setup_auth()
        self._setup_qdrant()
        self._setup_models()
        print("RAG Pipeline sẵn sàng")

    def _setup_auth(self):
        """Đăng nhập HuggingFace"""
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                login(token=hf_token)
                print("Đã login HuggingFace.")
            except Exception as e:
                print(f"Login HF thất bại: {e}")

    def _setup_qdrant(self):
        """Kết nối Qdrant Database"""
        self.collection_name = "qa_rag_data_science"
        try:
            self.qdrant_client = QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                timeout=60,
                check_compatibility=False
            )
            # Kiểm tra kết nối nhanh bằng cách lấy info collection
            self.qdrant_client.get_collection(self.collection_name)
            print("Kết nối Qdrant thành công.")
        except Exception as e:
            print(f"Lỗi kết nối Qdrant: {e}")
            raise e

    def _setup_models(self):
        """Load Embedding & LLM Models"""
        # 1. Embedding Model
        print("Đang load Embedding Model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large"
        )

        # 2. LLM Model
        print("Đang load LLM (Qwen2.5-1.5b-pro)...")
        model_name = "dai3107/qwen2.5-1.5b-pro"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()

        # Tạo Pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1, # Giữ thấp để chính xác
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True, # Nên để True nếu dùng temperature > 0
            return_full_text=False
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

    def retrieve_documents(self, query: str, top_k: int = 3):
        """Tìm kiếm vector trong Qdrant"""
        try:
            query_vector = self.embeddings.embed_query(f"query: {query}")
            search_results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True
            )
            
            # Lọc kết quả có điểm thấp (Thresholding)
            valid_results = [r for r in search_results.points if r.score > 0.35]
            
            if not valid_results:
                return None
                
            return valid_results
        except Exception as e:
            print(f"Lỗi Retrieve: {e}")
            return None

    def generate_answer(self, query: str):
        """Quy trình RAG hoàn chỉnh: Retrieve -> Prompt -> Generate"""
        
        # 1. Retrieve
        print(f"Đang tìm kiếm: {query}")
        results = self.retrieve_documents(query)
        
        if not results:
            return "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong cơ sở dữ liệu để trả lời câu hỏi này."

        # 2. Build Context
        context_texts = []
        for res in results:
            payload = res.payload
            # Ưu tiên lấy field 'answer' hoặc 'text', fallback các trường hợp khác
            content = payload.get("answer") or payload.get("tra_loi") or payload.get("text") or ""
            if content:
                context_texts.append(f"- {content}")
        
        context_str = "\n".join(context_texts)

        # 3. Build Prompt (Template chuẩn kỹ sư)
        prompt_template = f"""<|im_start|>system
Bạn là một trợ lý AI chuyên về Data Science. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên thông tin được cung cấp trong phần NGỮ CẢNH.
Nếu thông tin không có trong ngữ cảnh, hãy nói "Tôi không biết". Không được bịa đặt thông tin.

NGỮ CẢNH:
{context_str}
<|im_end|>
<|im_start|>user
Câu hỏi: {query}
<|im_end|>
<|im_start|>assistant
"""
        # 4. Generate
        print("Đang suy nghĩ...")
        try:
            response = self.llm.invoke(prompt_template)
            # Clean up response (đôi khi pipeline trả về cả prompt)
            if hasattr(response, "content"):
                return response.content.strip()
            return str(response).strip()
        except Exception as e:
            return f"Lỗi sinh câu trả lời: {str(e)}"

# =========================
# KHU VỰC TEST (Chỉ chạy khi chạy trực tiếp file này)
# =========================
if __name__ == "__main__":
    # Khởi tạo Pipeline
    rag = RAGPipeline()
    
    while True:
        q = input("\nMời nhập câu hỏi (gõ 'exit' để thoát): ")
        if q.lower() in ["exit", "quit"]:
            break
        
        ans = rag.generate_answer(q)
        print(f"\nTrả lời:\n{ans}\n" + "-"*50)
