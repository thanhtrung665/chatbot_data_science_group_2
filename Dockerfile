# 1. Chọn Base Image: Dùng Python 3.10 bản slim cho nhẹ
FROM python:3.10-slim

# 2. Thiết lập thư mục làm việc trong Container
WORKDIR /app

# 3. Cài đặt các thư viện hệ thống cần thiết (gcc để biên dịch một số lib python)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy file requirements trước để tận dụng Docker Cache
COPY requirements.txt .

# 5. Cài đặt thư viện Python (Thêm --no-cache-dir để giảm dung lượng image)
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy toàn bộ code vào container (trừ những file trong .dockerignore)
COPY . .

# 7. Mở cổng 8501 (Cổng mặc định của Streamlit)
EXPOSE 8501

# 8. Kiểm tra sức khỏe ứng dụng (Healthcheck)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 9. Lệnh chạy ứng dụng
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]