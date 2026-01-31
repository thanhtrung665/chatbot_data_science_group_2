# Xuất markdown
import fitz 
import pytesseract
from pdf2image import convert_from_path
import numpy as np
import cv2
import os

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

def process_smart_pdf(pdf_path, output_markdown):
    doc = fitz.open(pdf_path)
    with open(output_markdown, "w", encoding="utf-8") as f:
        # Ghi tiêu đề
        file_name = os.path.basename(pdf_path)
        f.write(f"# Nội dung tài liệu LLM Handbook\n")
        f.write(f"** Tổng số trang:** {len(doc)}\n\n---\n\n")
        
        print(f'Đang xử lý file: {pdf_path} có {len(doc)} trang')
        
        for i, page in enumerate(doc):
            print(f"Đang check trang {i+1}")
            text = page.get_text()
            
            # Ghi Header từng trang
            f.write(f"## Trang {i+1}\n\n")
            
            # Logic Hybrid
            if len(text.strip()) > 50:
                print(f"[Native] Trang {i+1}")
                f.write("**[Nguồn: Text gốc]**\n\n")
                f.write(text)
            else:
                print(f"[OCR] Trang {i+1}")
                # render ảnh
                pix = page.get_pixmap(dpi=300)
                img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                
                if pix.n == 3:
                    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                else:
                    img_cv = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                    
                # Preprocess & OCR
                processed_img = preprocess_image(img_cv)
                try:
                    ocr_text = pytesseract.image_to_string(processed_img, lang='vie+eng', config='--psm 6')
                except pytesseract.TesseractError:
                    print(" Chưa có ngôn ngữ 'vie' chuyển sang 'eng'")
                    ocr_text = pytesseract.image_to_string(processed_img, lang='eng', config='--psm 6')
                
                f.write("**[Nguồn: OCR Scan]**\n\n")
                f.write(ocr_text) # Ghi text OCR vào file
            f.write("\n\n---\n\n")
            # Thêm dòng kẻ ngang phân cách các trang trong markdown
        print(f"Đã xuất kết quả ra file : {output_markdown}")

# ==== Chạy chương trình ====
pdf_path = r'chatbot_DAKHDL/data/LLM Engineers Handbook.pdf'
output_markdown = 'LLM_Handbook_Extracted.md'
process_smart_pdf(pdf_path, output_markdown)