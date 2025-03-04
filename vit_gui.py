import os
import numpy as np
import cv2
import torch
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from transformers import ViTImageProcessor, ViTForImageClassification
import threading
import tkinter.ttk as ttk

# --- Giao diện GUI ---
root = tk.Tk()
root.title("VIT Image Search")
root.geometry("735x616")
root.configure(bg="white")
root.option_add("*Font", "Arial 12")

placeholder_img = Image.open("original.png")  # Đường dẫn ảnh mặc định
placeholder_img = placeholder_img.resize((50, 50))  # Resize ảnh cho phù hợp
placeholder_tk = ImageTk.PhotoImage(placeholder_img)

# Khung ảnh gốc (Original Image)
original_label = tk.Label(root, text="Original Image",image=placeholder_tk, bg="white",relief="solid", bd=1)
original_label.place(x=20, y=20, width=220, height=190)  # Cố định kích thước bằng pixel

# Nhãn ảnh tương tự (Similar Images)
similar_labels = []
for i in range(8):
    lbl = tk.Label(root, bg="#eeeeee")
    lbl.place(x=20 + (i % 4) * 180, y=270 + (i // 4) * 180, width=150, height=130)
    similar_labels.append(lbl)


select_button = tk.Button(root, text="Chọn ảnh", state=tk.DISABLED, bg = "#d3d3d3")
select_button.place(x=90, y=190)

tk.Label(root, text="Similar Images", font=("Arial", 13, "bold"), bg = "white").place(x=20, y=230)

loading_label = tk.Label(root, text="Đang tải mô hình...", font=("Arial", 10, "bold"), bg = "white")
loading_label.place(x=410, y=130)

#thanh tiến trình 
import tkinter.ttk as ttk

# Tạo style mới cho Progressbar
style = ttk.Style()
style.theme_use("clam")  # Dùng theme hỗ trợ đổi màu

# Đổi màu nền (background) và màu chạy (trough)
style.configure("Custom.Horizontal.TProgressbar",
                background="#696969",  # Màu chạy (Progress Color)
                troughcolor="light gray",   # Màu nền (Background Color)
                bordercolor="gray",
                lightcolor="#9c9c9c",
                darkcolor="#363636")

progress = ttk.Progressbar(root, style="Custom.Horizontal.TProgressbar", orient="horizontal", length=300, mode="determinate")
progress.place(x=330, y=100)  # Vị trí dưới loading_label

# --- Xử lý ảnh và ViT ---
dataset_dir = "images_mr"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Biến toàn cục lưu trữ dữ liệu
processor, model, preprocessed_src_images, src_images = None, None, None, None

"""def load_model():
    global processor, model, preprocessed_src_images, src_images
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)
    
    image_filenames = os.listdir(dataset_dir)[:500]
    src_images = [cv2.cvtColor(cv2.imread(os.path.join(dataset_dir, f)), cv2.COLOR_BGR2RGB) for f in image_filenames]
    preprocessed_src_images = preprocessing(src_images)
    
    select_button.config(state=tk.NORMAL, command=select_image)
    loading_label.config(text="Mô hình đã sẵn sàng!")"""

def update_progress(percent, text):
    """Cập nhật tiến trình & nhãn thông báo"""
    progress["value"] = percent
    loading_label.config(text=text)
    root.update()

def load_model():
    global processor, model, preprocessed_src_images, src_images
    
    update_progress(0, "Bắt đầu tải mô hình...")
    
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    update_progress(30, "Tải trình xử lí ảnh... (30%)")

    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)
    update_progress(60, "Tải mô hình ViT... (60%)")
    
    image_filenames = os.listdir(dataset_dir)
    src_images = [cv2.cvtColor(cv2.imread(os.path.join(dataset_dir, f)), cv2.COLOR_BGR2RGB) for f in image_filenames]
    update_progress(80, "Đọc ảnh dữ liệu... (80%)")

    preprocessed_src_images = preprocessing(src_images)
    update_progress(100, "Mô hình đã sẵn sàng!")
    
    select_button.config(state=tk.NORMAL, command=select_image)


def preprocessing(images):
    inputs = processor(images, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0, :].cpu().numpy()
    return output

def cosine_similarity(query_vector, src_vectors):
    return np.dot(src_vectors / np.linalg.norm(src_vectors, axis=1)[:, None], query_vector / np.linalg.norm(query_vector))

def ranking(preprocessed_query_image, preprocessed_src_images, top_k=8):
    scores = cosine_similarity(preprocessed_query_image, preprocessed_src_images)
    ranked_list = np.argsort(scores)[::-1][:top_k]
    return ranked_list, scores[ranked_list]

def resize_and_crop(image, target_size):
    """Resize ảnh sao cho lấp đầy target_size mà không bị méo (cover toàn bộ khung)."""
    img_w, img_h = image.size
    target_w, target_h = target_size

    # Tính toán tỷ lệ phóng to để chiều ngang luôn bằng target_w
    scale = max(target_w / img_w, target_h / img_h)  # Chọn scale lớn hơn để cover toàn bộ khung
    new_size = (int(img_w * scale), int(img_h * scale))  # Kích thước sau khi phóng to

    # Resize ảnh
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Cắt ảnh để đúng kích thước khung
    left = (new_size[0] - target_w) / 2
    top = (new_size[1] - target_h) / 2
    right = left + target_w
    bottom = top + target_h

    return image.crop((left, top, right, bottom))


def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        query_image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        preprocessed_query_image = preprocessing(query_image).squeeze(0)
        ranked_list, scores = ranking(preprocessed_query_image, preprocessed_src_images, top_k=8)
        
        # Resize và hiển thị ảnh gốc (Original Image)
        img = Image.fromarray(query_image)
        img = resize_and_crop(img, (250, 250))  # Resize để lấp đầy chiều ngang khung
        img_tk = ImageTk.PhotoImage(img)
        original_label.config(image=img_tk)
        original_label.image = img_tk

        # Hiển thị ảnh tương tự (Similar Images)
        for idx, widget in enumerate(similar_labels):
            src_idx = ranked_list[idx]
            img = Image.fromarray(src_images[src_idx])
            img = resize_and_crop(img, (150, 150))  # Resize để lấp đầy chiều ngang khung
            img_tk = ImageTk.PhotoImage(img)
            widget.config(image=img_tk, text=f"Sim: {scores[idx]:.2f}")
            widget.image = img_tk



# Chạy luồng tải mô hình riêng biệt để không làm treo giao diện
threading.Thread(target=load_model, daemon=True).start()

root.mainloop()
