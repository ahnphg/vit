import os
import sys
import numpy as np
import cv2
import h5py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from torch.utils.data import Dataset, DataLoader

# Đặt mã hóa UTF-8 cho console trên Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class ImageSearchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng Dụng Tìm Kiếm Ảnh")
        self.root.geometry("1000x600")

        # Dataset settings
        self.dataset_dir = "animals"
        # Lấy tất cả tệp ảnh, không giới hạn
        self.image_filenames = [f for f in os.listdir(self.dataset_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.batch_size = 32
        self.features_file = 'features.h5'
        self.top_k = 10

        # Load model and features
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(self.device)
        self.preprocessed_src_images = self.load_or_extract_features()

        # GUI components
        self.query_image_path = None
        self.query_image_label = tk.Label(self.root, text="Chưa chọn ảnh")
        self.query_image_label.pack(pady=10)

        self.image_display = tk.Label(self.root)
        self.image_display.pack(pady=10)

        self.browse_button = tk.Button(self.root, text="Chọn Ảnh", command=self.browse_image)
        self.browse_button.pack(pady=5)

        self.search_button = tk.Button(self.root, text="Tìm Ảnh Tương Tự", command=self.search_images)
        self.search_button.pack(pady=5)

        self.result_frame = tk.Frame(self.root)
        self.result_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.result_labels = []

    def load_or_extract_features(self):
        try:
            with h5py.File(self.features_file, 'r') as f:
                features = f['features'][:]
            print(f"Đã nạp đặc trưng từ {self.features_file}")
            # Kiểm tra đồng bộ hóa
            if len(features) != len(self.image_filenames):
                print(f"Cảnh báo: Số đặc trưng ({len(features)}) không khớp với số ảnh ({len(self.image_filenames)}). Tái tạo đặc trưng...")
                return self.extract_features()
            return features
        except (OSError, KeyError, FileNotFoundError):
            print(f"{self.features_file} không tìm thấy hoặc bị lỗi. Tạo đặc trưng mới...")
            return self.extract_features()

    def extract_features(self):
        class ImageDataset(Dataset):
            def __init__(self, image_paths, dataset_dir):
                self.image_paths = image_paths
                self.dataset_dir = dataset_dir

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                filepath = os.path.join(self.dataset_dir, self.image_paths[idx])
                image = cv2.imread(filepath)
                if image is None:
                    raise ValueError(f"Không thể nạp ảnh: {filepath}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = prel.image = cv2.resize(image, (224, 224))
                return image

        dataset = ImageDataset(self.image_filenames, self.dataset_dir)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        print(f"Đang tạo đặc trưng và lưu vào {self.features_file}...")
        with h5py.File(self.features_file, 'w') as f:
            dset = f.create_dataset('features', shape=(len(self.image_filenames), 768), dtype=np.float32)
            for i, batch_images in enumerate(dataloader):
                features = self.preprocessing(batch_images)
                dset[i * self.batch_size:(i + 1) * self.batch_size] = features
        print(f"Đặc trưng đã lưu vào {self.features_file}")
        with h5py.File(self.features_file, 'r') as f:
            return f['features'][:]

    def preprocessing(self, images):
        inputs = self.processor(images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0, :].detach().cpu().numpy()
        return output

    def cosine_similarity(self, query_vector, src_vectors):
        query_norm = np.linalg.norm(query_vector)
        normalized_query = query_vector / query_norm
        src_norms = np.linalg.norm(src_vectors, axis=1)
        normalized_src = src_vectors / src_norms[:, np.newaxis]
        return np.dot(normalized_src, normalized_query)

    def ranking(self, preprocessed_query_image, src_features):
        scores = self.cosine_similarity(preprocessed_query_image, src_features)
        ranked_list = np.argsort(scores)[::-1][:self.top_k]
        scores = scores[ranked_list]
        return ranked_list, scores

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.query_image_path = file_path
            self.query_image_label.config(text=f"Đã chọn: {os.path.basename(file_path)}")
            self.display_query_image(file_path)

    def display_query_image(self, file_path):
        image = Image.open(file_path)
        image = image.resize((200, 200), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.image_display.config(image=photo)
        self.image_display.image = photo  # Giữ tham chiếu để tránh garbage collection

    def search_images(self):
        if not self.query_image_path:
            messagebox.showerror("Lỗi", "Vui lòng chọn một ảnh trước!")
            return

        # Xóa kết quả trước đó
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        self.result_labels.clear()

        # Nạp và tiền xử lý ảnh truy vấn
        query_image = cv2.imread(self.query_image_path)
        if query_image is None:
            messagebox.showerror("Lỗi", f"Không thể nạp ảnh tại {self.query_image_path}")
            return

        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
        query_image = cv2.resize(query_image, (224, 224))
        preprocessed_query_image = self.preprocessing([query_image]).squeeze(0)

        # Thực hiện xếp hạng
        ranked_list, scores = self.ranking(preprocessed_query_image, self.preprocessed_src_images)

        # Hiển thị kết quả
        tk.Label(self.result_frame, text=f"Top {self.top_k} ảnh tương tự", font=("Arial", 12, "bold")).pack(pady=5)

        canvas = tk.Canvas(self.result_frame)
        scrollbar = tk.Scrollbar(self.result_frame, orient="vertical", command=canvas.yview, width=20)  # Tăng chiều rộng thanh cuộn
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Tạo lưới 2 hàng x 5 cột
        for i in range(2):  # 2 hàng
            for j in range(5):  # 5 cột
                index = i * 5 + j
                if index >= len(ranked_list) or index >= self.top_k:
                    break  # Thoát nếu không còn ảnh để hiển thị

                idx = ranked_list[index]
                score = scores[index]

                # Kiểm tra chỉ số hợp lệ
                if idx >= len(self.image_filenames):
                    print(f"Cảnh báo: Chỉ số {idx} vượt quá danh sách ảnh ({len(self.image_filenames)})")
                    continue
                src_image_path = os.path.join(self.dataset_dir, self.image_filenames[idx])
                src_image = cv2.imread(src_image_path)
                if src_image is None:
                    print(f"Lỗi: Không thể nạp ảnh nguồn {self.image_filenames[idx]}")
                    continue
                src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)

                # Chuyển sang PIL Image để hiển thị
                src_image_pil = Image.fromarray(src_image)
                src_image_pil = src_image_pil.resize((150, 150), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(src_image_pil)

                # Tạo frame cho mỗi ô lưới
                frame = tk.Frame(scrollable_frame)
                frame.grid(row=i, column=j, padx=10, pady=5, sticky="nsew")

                label = tk.Label(frame, image=photo)
                label.image = photo  # Giữ tham chiếu
                label.pack()

                tk.Label(frame, text=f"Độ tương đồng: {score:.4f}").pack()
                self.result_labels.append(label)

        # Cấu hình trọng số cho lưới để căn chỉnh đều
        for i in range(2):
            scrollable_frame.grid_rowconfigure(i, weight=1)
        for j in range(5):
            scrollable_frame.grid_columnconfigure(j, weight=1)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSearchGUI(root)
    app.run()