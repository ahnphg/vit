import os
import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from torch.utils.data import Dataset, DataLoader

# Dataset
dataset_dir = "animals"
image_filenames = os.listdir(dataset_dir)[:5000]
batch_size = 32
features_file = 'features.h5'

# Hàm trích xuất đặc trưng (dùng lại nếu cần)
def preprocessing(images, processor, model, device):
    inputs = processor(images, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0, :].detach().cpu().numpy()
    return output

# Hàm trích xuất đặc trưng nếu không có features.h5
def extract_features_if_needed():
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
                raise ValueError(f"Cannot load image: {filepath}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            return image

    # Load mô hình
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)

    # Tạo Dataset và DataLoader
    dataset = ImageDataset(image_filenames, dataset_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Trích xuất và lưu đặc trưng
    print(f"Extracting features to {features_file}...")
    with h5py.File(features_file, 'w') as f:
        dset = f.create_dataset('features', shape=(len(image_filenames), 768), dtype=np.float32)
        for i, batch_images in enumerate(dataloader):
            features = preprocessing(batch_images, processor, model, device)
            dset[i * batch_size:(i + 1) * batch_size] = features
    print(f"Features saved to {features_file}")
    return processor, model, device

# Load đặc trưng
try:
    with h5py.File(features_file, 'r') as f:
        preprocessed_src_images = f['features'][:]
    print(f"Loaded features from {features_file}")
    # Load mô hình cho query
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)
except (OSError, KeyError, FileNotFoundError):
    print(f"{features_file} not found or corrupted. Extracting features...")
    processor, model, device = extract_features_if_needed()
    with h5py.File(features_file, 'r') as f:
        preprocessed_src_images = f['features'][:]

# Hàm cosine similarity
def cosine_similarity(query_vector, src_vectors):
    query_norm = np.linalg.norm(query_vector)
    normalized_query = query_vector / query_norm
    src_norms = np.linalg.norm(src_vectors, axis=1)
    normalized_src = src_vectors / src_norms[:, np.newaxis]
    return np.dot(normalized_src, normalized_query)

# Hàm ranking
def ranking(preprocessed_query_image, src_features, top_k=10):
    scores = cosine_similarity(preprocessed_query_image, src_features)
    ranked_list = np.argsort(scores)[::-1][:top_k]
    scores = scores[ranked_list]
    return ranked_list, scores

# Query processing
query_image_paths = [r".\animals\0bcf59f1ca.jpg"]
top_k = 10

for query_image_path in query_image_paths:
    if not os.path.exists(query_image_path):
        print(f"Error: File {query_image_path} does not exist")
        continue

    query_image = cv2.imread(query_image_path)
    if query_image is None:
        print(f"Error: Cannot load image at {query_image_path}")
        continue

    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    query_image = cv2.resize(query_image, (224, 224))
    preprocessed_query_image = preprocessing([query_image], processor, model, device).squeeze(0)

    ranked_list, scores = ranking(preprocessed_query_image, preprocessed_src_images, top_k=top_k)

    print("Query Image")
    plt.figure(figsize=(3, 3))
    plt.imshow(query_image)
    plt.axis("off")
    plt.show()

    print(f"Top {top_k} results")
    for idx, score in zip(ranked_list, scores):
        src_image = cv2.imread(os.path.join(dataset_dir, image_filenames[idx]))
        if src_image is None:
            print(f"Error: Cannot load source image {image_filenames[idx]}")
            continue
        src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(3, 3))
        plt.imshow(src_image)
        plt.title(f"Similarity: {score:.4f}", fontsize=10)
        plt.axis("off")
        plt.show()