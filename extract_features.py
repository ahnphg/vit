import os
import numpy as np
import cv2
import h5py
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from torch.utils.data import Dataset, DataLoader

# Dataset
dataset_dir = "animals"
image_filenames = os.listdir(dataset_dir)
batch_size = 32

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

# Hàm trích xuất đặc trưng
def preprocessing(images):
    inputs = processor(images, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0, :].detach().cpu().numpy()
    return output

# Tạo Dataset và DataLoader
dataset = ImageDataset(image_filenames, dataset_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Trích xuất và lưu đặc trưng
features_file = 'features.h5'
print(f"Extracting features to {features_file}...")
with h5py.File(features_file, 'w') as f:
    dset = f.create_dataset('features', shape=(len(image_filenames), 768), dtype=np.float32)
    for i, batch_images in enumerate(dataloader):
        features = preprocessing(batch_images)
        dset[i * batch_size:(i + 1) * batch_size] = features
print(f"Features saved to {features_file}")