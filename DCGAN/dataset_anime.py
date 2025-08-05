import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import gdown
import zipfile
import shutil


class AnimeFacesDataset(Dataset):
    """Anime Faces Dataset"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # Get all image files
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
        
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {root_dir}")
        
        print(f"Found {len(self.image_paths)} anime face images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Return image and dummy label (0) for compatibility
        return image, 0


def download_anime_faces(data_dir):
    """Download Anime Faces dataset from Kaggle mirror on Google Drive"""
    # Google Drive ID for the anime faces dataset
    # Alternative mirror since original Kaggle dataset requires authentication
    file_id = "1jdJXkQIWVGOeb3gKUlKMFGVvCXlvf75P"  # This is a common mirror
    
    zip_path = os.path.join(data_dir, "anime_faces.zip")
    extract_path = os.path.join(data_dir, "anime_faces")
    
    if os.path.exists(extract_path) and len(os.listdir(extract_path)) > 0:
        print("Anime faces dataset already exists")
        return extract_path
    
    print("Downloading Anime Faces dataset...")
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # Try to download from Google Drive mirror
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, zip_path, quiet=False)
        
        # Extract the dataset
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Remove zip file
        os.remove(zip_path)
        
        # Find the extracted folder (might have different names)
        extracted_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        if extracted_folders:
            source = os.path.join(data_dir, extracted_folders[0])
            if source != extract_path:
                shutil.move(source, extract_path)
        
        return extract_path
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nPlease download the anime faces dataset manually:")
        print("1. Download from: https://www.kaggle.com/splcher/animefacedataset")
        print("2. Extract the files to:", extract_path)
        raise


def get_anime_dataloader(config):
    """
    Create DataLoader for Anime Faces dataset
    """
    transform = transforms.Compose([
        transforms.Resize(config['dataset']['image_size']),
        transforms.CenterCrop(config['dataset']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    data_dir = config['dataset']['data_dir']
    os.makedirs(data_dir, exist_ok=True)
    
    # Download dataset if needed
    anime_dir = download_anime_faces(data_dir)
    
    dataset = AnimeFacesDataset(
        root_dir=anime_dir,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers'],
        drop_last=True
    )
    
    return dataloader