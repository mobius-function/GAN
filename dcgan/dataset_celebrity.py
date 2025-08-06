import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import requests
import json
from io import BytesIO


class CelebrityFacesDataset(Dataset):
    """Celebrity Faces Dataset from Hugging Face"""

    def __init__(self, transform=None, max_images=None, split="train", train_ratio=0.8):
        self.transform = transform
        self.max_images = max_images
        self.split = split
        self.train_ratio = train_ratio

        # Get dataset info first
        print(f"Loading celebrity faces dataset from Hugging Face ({split} split)...")
        self._load_dataset_info()

        # Split into train/validation
        self._split_dataset()

        print(f"Found {len(self.image_urls)} celebrity face images for {split} split")

    def _load_dataset_info(self):
        """Load dataset info from HF API"""
        base_url = "https://datasets-server.huggingface.co/rows"
        dataset = "ares1123/celebrity_dataset"

        self.all_image_urls = []
        offset = 0
        batch_size = 100

        while True:
            url = f"{base_url}?dataset={dataset}&config=default&split=train&offset={offset}&length={batch_size}"

            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()

                if not data.get("rows"):
                    break

                # Extract image URLs
                for row in data["rows"]:
                    img_url = row["row"]["image"]["src"]
                    self.all_image_urls.append(img_url)

                    if self.max_images and len(self.all_image_urls) >= self.max_images:
                        self.all_image_urls = self.all_image_urls[:self.max_images]
                        return

                offset += batch_size

                # If we got less than batch_size, we're done
                if len(data["rows"]) < batch_size:
                    break

            except Exception as e:
                print(f"Error loading batch at offset {offset}: {e}")
                break
    
    def _split_dataset(self):
        """Split dataset into train and validation sets"""
        n_total = len(self.all_image_urls)
        n_train = int(n_total * self.train_ratio)
        
        if self.split == "train":
            self.image_urls = self.all_image_urls[:n_train]
        elif self.split == "val" or self.split == "validation":
            self.image_urls = self.all_image_urls[n_train:]
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'val'/")

    def __len__(self):
        return len(self.image_urls)

    def __getitem__(self, idx):
        img_url = self.image_urls[idx]

        try:
            # Download image
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()

            # Open image from bytes
            image = Image.open(BytesIO(response.content)).convert("RGB")

            if self.transform:
                image = self.transform(image)

            # Return image and dummy label (0) for compatibility
            return image, 0

        except Exception as e:
            print(f"Error loading image {idx} from {img_url}: {e}")
            # Return a black image as fallback
            image = Image.new("RGB", (256, 256), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, 0


def check_hf_dataset():
    """Check if Hugging Face dataset is accessible"""
    try:
        url = "https://datasets-server.huggingface.co/rows?dataset=ares1123%2Fcelebrity_dataset&config=default&split=train&offset=0&length=1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return len(data.get("rows", [])) > 0
    except Exception as e:
        print(f"Error checking HF dataset: {e}")
        return False


def get_dataloader(config):
    """
    Create DataLoader for Celebrity Faces dataset from Hugging Face
    """
    transform = transforms.Compose(
        [
            transforms.Resize(config["dataset"]["image_size"]),
            transforms.CenterCrop(config["dataset"]["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Check if HF dataset is accessible
    if not check_hf_dataset():
        raise RuntimeError("Cannot access Hugging Face celebrity dataset")

    # Limit dataset size for faster training (optional)
    max_images = config["dataset"].get("max_images", None)

    dataset = CelebrityFacesDataset(transform=transform, max_images=max_images)

    dataloader = DataLoader(
        dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
        drop_last=True,
    )

    return dataloader
