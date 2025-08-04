import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import CelebA


def get_celeba_dataloader(config):
    """
    Create DataLoader for CelebA dataset
    """
    transform = transforms.Compose([
        transforms.Resize(config['dataset']['image_size']),
        transforms.CenterCrop(config['dataset']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    os.makedirs(config['dataset']['data_dir'], exist_ok=True)
    
    try:
        dataset = CelebA(
            root=config['dataset']['data_dir'],
            split='train',
            transform=transform,
            download=True
        )
    except Exception as e:
        print(f"Error loading CelebA dataset: {e}")
        print("You may need to download CelebA manually from:")
        print("http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
        print(f"Place the files in: {config['dataset']['data_dir']}")
        raise
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers'],
        drop_last=True
    )
    
    return dataloader