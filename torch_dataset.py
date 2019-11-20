import torch
from torch.utils.data import Dataset
import PIL.Image as pil
import os

class SVHN_Dataset(Dataset):
    """Abstract PyTorch dataset class for the Street View House Numbers dataset"""

    def __init__(self, metadata, data_path, transform=None):
        
        # Metadata is an array of dicts with keys:
        # filename, label, height, width, top, left
        self.metadata = metadata
        self.data_path = data_path
        
        # Transformations to apply to each image 
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        
        # Load the image
        img = pil.open(os.path.join(os.path.join(self.data_path, self.metadata[idx]['filename'])))

        # Index the metadata array to grab this images bounding box info
        meta = self.metadata[idx]
        
        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(meta['length']), torch.tensor(meta['digits'])