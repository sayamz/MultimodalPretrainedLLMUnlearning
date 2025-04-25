import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset, default_collate
import requests
from io import BytesIO
import torch
import os

class LAIONDataset(Dataset):
    def __init__(self, csv_file, preprocess, forget_set=False, retain_set=False):
        self.data = pd.read_csv(csv_file)
        self.preprocess = preprocess
        
        # Use 5% of the dataset as forget set and another 5% as retain set for testing after unlearning
        if forget_set:
            self.data = self.data.sample(frac=0.01, random_state=1)
        elif retain_set:
            self.data = self.data.sample(frac=0.02, random_state=2)
        else:
            self.data = self.data.sample(frac=0.1, random_state=3)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_url = row['URL']
        text = row['TEXT']
        
        # Load and preprocess the image
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            image = Image.open(BytesIO(response.content))
            image = self.preprocess(image)
        except (UnidentifiedImageError, requests.exceptions.RequestException):
            # Handle the case where the image cannot be identified or request fails
            image = None
            
        return image, text

def custom_collate(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return torch.tensor([]), []
    images, texts = zip(*batch)
    return default_collate(images), list(texts)
