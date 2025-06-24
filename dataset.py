import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class carData(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        #self.images = os.listdir(img_dir) #masks has same names
    
    def __len__(self):
        return len(self.mask_dir)
    
    def __getitem__(self, index):
        #img_path = os.path.join(self.img_dir, self.images[index])
        #mask_path = os.path.join(self.img_dir, self.images[index])
        image = np.array(Image.open(self.img_dir[index]).convert("RGB")) #RGB probably default
        #mask = np.array(Image.open(self.img_dir[index]).convert("L"), dtype=np.float32) #grayscale? maybe float32?
        mask = np.array(Image.open(self.mask_dir[index]))
        
        #augment data for both image and mask if transform is not None
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()

        return image, mask



"""
# src/datamodules.py
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CarDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.images = sorted((self.root / "images").glob("*.png"))
        self.transform = transform

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = img_path.with_name(img_path.stem + ".png").with_parent(
            img_path.parents[1] / "masks")
        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path)          # mode 'L', classes 0‑4

        if self.transform:
            augmented = self.transform(image=np.array(image),
                                       mask=np.array(mask))
            image, mask = augmented["image"], augmented["mask"]

        return image.float()/255, mask.long()   # normalised img, raw mask

def get_dataloaders(root, val_frac=0.15, test_frac=0.15, batch=8, seed=42):
    full = CarDataset(root, transform=train_tfms())  # will be stabbed later
    n = len(full)
    n_test  = int(n * test_frac)
    n_val   = int(n * val_frac)
    n_train = n - n_val - n_test
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(full, [n_train, n_val, n_test],
                                             generator=g)

    # Freeze augments for val/test
    val_ds.dataset.transform  = val_tfms()
    test_ds.dataset.transform = val_ds.dataset.transform

    return (DataLoader(train_ds, batch, shuffle=True,  num_workers=4, pin_memory=True),
            DataLoader(val_ds,   batch, shuffle=False, num_workers=4),
            DataLoader(test_ds,  batch, shuffle=False, num_workers=4))

"""