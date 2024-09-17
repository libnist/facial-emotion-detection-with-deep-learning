from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt

from random import choices

def get_paths(path):
    path = Path(path)
    data_paths = []
    targets = {}
    i = 0
    for root, dirs, files in path.walk():
        if len(files) > 0:
            paths = [(root/file, i) for file in files]
            data_paths += paths
            targets[i] = root.name
            i += 1
            
    return data_paths, targets


class ImageDatasetFromDirectory(Dataset):
    def __init__(self, path):
        
        self.image_paths, self._targets = get_paths(path)
    
    @property
    def targets(self):
        return self._targets
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        transform = transforms.Compose(
            [transforms.Grayscale(1),
             transforms.Resize(size=(48, 48)),
             transforms.ToTensor()]
        )
        
        image_path, image_lable = self.image_paths[idx]
        
        img = Image.open(image_path)
        
        return transform(img), torch.tensor(image_lable)
    
def draw_images(path, figsize=(10, 10)):
    
    plt.figure(figsize=figsize)
    
    paths, targets = get_paths(path)
    
    samples = choices(paths, k=9)

    for i, (img, tgt) in enumerate(samples):

        plt.subplot(3, 3, i+1)

        plt.imshow(Image.open(img), cmap="gray")
        plt.title(targets[tgt].capitalize())
        plt.axis("off")