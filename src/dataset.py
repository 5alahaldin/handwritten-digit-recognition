import os
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DigitDataset(Dataset):
  def __init__(self, root_dir, transform=None):
    self.root_dir = root_dir
    self.transform = transform if transform else transforms.ToTensor()
    self.image_paths = []
    self.labels = []

    for label in range(10):
      class_dir = os.path.join(root_dir, str(label))
      if not os.path.exists(class_dir):
        continue
      for filename in os.listdir(class_dir):
        if filename.endswith('.png'):
          self.image_paths.append(os.path.join(class_dir, filename))
          self.labels.append(label)

    if not self.image_paths:
      raise RuntimeError(f"No images found in {root_dir}.")

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    image_path = self.image_paths[idx]
    label = self.labels[idx]
    try:
      image = Image.open(image_path).convert('L')
      if self.transform:
        image = self.transform(image)
      return image, label
    except UnidentifiedImageError:
      raise ValueError(f"Unrecognized image format: {image_path}")
    except Exception as e:
      raise RuntimeError(f"Failed to load {image_path}: {e}")
