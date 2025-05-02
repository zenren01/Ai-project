import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms as transforms
from torchvision.utils import make_grid
import numpy as np
import math
import torch.utils.data as data

# The function make_dataset(root) takes a root directory path as input (root) and aims to create a list of image paths within that directory.
def make_dataset(root):
  images = []
  for fname in sorted(os.listdir(root)):
    path = os.path.join(root,fname)
    images.append(path)
  return images

# imgs_motion= make_dataset(motion_root)
# imgs_defo = make_dataset(defo_root)
# imgs_sharp = make_dataset(sharp_root)

# setting up parameters and transforms
import torchvision.transforms as transforms
transform_list = [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
batch_size = 1
num_workers = 2

#Creating the Custom Dataset
class CustomDataset:
  def __init__(self, features, targets, transform=None):
    self.features = features
    self.targets = targets
    transform_list = [transforms.RandomApply([transforms.RandomCrop(256)]),#crops the image of size 256 at a random location
                      transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),#normalization helps standardizes input data and makes learning easier for the model
                                                (0.5, 0.5, 0.5))]
    self.transform = transforms.Compose(transform_list)

  def __len__(self):
    return len(self.features)

  def __getitem__(self, idx):
    current_sample = self.features[idx, :]
    current_target = self.targets[idx]
    if self.transform:
      current_sample = self.transform(current_sample)
      current_target = self.transform(current_target)
    return {
        "Blurred": (current_sample),
        "Sharp": (current_target)
    }
# creating custom dataset for motion blurred-sharp images
##
custom_dataset_m = CustomDataset(features = imgs_motion_paths, targets = imgs_sharp_paths)