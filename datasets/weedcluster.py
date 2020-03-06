import os
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import numpy as np 

class WeedClusterDataset(Dataset):
    """weed_cluster class only dataset.  Use when only detecting weed clusters to minimize memory usage."""

    def __init__(self, root, split='train'):
        """
        Args:
            root (string): Directory that includes directory of images and directory of labels. Should end in '/'
        """
        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        self.split = split
        self.root = root
        self.images = os.listdir(self.root + '/' + split + '/images/rgb/')
        

        
    @classmethod
    def decode_target(cls, target):
        return target

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = torch.from_numpy(np.array(Image.open(self.root + '/' + self.split +  'images/rgb/' + self.images[index]).convert('RGB')).reshape((3,512,512)))
        # print("img shape: ", img.shape)
        target = torch.from_numpy(np.array(Image.open(self.root + '/' + self.split +  'labels/weed_cluster/' + self.images[index][:-4] + '.png')))
        # print("target shape: ", target.shape)
        return img, target

    