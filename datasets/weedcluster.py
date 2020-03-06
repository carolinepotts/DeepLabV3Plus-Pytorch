import os
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

class WeedClusterDataset(Dataset):
    """weed_cluster class only dataset.  Use when only detecting weed clusters to minimize memory usage."""

    def __init__(self, root, split='train'):
        """
        Args:
            root (string): Directory that includes directory of images and directory of labels. Should end in '/'
        """
        
        self.root = root
        self.images = os.listdir(self.root + 'images/rgb/')
        self.split = split

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.root + 'images/rgb/' + self.images[index]).convert('RGB')
        target = np.array(Image.open(self.root + 'labels/weed_cluster/' + self.images[index][:-4] + '.png'))
        return img, target