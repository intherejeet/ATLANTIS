from .base import *
import random
import os
import torchvision

random.seed(42)  # Ensure consistent shuffling with a fixed seed

class CUBirds(BaseDataset):
    def __init__(self, root, mode, transform=None):
        # Initialize the base class first
        super(CUBirds, self).__init__(root, mode, transform)

        self.root = root
        self.mode = mode
        self.transform = transform

        # Load dataset
        dataset = torchvision.datasets.ImageFolder(root=self.root)
        self.ys = [y for _, y in dataset.imgs]
        self.I = list(range(len(dataset.imgs)))
        self.im_paths = [img_path for img_path, _ in dataset.imgs]

    def get_image_paths(self):
        return self.im_paths[:], self.ys[:]
