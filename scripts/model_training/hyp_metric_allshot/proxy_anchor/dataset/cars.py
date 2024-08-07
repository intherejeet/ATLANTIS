# from .base import *
# import scipy.io

# class Cars(BaseDataset):
#     def __init__(self, root, mode, transform = None):
#         self.root = root + '/cars196'
#         self.mode = mode
#         self.transform = transform
#         if self.mode == 'train':
#             self.classes = range(0,98)
#         elif self.mode == 'eval':
#             self.classes = range(98,196)
                
#         BaseDataset.__init__(self, self.root, self.mode, self.transform)
#         annos_fn = 'cars_annos.mat'
#         cars = scipy.io.loadmat(os.path.join(self.root, annos_fn))
#         ys = [int(a[5][0] - 1) for a in cars['annotations'][0]]
#         im_paths = [a[0][0] for a in cars['annotations'][0]]
#         index = 0
#         for im_path, y in zip(im_paths, ys):
#             if y in self.classes: # choose only specified classes
#                 self.im_paths.append(os.path.join(self.root, im_path))
#                 self.ys.append(y)
#                 self.I += [index]
#                 index += 1

from .base import *
import random
import os
import torchvision

random.seed(42)  # Ensure consistent shuffling with a fixed seed

class Cars(BaseDataset):
    def __init__(self, root, mode, transform=None):
        # Initialize the base class first
        super(Cars, self).__init__(root, mode, transform)

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
