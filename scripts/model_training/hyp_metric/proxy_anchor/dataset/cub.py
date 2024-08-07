from .base import *
    
class CUBirds(BaseDataset):
    def __init__(self, root, mode, transform=None):
        self.root = root + '/CUB_200_2011'
        self.mode = mode
        self.transform = transform
        # Fetch all class labels once to determine total number of classes
        all_class_labels = [i[1] for i in torchvision.datasets.ImageFolder(root=os.path.join(self.root, 'images')).imgs]
        total_num_classes = max(all_class_labels) + 1  # Assuming labels are 0-indexed

        self.test_num_classes = 100  # fixed as per your condition
        self.train_num_classes = total_num_classes - self.test_num_classes  # dynamically calculated
        if self.mode == 'train':
            self.classes = range(0, self.train_num_classes)
        elif self.mode == 'eval':
            self.classes = range(self.train_num_classes, self.train_num_classes+100)
        
        # Initialize the base class
        BaseDataset.__init__(self, self.root, self.mode, self.transform)

        index = 0
        for i in torchvision.datasets.ImageFolder(root=os.path.join(self.root, 'images')).imgs:
            y = i[1]
            fn = os.path.split(i[0])[1]
            if y in self.classes and fn[:2] != '._':
                self.ys += [y]
                self.I += [index]
                self.im_paths.append(os.path.join(self.root, i[0]))
                index += 1
            
    def get_image_paths(self):
        return self.im_paths[:], self.ys[:]
