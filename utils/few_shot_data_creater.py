import os
import shutil
from torchvision import datasets
from collections import defaultdict

num_imgs = 3

# Function to create a few-shot subset for first 100 classes and full set for others
def create_subset(src_folder, dest_folder, num_imgs):
    dataset = datasets.ImageFolder(src_folder)
    class_to_idx = dataset.class_to_idx
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    class_counter = defaultdict(int)
    
    for img_path, label in dataset.imgs:
        class_name = idx_to_class[label]

        # Create destination class folder if it doesn't exist
        dest_class_folder = os.path.join(dest_folder, class_name)
        os.makedirs(dest_class_folder, exist_ok=True)

        # For first 100 classes, limit to num_imgs images each
        if label < 100:
            if class_counter[class_name] < num_imgs:
                dest_img_path = os.path.join(dest_class_folder, os.path.basename(img_path))
                shutil.copy(img_path, dest_img_path)
                class_counter[class_name] += 1
        # For remaining classes, copy all images
        else:
            dest_img_path = os.path.join(dest_class_folder, os.path.basename(img_path))
            shutil.copy(img_path, dest_img_path)

# Source and destination directories
src_folder = "/path/to/datasets/orig/CUB_200_2011/images"
dest_folder = f"/path/to/datasets/fewshot/{num_imgs}/CUB_200_2011/images"

# Create the subset
create_subset(src_folder, dest_folder, num_imgs)
