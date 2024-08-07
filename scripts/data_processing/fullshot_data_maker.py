import os
import shutil
import random
from pathlib import Path
import torchvision

def split_dataset(source_dir, dest_dir, train_ratio=0.5):
    """
    Split dataset into train and test sets, maintaining class subdirectories, 
    and save to separate 'train' and 'test' directories.

    :param source_dir: Directory containing the original dataset.
    :param dest_dir: Base directory where train and test directories will be created.
    :param train_ratio: Ratio of data to be used for training (default is 0.5).
    """
    random.seed(21)  # For reproducibility

    # Create train and test directories
    train_dir = Path(dest_dir) / 'train'
    test_dir = Path(dest_dir) / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = torchvision.datasets.ImageFolder(root=source_dir)
    class_images = {class_id: [] for class_id in range(len(dataset.classes))}
    for img_path, class_id in dataset.imgs:
        class_images[class_id].append(img_path)

    # Create class subdirectories and split images
    for class_id, images in class_images.items():
        class_name = dataset.classes[class_id]
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (test_dir / class_name).mkdir(parents=True, exist_ok=True)

        random.shuffle(images)
        split_point = int(len(images) * train_ratio)
        train_images = images[:split_point]
        test_images = images[split_point:]

        for img in train_images:
            shutil.copy(img, train_dir / class_name / Path(img).name)
        for img in test_images:
            shutil.copy(img, test_dir / class_name / Path(img).name)

    print(f"Dataset split complete. Train data: {train_dir}, Test data: {test_dir}")

if __name__ == "__main__":
    source_directory = "path/to/original/dataset"  # Replace with the path to your dataset
    destination_directory = "path/to/destination/directory"  # Replace with your destination path
    split_dataset(source_directory, destination_directory)
