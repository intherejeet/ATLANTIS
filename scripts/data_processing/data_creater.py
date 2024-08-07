import os
import shutil
import random
from torchvision import datasets
from collections import defaultdict

def get_min_images(src, classes):
    min_images = float("inf")
    for class_name in classes:
        src_class_path = os.path.join(src, class_name)
        if os.path.exists(src_class_path):
            num_images = len(os.listdir(src_class_path))
            min_images = min(min_images, num_images)
    return min_images

def merge_datasets(src1, src2, dest, num_imgs_syn=None, num_imgs_orig=None):
    os.makedirs(dest, exist_ok=True)
    
    dataset1 = datasets.ImageFolder(src1)
    dataset2 = datasets.ImageFolder(src2)

    class_counter = defaultdict(int)

    classes1 = dataset1.classes[:98]
    classes2 = dataset2.classes[:98]
    remaining_classes2 = dataset2.classes[98:]

    min_images = get_min_images(src1, classes1) if num_imgs_syn == "min" else None

    for class_name in set(classes1 + classes2):
        dest_class_path = os.path.join(dest, class_name)
        os.makedirs(dest_class_path, exist_ok=True)

        for src, classes, num_imgs in [(src1, classes1, num_imgs_syn), (src2, classes2, num_imgs_orig)]:
            if class_name in classes:
                src_class_path = os.path.join(src, class_name)
                images_to_copy = min_images if src == src1 and min_images is not None else num_imgs
                
                available_files = os.listdir(src_class_path)
                
                if src == src1 and num_imgs_syn and len(available_files) < num_imgs_syn:
                    available_files *= (num_imgs_syn // len(available_files))
                    available_files += random.sample(os.listdir(src_class_path), num_imgs_syn % len(available_files))

                for i, file in enumerate(available_files):
                    if images_to_copy is not None and i >= images_to_copy:
                        break

                    existing_files = os.listdir(dest_class_path)
                    new_file = f"{file[:-4]}_copy{i}{file[-4:]}" if file in existing_files else file
                    shutil.copy(os.path.join(src_class_path, file), os.path.join(dest_class_path, new_file))

                    if src == src2:
                        class_counter[class_name] += 1
        
        if len(os.listdir(dest_class_path)) == 0:
            os.rmdir(dest_class_path)
            
    for class_name in remaining_classes2:
        src_class_path = os.path.join(src2, class_name)
        dest_class_path = os.path.join(dest, class_name)
        shutil.copytree(src_class_path, dest_class_path)

if __name__ == "__main__":
    num_imgs_syn_list = [None]
    num_imgs_orig_list = [None]
    delta_list = [1, 1.2]

    for num_imgs_syn in num_imgs_syn_list:
        for num_imgs_orig in num_imgs_orig_list:
            for delta in delta_list:
                syn_name = 'all' if num_imgs_syn is None else num_imgs_syn
                orig_name = 'all' if num_imgs_orig is None else num_imgs_orig

                name = f'syn_{syn_name}_orig_{orig_name}_delta_{delta}'

                if delta == 'infty':
                    syn = "path/to/syn_train"  # Replace with your actual path
                else:
                    syn = f"path/to/cleaned/syn_train_threshold_{delta}"  # Replace with your actual path

                orig = "path/to/orig/cars196_reformated/images"  # Replace with your actual path

                dest = f"path/to/merged/{name}/images"  # Replace with your actual path

                merge_datasets(syn, orig, dest, num_imgs_syn=num_imgs_syn, num_imgs_orig=num_imgs_orig)
