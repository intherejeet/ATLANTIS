import os
import shutil
import random

def augment_data(src_data_path, augment_data_path, target_data_path, num_augment_images):
    # Ensure the target directory exists
    os.makedirs(target_data_path, exist_ok=True)

    # Step 1: Copy src_data_path to target_data_path
    for class_folder in os.listdir(src_data_path):
        shutil.copytree(os.path.join(src_data_path, class_folder), 
                        os.path.join(target_data_path, class_folder))

    # Step 2: Identify classes with only two images in target_data_path
    classes_with_two_images = []
    for class_folder in os.listdir(target_data_path):
        class_path = os.path.join(target_data_path, class_folder)
        if len(os.listdir(class_path)) == 2:
            classes_with_two_images.append(class_folder)

    # Step 3: Augment images from augment_data_path to target_data_path
    for class_folder in classes_with_two_images:
        src_class_path = os.path.join(augment_data_path, class_folder)
        target_class_path = os.path.join(target_data_path, class_folder)

        # Calculate the number of images to copy
        current_image_count = len(os.listdir(target_class_path))
        images_to_add = min(num_augment_images - current_image_count, len(os.listdir(src_class_path)))

        # List all images in the source class folder
        all_images = os.listdir(src_class_path)

        # Randomly select images to copy
        images_to_copy = random.sample(all_images, images_to_add) if images_to_add > 0 else []

        # Copy selected images
        for image in images_to_copy:
            src_image_path = os.path.join(src_class_path, image)
            target_image_path = os.path.join(target_class_path, image)
            shutil.copy2(src_image_path, target_image_path)

    print(f"Data augmentation completed. Classes with two images augmented with up to {num_augment_images} images each.")

# Example usage
num_classes_to_select = 75
num_samples_per_class = 2 
num_augment_images = 32  # Upper limit of images in each class after augmentation

# fullshot
# src_data_path = "path/to/fullshot/orig_imbalanced/classes"
# augment_data_path = "path/to/cleaned/syn_all_threshold_1.5"
# target_data_path = "path/to/augmented/fullshot_classes_threshold_images"

# zeroshot
src_data_path = "path/to/zeroshot/orig_imbalanced/classes"
augment_data_path = "path/to/cleaned/syn_train_threshold_1.5"
target_data_path = "path/to/augmented/zeroshot_classes_threshold_images"

augment_data(src_data_path, augment_data_path, target_data_path, num_augment_images)
