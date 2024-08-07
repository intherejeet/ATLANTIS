import os
import random
import shutil

def create_imbalanced_dataset(original_dataset_path, new_dataset_path, synthetic_data_path, num_classes_to_select, num_samples_per_class, seed=42):
    def is_image(file_name):
        return any(file_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])

    # Set the seed for reproducibility
    random.seed(seed)

    # Count images in each class in synthetic data and sort classes by image count
    synthetic_classes = os.listdir(synthetic_data_path)
    class_image_counts = []
    for class_folder in synthetic_classes:
        class_path = os.path.join(synthetic_data_path, class_folder)
        image_count = len([img for img in os.listdir(class_path) if is_image(img)])
        class_image_counts.append((class_folder, image_count))

    # Sort classes based on the number of images (descending order)
    class_image_counts.sort(key=lambda x: x[1], reverse=True)

    # Select top 'num_classes_to_select' classes
    selected_classes = [class_name for class_name, _ in class_image_counts[:num_classes_to_select]]
    
    # Print the names and the number of images in the selected classes
    print(f"Selected {num_classes_to_select} classes for imbalance:")
    for class_name, image_count in class_image_counts[:num_classes_to_select]:
        print(f"Class: {class_name}, Images: {image_count}")
    
    # Print the number of images in the x-th class
    if num_classes_to_select <= len(class_image_counts):
        xth_class_images = class_image_counts[num_classes_to_select-1][1]
        print(f"The {num_classes_to_select}-th class has {xth_class_images} images.")

    # Ensure the new dataset directory exists
    os.makedirs(new_dataset_path, exist_ok=True)

    # Copy and create imbalance in the selected classes
    for class_folder in os.listdir(original_dataset_path):
        class_source_path = os.path.join(original_dataset_path, class_folder)
        class_dest_path = os.path.join(new_dataset_path, class_folder)
        try:
            if class_folder in selected_classes:
                # If class is selected, copy and keep only a limited number of images
                os.makedirs(class_dest_path, exist_ok=True)
                all_images = [img for img in os.listdir(class_source_path) if is_image(img)]
                kept_images = random.sample(all_images, min(len(all_images), num_samples_per_class))
                for image in kept_images:
                    shutil.copy2(os.path.join(class_source_path, image), class_dest_path)
            else:
                # If class is not selected, copy the entire folder
                shutil.copytree(class_source_path, class_dest_path)
        except Exception as e:
            print(f"Error processing {class_folder}: {e}")

    print(f"Imbalanced dataset created based on top {num_classes_to_select} classes from synthetic data.")


num_classes_to_select = 75  # Number of classes to select
num_samples_per_class = 2  # Number of samples per selected class
seed = 42  # Seed for reproducibility

# fullshot
# original_dataset_path = "path/to/fullshot/orig/split1/train"
# synthetic_data_path = "path/to/cleaned/syn_all_threshold_1.5"
# new_dataset_path = "path/to/orig_imbalanced/fullshot_classes"

# zeroshot
original_dataset_path = "path/to/zeroshot/orig/train"
synthetic_data_path = "path/to/cleaned/syn_train_threshold_1.5"
new_dataset_path = "path/to/orig_imbalanced/zeroshot_classes"

create_imbalanced_dataset(original_dataset_path, new_dataset_path, synthetic_data_path, num_classes_to_select, num_samples_per_class, seed)
