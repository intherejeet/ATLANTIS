import os
import json
import shutil

# Replace these paths with the appropriate paths on your system
json_file_path = 'path/to/category_mapping.json'  # Replace with your actual JSON file path

# Define root folder paths
# root_folder_path = 'path/to/fullshot/split1/train'  # Uncomment and replace with your actual fullshot path
root_folder_path = 'path/to/syn_train_threshold_1.5'  # Replace with your actual root folder path

# Define new root folder path
# new_root_folder_path = 'path/to/domain_exps/orig/fullshot/train'  # Uncomment and replace with your actual new root folder path
new_root_folder_path = 'path/to/domain_exps/syn/zeroshot'  # Replace with your actual new root folder path

# Load the mapping from the JSON file
with open(json_file_path, 'r') as json_file:
    class_to_domain_mapping = json.load(json_file)

# Create the new directory structure and copy the images
for class_name, domain_name in class_to_domain_mapping.items():
    # Define the source and destination paths
    class_folder_path = os.path.join(root_folder_path, class_name)
    domain_folder_path = os.path.join(new_root_folder_path, domain_name, class_name)
    
    # Check if the class folder exists to prevent errors
    if not os.path.exists(class_folder_path):
        print(f"Warning: The folder for class '{class_name}' does not exist. Skipping.")
        continue
    
    # Create the domain and class folders in the new structure if they don't exist
    os.makedirs(domain_folder_path, exist_ok=True)
    
    # Copy each image from the class folder to the new domain/class folder
    for image_name in os.listdir(class_folder_path):
        # Ensure that we're only working with .jpg files
        if image_name.lower().endswith('.jpg'):
            source_image_path = os.path.join(class_folder_path, image_name)
            destination_image_path = os.path.join(domain_folder_path, image_name)
            shutil.copy2(source_image_path, destination_image_path)

print("Reorganization complete.")
