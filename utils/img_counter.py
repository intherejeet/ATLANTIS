import os

def count_images_in_directory(directory):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    image_count = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_count += 1

    return image_count

# Example usage
directory_path = '/path/to/robustretrieval/data_generator/data/domainwise/unknown'  # Replace with your directory path
total_images = count_images_in_directory(directory_path)
print(f"Total number of image files: {total_images}")
