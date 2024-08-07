import os
import shutil
import scipy.io as sio

def sanitize_name(name):
    """Sanitize the class name by replacing or removing problematic characters."""
    for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(char, '_')
    return name

def create_class_directories(dest_path, class_names):
    for i, name in enumerate(class_names, start=1):
        dir_name = f"{i:03d}.{sanitize_name(name.replace(' ', '_'))}"
        os.makedirs(os.path.join(dest_path, dir_name), exist_ok=True)

def copy_images_to_class_dirs(base_path, dest_path, annotations, class_names, is_train=True):
    for anno in annotations:
        original_image_name = anno[5][0]
        class_idx = anno['class'][0][0] - 1  # MATLAB indices start at 1
        class_name = sanitize_name(class_names[class_idx].replace(' ', '_'))
        src_path = os.path.join(base_path, 'cars_train' if is_train else 'cars_test', original_image_name)
        dest_image_name = f"test_{original_image_name}" if not is_train else original_image_name
        dest_dir = os.path.join(dest_path, f"{class_idx+1:03d}.{class_name}")
        dest_file_path = os.path.join(dest_dir, dest_image_name)

        if not os.path.exists(src_path):
            print(f"File not found: {src_path}")
            continue

        try:
            shutil.copy2(src_path, dest_file_path)
        except Exception as e:
            print(f"Error copying file {src_path} to {dest_file_path}: {e}")

def main():
    dataset_path = 'path/to/raw/cars196'  # Replace with your actual dataset path
    dest_path = 'path/to/cars196_reformatted'  # Replace with your actual destination path
    cars_meta = sio.loadmat(os.path.join(dataset_path, 'cars_meta.mat'))
    cars_train_annos = sio.loadmat(os.path.join(dataset_path, 'cars_train_annos.mat'))
    cars_test_annos_withlabels = sio.loadmat(os.path.join(dataset_path, 'cars_test_annos_withlabels.mat'))

    class_names = [sanitize_name(cn[0].replace(' ', '_')) for cn in cars_meta['class_names'][0]]
    create_class_directories(dest_path, class_names)
    copy_images_to_class_dirs(dataset_path, dest_path, cars_train_annos['annotations'][0], class_names, is_train=True)
    copy_images_to_class_dirs(dataset_path, dest_path, cars_test_annos_withlabels['annotations'][0], class_names, is_train=False)

if __name__ == "__main__":
    main()
