import os
import shutil

def merge_directories(base, target):
    if not os.path.exists(base):
        os.makedirs(base)

    for root, dirs, files in os.walk(target):
        for file in files:
            target_file = os.path.join(root, file)
            relative_path = os.path.relpath(root, target)
            base_subdir = os.path.join(base, relative_path)

            if not os.path.exists(base_subdir):
                os.makedirs(base_subdir)

            base_file = os.path.join(base_subdir, file)

            while os.path.exists(base_file):
                base_name, extension = os.path.splitext(base_file)
                base_file = f"{base_name}_copy{extension}"

            shutil.copy(target_file, base_file)


if __name__ == "__main__":
    base = '/path/to/datasets/base'
    target = '/path/to/datasets/target'
    
    merge_directories(base, target)