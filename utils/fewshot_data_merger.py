import os
import shutil
from torchvision.datasets import ImageFolder

def merge_datasets(synthetic_data_dir, few_shot_data_dir, merged_dir):
    os.makedirs(merged_dir, exist_ok=True)
    
    synthetic_dataset = ImageFolder(synthetic_data_dir)
    few_shot_dataset = ImageFolder(few_shot_data_dir)

    synthetic_classes = synthetic_dataset.classes[:100]
    few_shot_classes = few_shot_dataset.classes[:100]
    remaining_few_shot_classes = few_shot_dataset.classes[100:]
    
    classes_to_merge = set(synthetic_classes + few_shot_classes)

    # Merge first 100 classes
    for class_name in classes_to_merge:
        merged_class_path = os.path.join(merged_dir, class_name)
        os.makedirs(merged_class_path, exist_ok=True)

        if class_name in synthetic_classes:
            synthetic_class_path = os.path.join(synthetic_data_dir, class_name)
            for file in os.listdir(synthetic_class_path):
                shutil.copy(os.path.join(synthetic_class_path, file), os.path.join(merged_class_path, file))

        if class_name in few_shot_classes:
            few_shot_class_path = os.path.join(few_shot_data_dir, class_name)
            for file in os.listdir(few_shot_class_path):
                new_file = f"copy_{file}" if file in os.listdir(merged_class_path) else file
                shutil.copy(os.path.join(few_shot_class_path, file), os.path.join(merged_class_path, new_file))

    # Copy remaining 100 classes from few-shot to merged
    for class_name in remaining_few_shot_classes:
        source_class_path = os.path.join(few_shot_data_dir, class_name)
        dest_class_path = os.path.join(merged_dir, class_name)
        shutil.copytree(source_class_path, dest_class_path)


if __name__ == "__main__":
    imgs = 0
    threshold = 'infty'


    # synthetic_data_dir = f"/path/to/datasets/syn_cleaned/threshold_{threshold}"
    # few_shot_data_dir = f"/path/to/datasets/fewshot/org/{imgs}/CUB_200_2011/images"
    # merged_dir = f"/path/to/datasets/fewshot/merged/imgs_{imgs}_T_{threshold}/CUB_200_2011/images"
    # merge_datasets(synthetic_data_dir, few_shot_data_dir, merged_dir)

    ##### for orig data merging; comment ubove
    synthetic_data_dir = f"/path/to/datasets/cycle2/syn"
    orig_data_dir = f"/path/to/datasets/orig/CUB_200_2011/images"
    merged_dir = f"/path/to/datasets/cycle/merged/imgs_{imgs}_T_{threshold}/CUB_200_2011/images"
    merge_datasets(synthetic_data_dir, orig_data_dir, merged_dir)
